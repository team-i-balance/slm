"""기밀·개인정보 구간 탐지 모델 평가.

  python evaluate.py --model keyword                       # 하네스 확인용 바보 기준선
  python evaluate.py --model gliner --name models/v1/final --threshold 0.45 [--pii-threshold 0.3]
  python evaluate.py ... --eval data/eval.jsonl            # v1 평가셋(104문장)

평가셋 한 줄: {"text", "spans": [[구간, ID]], "set": "moa"|"logi"|"pii"}
  문장마다 그 회사의 정책 규칙 + 고정 라벨(P-*)을 모델에 넣는다 (moa·pii → 모아랩스 규칙, logi → 누리로지스 규칙)

지표 (마스킹은 '놓치지 않기'가 우선이라 재현율을 먼저 본다)
  재현율        정답 구간 중 같은 그룹(기밀/개인정보) 예측과 겹친 비율
                기밀은 학습 규칙(R) / 처음 보는 규칙(H) / 처음 보는 회사(L), 개인정보는 P-NAME·P-ADDR·P-SENS 로 나눠 본다
  정밀도        예측 구간 중 같은 그룹 정답과 겹친 비율
  오탐 문장률    정답이 없는 문장에서 하나라도 잡은 비율
  문장 정확도    '가릴 게 있다/없다' 판정이 맞은 문장 비율
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"


def load_rules():
    """{정책 이름: 규칙 목록}. 고정 라벨은 모든 정책에 붙는다."""
    fixed = json.loads((DATA / "fixed_labels.json").read_text())["labels"]
    policies = {"moa": json.loads((DATA / "rules.json").read_text())["rules"]}
    # 처음 보는 회사 정책: eval_raw/<set>.rules.json (logi, 업종별 food·edu·… 등)
    for p in sorted((DATA / "eval_raw").glob("*.rules.json")):
        policies[p.name.removesuffix(".rules.json")] = json.loads(p.read_text())["rules"]
    policies["pii"] = policies["moa"]
    return {k: v + fixed for k, v in policies.items()}


def load_items(path):
    items = []
    for i, line in enumerate(path.read_text().splitlines()):
        d = json.loads(line)
        spans, used = [], defaultdict(int)
        for s, rid in d["spans"]:
            # 같은 문자열이 여러 번이면 앞에서부터 차례로 (같은 구간에 ID 두 개면 같은 위치)
            st = -1
            for _ in range(used[(s, rid)] + 1):
                st = d["text"].find(s, st + 1)
            assert st >= 0, f"{path.name} {i + 1}행: '{s}' 가 원문에 없음"
            used[(s, rid)] += 1
            spans.append((st, st + len(s), rid))
        items.append({"text": d["text"], "spans": spans, "set": d.get("set", "moa")})
    return items


def group(rid):
    if rid.startswith("P-"):
        return rid
    return {"R": "기밀-학습 규칙", "H": "기밀-처음 보는 규칙", "L": "기밀-처음 보는 회사"}[rid[0]]


def kind(rid):
    return "개인정보" if rid.startswith("P-") else "기밀"


# ------------------------------------------------------------------ 모델 어댑터
# predict(text, rules) -> [(start, end, rule_id, score)]

class KeywordBaseline:
    """규칙 이름의 단어가 문장에 있으면 그 단어를 잡는다. 하네스 검증용."""

    def __init__(self, **_):
        pass

    def predict(self, text, rules):
        out = []
        for r in rules:
            for w in r["label"].replace("·", " ").split():
                if len(w) >= 2 and w in text:
                    st = text.find(w)
                    out.append((st, st + len(w), r["id"], 0.5))
        return out


class GlinerModel:
    def __init__(self, name="urchade/gliner_multi-v2.1", desc=False, threshold=0.4, pii_threshold=None,
                 nested=True, onnx=None, refine=False, **_):
        from gliner import GLiNER
        if onnx:
            self.m = GLiNER.from_pretrained(name, local_files_only=True, load_onnx_model=True, onnx_model_file=onnx)
        else:
            self.m = GLiNER.from_pretrained(name)
        self.th, self.pii_th = threshold, pii_threshold if pii_threshold is not None else threshold
        self.nested = nested
        self.refine = None
        if refine:
            from splitter import EojeolSplitter
            self.refine = EojeolSplitter()
        # train.py 로 학습한 모델은 학습 때 쓴 분리기·라벨 형식을 그대로 따른다
        meta = Path(name) / "train_meta.json"
        self.desc = desc
        if meta.exists():
            from splitter import EojeolSplitter
            self.m.data_processor.words_splitter = EojeolSplitter()
            self.desc = json.loads(meta.read_text())["desc"]

    def predict(self, text, rules):
        # GLiNER 는 라벨 문자열을 입력으로 받는다. desc 면 '이름: 설명' 을 라벨로 쓴다
        label2id = {}
        for r in rules:
            label2id.setdefault(f"{r['label']}: {r['description']}" if self.desc else r["label"], r["id"])
        ents = self.m.predict_entities(text, list(label2id), threshold=min(self.th, self.pii_th),
                                       flat_ner=not self.nested, multi_label=self.nested)
        out = []
        for e in ents:
            rid = label2id[e["label"]]
            if e["score"] >= (self.pii_th if rid.startswith("P-") else self.th):
                st, en = e["start"], e["end"]
                if self.refine:
                    from detector import refine_span
                    st, en = refine_span(text, st, en, rid, self.refine)
                out.append((st, en, rid, e["score"]))
        return out


MODELS = {"keyword": KeywordBaseline, "gliner": GlinerModel}


# ------------------------------------------------------------------ 채점
def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def evaluate(model, policies, items):
    rec = defaultdict(lambda: [0, 0])     # 그룹 → [맞음, 전체]
    by_set = defaultdict(lambda: [0, 0])  # set → 기밀 [맞음, 전체]
    exact = defaultdict(lambda: [0, 0])   # 기밀/개인정보 → [경계까지 같음, 잡은 정답 구간]
    rule_ok = [0, 0]                      # 기밀 구간 규칙 ID 일치
    prec = defaultdict(lambda: [0, 0])    # 기밀/개인정보 → [맞음, 예측 수]
    neg = defaultdict(lambda: [0, 0])     # set → [오탐 문장, 음성 문장]
    acc = [0, 0]
    # 등급 판정: 문장 등급 = 걸린 규칙 grade 중 가장 높은 것 (C > S > O)
    order = ["O", "S", "C"]
    grade_of = {r["id"]: r.get("grade", "S") for rules in policies.values() for r in rules}
    sent_grade = lambda spans: max((grade_of[s[2]] for s in spans), key=order.index, default="O")
    gconf = defaultdict(int)              # (정답 등급, 예측 등급) → 문장 수
    lat, misses, fps = [], defaultdict(list), defaultdict(list)
    for it in items:
        t0 = time.perf_counter()
        preds = model.predict(it["text"], policies[it["set"]])
        lat.append((time.perf_counter() - t0) * 1000)
        for g in it["spans"]:
            hit = any(overlap(g, p) and kind(p[2]) == kind(g[2]) for p in preds)
            rec[group(g[2])][0] += hit
            if hit:
                exact[kind(g[2])][0] += any(p[0] == g[0] and p[1] == g[1] and kind(p[2]) == kind(g[2]) for p in preds)
                exact[kind(g[2])][1] += 1
            rec[group(g[2])][1] += 1
            if kind(g[2]) == "기밀":
                by_set[it["set"]][0] += hit
                by_set[it["set"]][1] += 1
                rule_ok[0] += any(overlap(g, p) and p[2] == g[2] for p in preds)
                rule_ok[1] += 1
            if not hit:
                misses[group(g[2])].append(it["text"][g[0]:g[1]])
        for p in preds:
            ok = any(overlap(p, g) and kind(p[2]) == kind(g[2]) for g in it["spans"])
            prec[kind(p[2])][0] += ok
            prec[kind(p[2])][1] += 1
            if not ok:
                fps[kind(p[2])].append((it["text"][p[0]:p[1]], p[2]))
        if not it["spans"]:
            neg[it["set"]][0] += bool(preds)
            neg[it["set"]][1] += 1
        acc[0] += bool(preds) == bool(it["spans"])
        acc[1] += 1
        gconf[(sent_grade(it["spans"]), sent_grade(preds))] += 1

    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  -  "
    tot = lambda keys: [sum(rec[k][i] for k in keys) for i in (0, 1)]
    conf = [k for k in rec if not k.startswith("P-")]
    pii = [k for k in rec if k.startswith("P-")]
    print(f"문장 {len(items)} / 정답 구간 {sum(v[1] for v in rec.values())}")
    print(f"기밀 재현율             {pct(*tot(conf))}  ({tot(conf)[0]}/{tot(conf)[1]})   규칙ID 일치 {pct(*rule_ok)}")
    for k in sorted(conf):
        print(f"  ├ {k:16s} {pct(*rec[k])}  ({rec[k][0]}/{rec[k][1]})")
    if len(by_set) > 1:
        print("  묶음별 기밀 재현율: " + ", ".join(f"{k} {pct(*v).strip()}({v[0]}/{v[1]})" for k, v in sorted(by_set.items())))
    if pii:
        print(f"개인정보 재현율         {pct(*tot(pii))}  ({tot(pii)[0]}/{tot(pii)[1]})")
        for k in sorted(pii):
            print(f"  ├ {k:16s} {pct(*rec[k])}  ({rec[k][0]}/{rec[k][1]})")
    for k in ("기밀", "개인정보"):
        if prec[k][1]:
            print(f"{k} 정밀도{' ' * (16 - 2 * len(k))}{pct(*prec[k])}  ({prec[k][0]}/{prec[k][1]})")
    print("경계 정확 일치         " + ", ".join(f"{k} {pct(*v).strip()} ({v[0]}/{v[1]})" for k, v in sorted(exact.items()))
          + "   (잡은 정답 구간 중 시작·끝까지 같은 예측이 있는 비율)")
    allneg = [sum(v[i] for v in neg.values()) for i in (0, 1)]
    detail = ", ".join(f"{k} {pct(*v).strip()}" for k, v in sorted(neg.items()))
    print(f"오탐 문장률             {pct(*allneg)}  ({allneg[0]}/{allneg[1]})   [{detail}]")
    print(f"문장 정확도             {pct(*acc)}")
    gok = sum(v for (g, p), v in gconf.items() if g == p)
    print(f"등급 판정 정확도        {pct(gok, acc[1])}   (행=정답, 열=예측 O/S/C)")
    for g in order:
        row = [gconf[(g, p)] for p in order]
        if sum(row):
            print(f"  {g}: {row}   맞음 {pct(row[order.index(g)], sum(row)).strip()}")
    print(f"지연 p50 {statistics.median(lat):.0f}ms / p95 {sorted(lat)[int(len(lat) * .95)]:.0f}ms")
    print("\n놓친 구간 예:")
    for k in sorted(misses):
        print(f"  {k}: {misses[k][:6]}")
    print("오탐 예:")
    for k in sorted(fps):
        print(f"  {k}: {fps[k][:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="keyword", choices=MODELS)
    ap.add_argument("--name", default="urchade/gliner_multi-v2.1")
    ap.add_argument("--desc", action="store_true", help="라벨에 규칙 설명까지 넣기")
    ap.add_argument("--threshold", type=float, default=0.4, help="기밀 기준값")
    ap.add_argument("--pii-threshold", type=float, default=None, help="개인정보 기준값 (기본: 기밀과 같음)")
    ap.add_argument("--flat", action="store_true", help="겹치는 구간 허용 안 함 (v1 평가 방식)")
    ap.add_argument("--eval", default=None, help="평가 파일 (기본: eval_v2.jsonl, 없으면 eval.jsonl)")
    ap.add_argument("--set", default=None, help="moa / logi / pii 중 하나만")
    ap.add_argument("--refine", action="store_true", help="구간 경계 후처리(detector.refine_span) 적용")
    ap.add_argument("--onnx", default=None, help="ONNX 파일 이름 (예: model_quantized.onnx). --name 은 그 폴더")
    a = ap.parse_args()
    path = Path(a.eval) if a.eval else (DATA / "eval_v2.jsonl" if (DATA / "eval_v2.jsonl").exists() else DATA / "eval.jsonl")
    policies, items = load_rules(), load_items(path)
    if a.set:
        items = [it for it in items if it["set"] == a.set]
    model = MODELS[a.model](name=a.name, desc=a.desc, threshold=a.threshold, pii_threshold=a.pii_threshold,
                            nested=not a.flat, onnx=a.onnx, refine=a.refine)
    model.predict(items[0]["text"], policies[items[0]["set"]])  # 첫 호출 워밍업은 지연 측정에서 뺀다
    evaluate(model, policies, items)


if __name__ == "__main__":
    main()
