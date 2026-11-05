"""기밀 구간 탐지 모델 평가.

  python evaluate.py --model keyword                 # 하네스 확인용 바보 기준선
  python evaluate.py --model gliner --name urchade/gliner_multi-v2.1 [--desc]

지표 (마스킹은 '놓치지 않기'가 우선이므로 재현율을 먼저 본다):
  재현율      정답 구간 중 예측과 겹친 비율 (규칙 ID 무관 / 규칙 ID 일치)
  정밀도      예측 구간 중 정답과 겹친 비율
  오탐 문장률  정답이 없는 문장에서 하나라도 잡은 비율
  train/heldout 은 '학습에 쓴 규칙'과 '처음 보는 규칙'을 나눠 본 재현율
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

DATA = Path(__file__).parent / "data"


def load():
    rules = json.loads((DATA / "rules.json").read_text())["rules"]
    items = []
    for i, line in enumerate((DATA / "eval.jsonl").read_text().splitlines()):
        d = json.loads(line)
        spans = []
        for s, rid in d["spans"]:
            st = d["text"].find(s)
            assert st >= 0, f"{i + 1}행: '{s}' 가 원문에 없음"
            spans.append((st, st + len(s), rid))
        items.append({"text": d["text"], "spans": spans})
    return rules, items


# ------------------------------------------------------------------ 모델 어댑터
# predict(text) -> [(start, end, rule_id, score)]

class KeywordBaseline:
    """규칙 이름의 단어가 문장에 있으면 그 단어를 잡는다. 하네스 검증용."""

    def __init__(self, rules, **_):
        self.rules = rules

    def predict(self, text):
        out = []
        for r in self.rules:
            for w in r["label"].replace("·", " ").split():
                if len(w) >= 2 and w in text:
                    st = text.find(w)
                    out.append((st, st + len(w), r["id"], 0.5))
        return out


class GlinerModel:
    def __init__(self, rules, name="urchade/gliner_multi-v2.1", desc=False, threshold=0.4, **_):
        from gliner import GLiNER
        self.m = GLiNER.from_pretrained(name)
        self.threshold = threshold
        # train.py 로 학습한 모델은 학습 때 쓴 분리기·라벨 형식을 그대로 따른다
        meta = Path(name) / "train_meta.json"
        if meta.exists():
            from splitter import EojeolSplitter
            self.m.data_processor.words_splitter = EojeolSplitter()
            desc = json.loads(meta.read_text())["desc"]
        # GLiNER 는 라벨 문자열을 입력으로 받는다. --desc 면 '이름: 설명' 을 라벨로 쓴다
        self.label2id = {(f"{r['label']}: {r['description']}" if desc else r["label"]): r["id"] for r in rules}

    def predict(self, text):
        ents = self.m.predict_entities(text, list(self.label2id), threshold=self.threshold)
        return [(e["start"], e["end"], self.label2id[e["label"]], e["score"]) for e in ents]


MODELS = {"keyword": KeywordBaseline, "gliner": GlinerModel}


# ------------------------------------------------------------------ 채점
def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def evaluate(model, rules, items):
    split = {r["id"]: r["split"] for r in rules}
    hit_any = hit_rule = gold_n = 0
    by_split = {"train": [0, 0], "heldout": [0, 0]}
    pred_n = pred_ok = 0
    neg_n = neg_fp = 0
    lat = []
    misses, fps = [], []
    for it in items:
        t0 = time.perf_counter()
        preds = model.predict(it["text"])
        lat.append((time.perf_counter() - t0) * 1000)
        for g in it["spans"]:
            gold_n += 1
            any_ = any(overlap(g, p) for p in preds)
            same = any(overlap(g, p) and p[2] == g[2] for p in preds)
            hit_any += any_
            hit_rule += same
            by_split[split[g[2]]][0] += any_
            by_split[split[g[2]]][1] += 1
            if not any_:
                misses.append((it["text"][g[0]:g[1]], g[2]))
        for p in preds:
            pred_n += 1
            ok = any(overlap(p, g) for g in it["spans"])
            pred_ok += ok
            if not ok:
                fps.append((it["text"][p[0]:p[1]], p[2]))
        if not it["spans"]:
            neg_n += 1
            neg_fp += bool(preds)

    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  -  "
    print(f"재현율(구간 겹침)      {pct(hit_any, gold_n)}  ({hit_any}/{gold_n})")
    print(f"재현율(규칙ID도 일치)  {pct(hit_rule, gold_n)}")
    print(f"  ├ 학습 규칙          {pct(*by_split['train'])}")
    print(f"  └ 처음 보는 규칙      {pct(*by_split['heldout'])}")
    print(f"정밀도                 {pct(pred_ok, pred_n)}  ({pred_ok}/{pred_n})")
    print(f"오탐 문장률            {pct(neg_fp, neg_n)}  ({neg_fp}/{neg_n})")
    print(f"지연 p50 {statistics.median(lat):.0f}ms / p95 {sorted(lat)[int(len(lat) * .95)]:.0f}ms")
    print("\n놓친 구간 예:", misses[:8])
    print("오탐 예:", fps[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="keyword", choices=MODELS)
    ap.add_argument("--name", default="urchade/gliner_multi-v2.1")
    ap.add_argument("--desc", action="store_true", help="라벨에 규칙 설명까지 넣기")
    ap.add_argument("--threshold", type=float, default=0.4)
    a = ap.parse_args()
    rules, items = load()
    model = MODELS[a.model](rules, name=a.name, desc=a.desc, threshold=a.threshold)
    model.predict(items[0]["text"])  # 첫 호출 워밍업은 지연 측정에서 뺀다
    evaluate(model, rules, items)


if __name__ == "__main__":
    main()
