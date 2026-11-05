"""생성 데이터 검사.  python3 data/check_raw.py data/train_raw/X.samples.jsonl [...]

검사: JSON 형식, 구간이 원문에 글자 그대로 있는지, 규칙 ID 존재(고정 라벨 P-* 포함), 평가셋(v1·v2) 문장 중복,
heldout 개념 누수 의심(경고만), 규칙별 양성/음성 개수.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
LEAK = re.compile(r"이사회|리콜|불량률|결함|해지|이탈|출장|동선|가격 ?인상|요금 ?인상|인상 ?예정|실험 결과|"
                  r"논문|NDA|비밀유지|장애|포스트모템|postmortem", re.I)


def rule_ids(samples_path: Path):
    ids = {r["id"] for r in json.loads((HERE / "rules.json").read_text())["rules"] if r["split"] == "train"}
    ids |= {r["id"] for r in json.loads((HERE / "fixed_labels.json").read_text())["labels"]}
    if samples_path.name.startswith("pii_"):  # 개인정보 전용 파일은 모든 업종 기밀 규칙과 섞을 수 있다
        ids |= {r["id"] for r in json.loads((HERE / "rule_bank.json").read_text())}
    rp = samples_path.with_name(samples_path.name.replace(".samples.jsonl", ".rules.json"))
    if rp.exists():
        ids |= {r["id"] for r in json.loads(rp.read_text())["rules"]}
    return ids


def check(path: Path):
    ids = rule_ids(path)
    eval_texts = {json.loads(l)["text"] for f in ("eval.jsonl", "eval_v2.jsonl") if (HERE / f).exists()
                  for l in (HERE / f).read_text().splitlines()}
    errs, leaks = [], []
    pos, neg = Counter(), Counter()
    seen = set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            text, spans = d["text"], d["spans"]
        except Exception as e:
            errs.append(f"{n}행 JSON 오류: {e}")
            continue
        if text in eval_texts:
            errs.append(f"{n}행 평가셋과 같은 문장")
        if text in seen:
            errs.append(f"{n}행 중복 문장")
        seen.add(text)
        for s in spans:
            if len(s) != 2:
                errs.append(f"{n}행 구간 형식 오류 {s}")
                continue
            sub, rid = s
            if rid not in ids:
                errs.append(f"{n}행 없는 규칙 ID {rid}")
            if sub not in text:
                errs.append(f"{n}행 구간이 원문에 없음: '{sub}'")
            elif sub.strip() != sub or not sub:
                errs.append(f"{n}행 구간 앞뒤 공백: '{sub}'")
            pos[rid] += 1
        if not spans:
            for rid in d.get("near", []) or ["?"]:
                neg[rid] += 1
        if spans and LEAK.search(text):
            leaks.append(f"{n}행 {text[:50]}")
    print(f"[{path.name}] 문장 {len(seen)} / 양성 구간 {sum(pos.values())} / 음성 문장 {sum(neg.values())} / 규칙 {len(pos)}")
    thin = [r for r in ids if r in pos or r in neg if pos[r] < 6]
    if thin:
        print("  양성 6개 미만 규칙:", thin)
    for e in errs[:30]:
        print("  오류:", e)
    for l in leaks[:10]:
        print("  누수 의심(heldout 개념):", l)
    return not errs


if __name__ == "__main__":
    ok = all([check(Path(p)) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
