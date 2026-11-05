"""평가 데이터 검사.  python3 data/check_eval.py data/eval_raw/X.jsonl [...]

검사: JSON 형식, 구간이 원문에 글자 그대로 있는지, 앞뒤 공백, 규칙 ID 존재(rules.json + fixed_labels.json
+ eval_raw/*.rules.json), 학습·검증 데이터와 같은 문장, 파일 안·파일 간 중복.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def known_ids():
    ids = {r["id"] for r in json.loads((HERE / "rules.json").read_text())["rules"]}
    ids |= {r["id"] for r in json.loads((HERE / "fixed_labels.json").read_text())["labels"]}
    for p in (HERE / "eval_raw").glob("*.rules.json"):
        ids |= {r["id"] for r in json.loads(p.read_text())["rules"]}
    return ids


def main(paths):
    ids = known_ids()
    train_texts = {json.loads(l)["text"] for n in ("train", "dev") for l in (HERE / f"{n}.jsonl").read_text().splitlines()}
    seen, ok = {}, True
    for p in map(Path, paths):
        errs, cnt, neg = [], Counter(), 0
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                text, spans = d["text"], d["spans"]
            except Exception as e:
                errs.append(f"{n}행 JSON 오류: {e}")
                continue
            if text in train_texts:
                errs.append(f"{n}행 학습 데이터와 같은 문장")
            if text in seen and seen[text] != p.name:
                errs.append(f"{n}행 {seen[text]} 와 같은 문장")
            elif text in seen:
                errs.append(f"{n}행 중복 문장")
            seen[text] = p.name
            neg += not spans
            for s in spans:
                if len(s) != 2:
                    errs.append(f"{n}행 구간 형식 오류 {s}")
                    continue
                sub, rid = s
                cnt[rid] += 1
                if rid not in ids:
                    errs.append(f"{n}행 없는 ID {rid}")
                if not sub or sub not in text:
                    errs.append(f"{n}행 구간이 원문에 없음: '{sub}'")
                elif sub.strip() != sub:
                    errs.append(f"{n}행 구간 앞뒤 공백: '{sub}'")
        print(f"[{p.name}] 문장 {sum(1 for l in p.read_text().splitlines() if l.strip())} / 구간 {sum(cnt.values())} / 음성 문장 {neg}")
        print("  ID별:", dict(sorted(cnt.items())))
        for e in errs[:30]:
            print("  오류:", e)
        ok &= not errs
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
