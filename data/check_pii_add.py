"""기존 학습 문장에 붙인 개인정보 라벨 검사.  python3 data/check_pii_add.py data/train_raw/X.pii.jsonl [...]

한 줄: {"text": "<X.samples.jsonl 의 원문 그대로>", "add": [["구간", "P-NAME|P-ADDR|P-SENS"], ...]}
검사: 원문이 같은 이름의 samples 파일에 있는지, 구간이 원문에 있는지, ID, 중복 줄.
"""
import json
import sys
from collections import Counter
from pathlib import Path

FIXED = {"P-NAME", "P-ADDR", "P-SENS"}


def check(path: Path):
    src = path.with_name(path.name.replace(".pii.jsonl", ".samples.jsonl"))
    texts = {json.loads(l)["text"] for l in src.read_text().splitlines() if l.strip()}
    errs, cnt, seen = [], Counter(), set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            text, add = d["text"], d["add"]
        except Exception as e:
            errs.append(f"{n}행 JSON 오류: {e}")
            continue
        if text not in texts:
            errs.append(f"{n}행 원문이 {src.name} 에 없음: {text[:40]}")
        if text in seen:
            errs.append(f"{n}행 중복")
        seen.add(text)
        for s in add:
            if len(s) != 2 or s[1] not in FIXED:
                errs.append(f"{n}행 형식/ID 오류 {s}")
                continue
            cnt[s[1]] += 1
            if not s[0] or s[0] not in text or s[0].strip() != s[0]:
                errs.append(f"{n}행 구간 문제: '{s[0]}'")
    print(f"[{path.name}] 문장 {len(seen)} / {dict(cnt)}")
    for e in errs[:30]:
        print("  오류:", e)
    return not errs


if __name__ == "__main__":
    sys.exit(0 if all([check(Path(p)) for p in sys.argv[1:]]) else 1)
