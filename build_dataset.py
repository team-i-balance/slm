"""data/train_raw/* → data/train.jsonl, data/dev.jsonl, data/rule_bank.json

출력 한 줄: {"text", "spans": [[start, end, rule_id]], "near": [...], "industry"}
단어 분리·라벨 문자열 구성은 학습 스크립트에서 한다 (분리기 실험을 위해 문자 위치로 저장).
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data"))
from check_raw import LEAK  # noqa: E402

DATA = Path(__file__).parent / "data"
RAW = DATA / "train_raw"
# 개인정보 규칙은 규칙 엔진(piiguard) 담당이라 SLM 학습에서 뺀다
EXCLUDE = {"COM-18", "CON-08", "CON-17", "GAME-13", "PUB-15"}


def char_spans(text, spans):
    """같은 문자열이 여러 번 나오면 앞에서부터 차례로 배정."""
    used, out = Counter(), []
    for sub, rid in spans:
        st, k = -1, used[sub]
        for _ in range(k + 1):
            st = text.find(sub, st + 1)
        if st < 0:
            st = text.find(sub)
        used[sub] += 1
        out.append([st, st + len(sub), rid])
    return out


def main():
    bank = {r["id"]: {**r, "industry": "모아랩스"} for r in json.loads((DATA / "rules.json").read_text())["rules"]
            if r["split"] == "train"}
    heldout_ids = {r["id"] for r in json.loads((DATA / "rules.json").read_text())["rules"] if r["split"] == "heldout"}
    for rp in sorted(RAW.glob("*.rules.json")):
        for r in json.loads(rp.read_text())["rules"]:
            assert r["id"] not in bank, f"규칙 ID 중복 {r['id']}"
            if r["id"] not in EXCLUDE:
                bank[r["id"]] = r
    eval_texts = {json.loads(l)["text"] for l in (DATA / "eval.jsonl").read_text().splitlines()}

    rows, seen, drop = [], set(), Counter()
    for sp in sorted(RAW.glob("*.samples.jsonl")):
        for line in sp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                drop["JSON 오류"] += 1
                continue
            t = d["text"]
            if t in seen or t in eval_texts:
                drop["중복/평가셋"] += 1
                continue
            if any(s[1] in EXCLUDE for s in d["spans"]):
                drop["개인정보 규칙"] += 1
                continue
            if any(len(s) != 2 or s[1] not in bank or s[0] not in t for s in d["spans"]):
                drop["구간/ID 오류"] += 1
                continue
            if any(s[1] in heldout_ids for s in d["spans"]):
                drop["heldout 규칙 사용"] += 1
                continue
            seen.add(t)
            rid0 = (d["spans"][0][1] if d["spans"] else (d.get("near") or ["?"])[0])
            rows.append({"text": t, "spans": char_spans(t, d["spans"]),
                         "near": [n for n in d.get("near", []) if n in bank],
                         "industry": bank.get(rid0, {}).get("industry", "?"),
                         "leak_suspect": bool(d["spans"] and LEAK.search(t))})

    random.Random(0).shuffle(rows)
    n_dev = max(50, len(rows) // 20)
    dev, train = rows[:n_dev], rows[n_dev:]
    for name, part in (("train", train), ("dev", dev)):
        with open(DATA / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DATA / "rule_bank.json").write_text(json.dumps(list(bank.values()), ensure_ascii=False, indent=1))

    pos = sum(bool(r["spans"]) for r in rows)
    print(f"규칙 {len(bank)}개 / 문장 {len(rows)} (양성 {pos}, 음성 {len(rows) - pos}) → train {len(train)}, dev {len(dev)}")
    print("업종별:", dict(Counter(r["industry"] for r in rows)))
    print("버린 것:", dict(drop))
    print("누수 의심 표시:", sum(r["leak_suspect"] for r in rows))
    lens = sorted(e - s for r in rows for s, e, _ in r["spans"])
    print(f"구간 길이(글자) 중앙값 {lens[len(lens) // 2]}, 95% {lens[int(len(lens) * .95)]}, 최대 {lens[-1]}")


if __name__ == "__main__":
    main()
