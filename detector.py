"""SLM 탐지기: 정책 규칙(회사 기밀) + 고정 라벨(이름·주소·개인 민감정보) 구간 탐지.

  from detector import SlmDetector
  det = SlmDetector("models/v2/final", policy="data/rules.json")
  det.detect("박서윤 대리 근태 불량으로 경고 징계 내렸어")
  # [{'start': 0, 'end': 3, 'type': 'NAME', 'rule_id': 'P-NAME', 'grade': 'S', 'action': '대체', 'score': 0.97}, ...]

출력 형식은 backend local_ai.detector.Detector.detect 와 같다 (start 포함, end 제외, Python 문자열 위치).
type: NAME / ADDRESS / SENSITIVE (고정 라벨), CONFIDENTIAL (정책 규칙). 정형 개인정보는 규칙 엔진 담당이라 없다.
네트워크·로그에 원문을 남기지 않는다. 모델은 로컬 폴더에서만 읽는다.

  python detector.py --model models/v2/final < 입력.txt     # 한 줄씩 탐지 결과 출력
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
FIXED_TYPES = {"P-NAME": "NAME", "P-ADDR": "ADDRESS", "P-SENS": "SENSITIVE"}
# 기준값: 평가셋 v2 에서 고른 값. 개인정보는 놓치지 않기 우선이라 더 낮다
DEFAULT_THRESHOLDS = {"CONFIDENTIAL": 0.45, "NAME": 0.4, "ADDRESS": 0.4, "SENSITIVE": 0.4}
# 한 번에 넣는 최대 단어 수. 규칙 라벨도 같은 입력에 들어가므로 모델 max_len(384)보다 넉넉히 작게
CHUNK_WORDS = 200
SENT_END = re.compile(r"(?<=[.!?。])\s+|\n+")


def load_policy(path):
    """rules.json 형식: {"rules": [{"id", "label", "description", "grade", "action", ...}]}"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = d["rules"] if isinstance(d, dict) else d
    return [r for r in rules if not r["id"].startswith("P-")]


class SlmDetector:
    def __init__(self, model_dir, policy=None, thresholds=None, device="cpu"):
        from gliner import GLiNER

        from splitter import EojeolSplitter
        path = Path(model_dir)
        if not (path / "gliner_config.json").is_file():
            raise FileNotFoundError(f"모델 폴더가 아님: {path}")
        self.m = GLiNER.from_pretrained(str(path), local_files_only=True, map_location=device)
        self.m.data_processor.words_splitter = EojeolSplitter()
        self.splitter = EojeolSplitter()
        meta = json.loads((path / "train_meta.json").read_text())
        self.desc = meta.get("desc", False)
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.fixed = json.loads((HERE / "data" / "fixed_labels.json").read_text())["labels"]
        self.set_policy(load_policy(policy) if policy else [])

    def set_policy(self, rules):
        """정책 규칙을 바꾼다. 재학습 없이 새 규칙이 바로 적용된다."""
        self.rules = {r["id"]: r for r in list(rules) + self.fixed}
        self.label2id = {}
        for r in self.rules.values():
            self.label2id.setdefault(f"{r['label']}: {r['description']}" if self.desc else r["label"], r["id"])

    def _chunks(self, text):
        """문장 경계로 끊어 CHUNK_WORDS 단어 이하 조각으로 묶는다. (시작 위치, 조각)"""
        pieces, pos = [], 0
        for m in SENT_END.finditer(text):
            pieces.append((pos, text[pos:m.start()]))
            pos = m.end()
        pieces.append((pos, text[pos:]))
        out, cur_st, cur_end, n = [], None, 0, 0
        for st, p in pieces:
            w = len(list(self.splitter(p)))
            if cur_st is not None and n + w > CHUNK_WORDS:
                out.append((cur_st, text[cur_st:cur_end]))
                cur_st, n = None, 0
            if cur_st is None:
                cur_st = st
            cur_end, n = st + len(p), n + w
        if cur_st is not None:
            out.append((cur_st, text[cur_st:cur_end]))
        return [(st, c) for st, c in out if c.strip()]

    def detect(self, text: str) -> list[dict]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        found = []
        low = min(self.th.values())
        for off, chunk in self._chunks(text):
            ents = self.m.predict_entities(chunk, list(self.label2id), threshold=low, flat_ner=False, multi_label=True)
            for e in ents:
                rid = self.label2id[e["label"]]
                typ = FIXED_TYPES.get(rid, "CONFIDENTIAL")
                if e["score"] < self.th[typ]:
                    continue
                r = self.rules[rid]
                found.append({"start": off + e["start"], "end": off + e["end"], "type": typ, "rule_id": rid,
                              "label": r["label"], "grade": r.get("grade", "S"), "action": r.get("action", "대체"),
                              "score": round(float(e["score"]), 3)})
        return sorted(found, key=lambda x: (x["start"], -x["end"]))


def main():
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/v2/final")
    ap.add_argument("--policy", default=str(HERE / "data" / "rules.json"))
    a = ap.parse_args()
    det = SlmDetector(a.model, policy=a.policy)
    for line in sys.stdin:
        line = line.rstrip("\n")
        for d in det.detect(line):
            print(f"{d['type']:12s} {d['rule_id']:7s} {d['grade']} {d['score']:.2f}  {line[d['start']:d['end']]}")
        print("-")


if __name__ == "__main__":
    main()
