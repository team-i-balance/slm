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

from splitter import WORD

HERE = Path(__file__).parent
FIXED_TYPES = {"P-NAME": "NAME", "P-ADDR": "ADDRESS", "P-SENS": "SENSITIVE"}
# 기준값: 평가셋 v2 에서 고른 값. 개인정보는 놓치지 않기 우선이라 더 낮다
DEFAULT_THRESHOLDS = {"CONFIDENTIAL": 0.45, "NAME": 0.35, "ADDRESS": 0.35, "SENSITIVE": 0.35}
# 한 번에 넣는 최대 단어 수. 규칙 라벨도 같은 입력에 들어가므로 모델 max_len(384)보다 넉넉히 작게
CHUNK_WORDS = 200
SENT_END = re.compile(r"(?<=[.!?。])\s+|\n+")


# ---- 구간 경계 후처리: 모델은 어절 조각 단위로 구간을 내므로 앞뒤에 조사·직함·역할어가 딸려 온다
#   '다온물산과' → '다온물산', '담당 한지우 매니저' → '한지우', '김민수의' → '김민수', '… 판정 받은 고객' → '… 판정 받은'
ROLE_WORDS = {"담당", "담당자", "수신", "참조", "발신", "작성자", "작성", "고객", "직원", "사원", "신청인", "대표", "보호자",
              "환자", "후임", "전임", "팀원", "동료", "후배", "선배", "To", "CC", "From"}
TITLE_WORDS = {"님", "씨", "대리", "과장", "차장", "부장", "팀장", "책임", "선임", "수석", "매니저", "사원", "주임", "실장", "본부장",
               "이사", "상무", "전무", "대표", "파트장", "PM", "프로", "고객", "환자", "군", "양"}
NAME_JOSA = ("에게서", "께서", "에게", "한테", "이랑", "의", "랑")  # 이름 끝 글자와 헷갈리지 않는 것만
# 띄어 쓰지 않고 붙은 호칭(+조사): '홍길동씨', '박세진님께' — 이름이 '씨·님'으로 끝나는 경우는 없다고 본다
HONORIFIC = tuple(h + j for h in ("씨", "님") for j in ("께서", "에게", "한테", "께", "의", "이", "은", "을", "가", "는", "를", "도", ""))
# 이름 뒤에서 떼어도 되는 꼬리 ('남궁민서' 의 '서' 처럼 mecab 이 이름 끝을 조사로 보는 경우가 있어 목록으로 한정)
NAME_TAIL = set(NAME_JOSA) | {"가", "는", "를", "와", "과", "도", "만", "님", "씨", "이가", "이는", "이를", "이도", "이랑", "이에게"}
PERSON_NOUNS = {"고객", "분", "환자", "직원", "사원", "사람", "씨", "님", "대상", "대상자", "인원"}
EDGE_PUNCT = " ,.;:·/|"
# 떼어도 안전한 격조사 ('만'·'도'·'이' 는 '1만'·숫자 단위·서술어와 헷갈려 뺀다)
CASE_JOSA = {"과", "와", "를", "을", "은", "는", "의", "에", "에서", "로", "으로", "에게", "한테", "와의", "과의", "이랑", "랑", "께", "가"}


def _pieces(splitter, text, start, end):
    """구간 안 어절 조각 [(시작, 끝, 꼬리인지)]. 꼬리 = 분리기가 한글 어절 끝에서 떼어낸 조사·어미 조각
    ('5,400' 의 '400' 처럼 기호 경계로 붙은 조각은 꼬리가 아니다)."""
    out = []
    for m in WORD.finditer(text):
        pos = m.start()
        for k, piece in enumerate(splitter.split_word(m.group())):
            a, b = pos, pos + len(piece)
            if a >= start and b <= end:
                out.append([a, b, k > 0])
            pos = b
    return out


def refine_span(text, start, end, rid, splitter):
    """모델 구간의 앞뒤를 다듬는다. 다 깎이면 원래 구간을 그대로 돌려준다."""
    ps = _pieces(splitter, text, start, end)
    if not ps:
        return start, end
    word = lambda p: text[p[0]:p[1]]
    if rid == "P-NAME":
        while len(ps) > 1 and not ps[0][2] and word(ps[0]) in ROLE_WORDS:
            ps.pop(0)
        while len(ps) > 1 and ((ps[-1][2] and word(ps[-1]) in NAME_TAIL) or (not ps[-1][2] and word(ps[-1]) in TITLE_WORDS)):
            ps.pop()
        st, en = ps[0][0], ps[-1][1]
        w = text[st:en]
        for j in HONORIFIC + NAME_JOSA:  # mecab 이 못 떼는 '김민수의', '홍길동씨'
            if w.endswith(j) and len(w) - len(j) >= 2 and " " not in w:
                en -= len(j)
                break
        return st, en
    # 기밀·주소·민감정보: 평가 기준은 서술어·단위까지 구간에 넣으므로('당뇨 진단을 받으셨어', '1만 대')
    # 확실한 격조사 꼬리와 끝에 붙은 사람 명사만 뗀다
    while len(ps) > 1:
        # '받'+'은' 처럼 한 글자 줄기 뒤의 '은·는·가'는 어미일 수 있어 남긴다
        if ps[-1][2] and word(ps[-1]) in CASE_JOSA and not (word(ps[-1]) in ("은", "는", "가") and len(word(ps[-2])) < 2):
            ps.pop()
        elif rid != "P-ADDR" and not ps[-1][2] and word(ps[-1]) in PERSON_NOUNS:
            ps.pop()
        else:
            break
    st, en = ps[0][0], ps[-1][1]
    while en > st and text[en - 1] in EDGE_PUNCT:
        en -= 1
    return (st, en) if en > st else (start, end)


def load_policy(path):
    """rules.json 형식: {"rules": [{"id", "label", "description", "grade", "action", ...}]}"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = d["rules"] if isinstance(d, dict) else d
    return [r for r in rules if not r["id"].startswith("P-")]


class SlmDetector:
    def __init__(self, model_dir, policy=None, thresholds=None, device="cpu", refine=True):
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
        self.refine = refine
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
                st, en = off + e["start"], off + e["end"]
                if self.refine:
                    st, en = refine_span(text, st, en, rid, self.splitter)
                found.append({"start": st, "end": en, "type": typ, "rule_id": rid,
                              "label": r["label"], "grade": r.get("grade", "S"), "action": r.get("action", "대체"),
                              "score": round(float(e["score"]), 3)})
        # 겹침 허용으로 예측하면 같은 규칙이 겹친 구간을 여러 개 낸다 → 같은 규칙끼리는 가장 긴 것만
        found.sort(key=lambda x: (x["start"], -x["end"]))
        out = []
        for d in found:
            if not any(o["rule_id"] == d["rule_id"] and o["start"] < d["end"] and d["start"] < o["end"] for o in out):
                out.append(d)
        return out

    def judge(self, text: str) -> dict:
        """등급 판정. 모델은 '어느 구간이 어느 규칙에 해당하는지'만 찾고,
        등급은 보안팀이 정한 규칙의 grade 에서 정해진다 (가장 높은 등급, C > S > O).
        confidence 는 백엔드 detection_confidence 와 같은 값: 등급을 정한 구간의 최고 점수 기준."""
        items = self.detect(text)
        grade = max((d["grade"] for d in items), key=GRADE_ORDER.index, default="O")
        top = max((d["score"] for d in items if d["grade"] == grade), default=0.0)
        conf = "high" if top >= HIGH_SCORE else "medium" if items else "low"
        return {"grade": grade, "detection_confidence": conf, "items": items}


GRADE_ORDER = ["O", "S", "C"]
# 이 점수 이상이면 '확실한 탐지'(backend app/rules.md 의 high)
HIGH_SCORE = 0.7


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
