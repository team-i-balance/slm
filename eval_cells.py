"""엑셀 셀 단위 탐지율 (문맥 없는 짧은 조각에서 모델이 얼마나 잡는지).

  python eval_cells.py --model models/v3/trim [--xlsx demo/samples/slm_partners.xlsx]

셀 하나씩 따로 모델에 넣는다 (백엔드 XLSX 검사와 같은 방식). 정책은 모아랩스 규칙(data/rules.json).
- 열별 탐지율: 머리글 행을 뺀 셀 중 무엇이든 하나 잡힌 비율 (머리글 열 이름으로 묶는다)
- 머리글 오탐: 첫 행 셀에서 잡힌 것
- 점검 문자열: 백엔드 연동에서 오탐이 났던 유형(생성기 문자열·머리글 단어·가명 토큰). 정답은 '아무것도 안 잡힘'
"""
import argparse
import re
import zipfile
from collections import defaultdict

from detector import SlmDetector

PROBES = ["Microsoft Excel Compatible / Openpyxl 3.1.5", "openpyxl", "Microsoft Office Word", "LibreOffice Calc 24.2",
          "이름", "성명", "담당자", "주소", "고객명", "비고", "계약 조건",
          "[주소_1]로 보내드릴게요", "[이름_2] 님께 회신 부탁", "[민감정보_1] 내렸어", "[기밀_3] 관련 자료 정리해줘",
          "[이름_1] 대리 [민감정보_1] 내렸어",
          # 정형 개인정보는 규칙 엔진 담당이라 SLM 은 잡지 않아야 한다 (배포 서버에서 본 오탐)
          "주민번호 900101-1234568", "카드 4111 1111 1111 1111", "10.20.30.40, 비밀번호는 Qwer!2345 야",
          "주문번호 ABC 신한 계좌 [계좌번호_1]", "주문번호 ABC 신한 계좌 110-123-456789",  # v3.1 학습 문장과 같음
          "결제건 QX-3301 대구은행 [계좌번호_1]로 환불", "주문 7731-B 수협 계좌 233-01-778812"]  # 학습에 없는 변형

# 정확히 이 구간들을 따로 잡아야 하는 문장 (배포 서버에서 본 사례와 같은 유형, 학습 문장과는 다르게)
#   (문장, [(구간, 고정 라벨)])  — 맞음 = 정답 구간마다 같은 위치·같은 라벨 예측이 있고, 정답 밖으로 넘치는 이름 구간이 없음
POS_PROBES = [("김민수의 번호를 김민수에게 전달해줘", [("김민수", "P-NAME"), ("김민수", "P-NAME")]),
              ("박지은 씨 메일은 박지은 씨 개인 주소로 보내", [("박지은", "P-NAME"), ("박지은", "P-NAME")]),
              ("정우성한테 온 자료를 정우성 팀장이 검토", [("정우성", "P-NAME"), ("정우성", "P-NAME")]),
              ("010-1234-5678 HIV 양성 판정 받은 고객", [("HIV 양성 판정", "P-SENS")]),
              ("고객 010-2222-3333 / C형 간염 확진 / 상담 요청", [("C형 간염 확진", "P-SENS")]),
              ("연락처 010-5555-0101, 공황장애로 약 복용 중인 분", [("공황장애로 약 복용 중", "P-SENS")])]


def pos_probe(det, text, gold):
    items = det.detect(text)
    spans, st = [], -1
    for sub, rid in gold:  # 같은 문자열이 여러 번이면 앞에서부터
        st = text.find(sub, st + 1)
        spans.append((st, st + len(sub), rid))
    found = all(any(d["rule_id"] == rid and d["start"] <= s and d["end"] >= e and (d["end"] - d["start"]) <= (e - s) + 3
                    for d in items) for s, e, rid in spans)
    return found, items


def read_xlsx(path):
    """openpyxl 없이 첫 시트의 셀 문자열을 행 목록으로 읽는다 (inlineStr·sharedStrings 둘 다)."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in re.findall(r"<si>(.*?)</si>", z.read("xl/sharedStrings.xml").decode(), re.S):
            shared.append("".join(re.findall(r"<t[^>]*>([^<]*)</t>", si)))
    rows = []
    for r in re.findall(r"<row[^>]*>(.*?)</row>", z.read("xl/worksheets/sheet1.xml").decode(), re.S):
        cells = []
        for attrs, body in re.findall(r"<c([^>]*)>(.*?)</c>", r, re.S):
            t = "".join(re.findall(r"<t[^>]*>([^<]*)</t>", body))
            if 't="s"' in attrs:
                t = shared[int(re.search(r"<v>(\d+)</v>", body).group(1))]
            elif not t:
                t = "".join(re.findall(r"<v>([^<]*)</v>", body))
            cells.append(unescape(t))
        rows.append(cells)
    return rows


def unescape(s):
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")


def show(items, text):
    return ", ".join(f"{text[d['start']:d['end']]}={d['rule_id']}({d['score']:.2f})" for d in items) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--xlsx", default="demo/samples/slm_partners.xlsx")
    a = ap.parse_args()
    det = SlmDetector(a.model, policy="data/rules.json")
    rows = read_xlsx(a.xlsx)
    head, body = rows[0], rows[1:]
    hit = defaultdict(int)
    for row in body:
        for h, cell in zip(head, row):
            items = det.detect(cell)
            hit[h] += bool(items)
            print(f"  [{h}] {cell} → {show(items, cell)}")
    print("열별 탐지율:", ", ".join(f"{h} {hit[h]}/{len(body)}" for h in head))
    fp = [(h, show(det.detect(h), h)) for h in head if det.detect(h)]
    print(f"머리글 오탐: {len(fp)}/{len(head)}", fp)
    bad = [(p, show(det.detect(p), p)) for p in PROBES if det.detect(p)]
    print(f"점검 문자열 오탐: {len(bad)}/{len(PROBES)}")
    for p, s in bad:
        print(f"  {p} → {s}")
    res = [(t, *pos_probe(det, t, g)) for t, g in POS_PROBES]
    print(f"양성 점검 문장(정확한 구간): {sum(ok for _, ok, _ in res)}/{len(res)}")
    for t, ok, items in res:
        print(f"  {'O' if ok else 'X'} {t} → {show(items, t)}")


if __name__ == "__main__":
    main()
