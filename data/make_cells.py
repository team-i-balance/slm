"""셀 단위 학습 예시 → data/train_raw/pii_cells.samples.jsonl"""
import json, random
rng = random.Random(7)
out = []
def neg(t, near): out.append({"text": t, "spans": [], "near": near})
def pos(t, spans): out.append({"text": t, "spans": spans})

# 1. 표 머리글 단어 (음성)
heads = {
 "P-NAME": ["이름", "성명", "담당자", "담당자명", "고객명", "작성자", "성함", "대표자", "신청인", "수신인", "참석자", "직원명", "사원명", "피보험자", "보호자", "Name", "Full Name", "Owner", "담당 영업", "책임자"],
 "P-ADDR": ["주소", "방문 주소", "배송지", "거주지", "주소지", "사업장 소재지", "근무지", "도로명주소", "우편번호", "Address", "소재지", "수령지"],
 "P-SENS": ["병력", "징계 이력", "건강 상태", "종교", "비고(건강)", "상벌 사항", "휴직 사유", "특이사항", "진단명"],
 "R06": ["거래처", "고객사", "거래처명", "업체명", "회사명", "Vendor", "Customer", "파트너사"],
 "R05": ["계약 조건", "계약 금액", "단가", "결제 조건", "할인율", "계약 기간", "Price", "공급가"],
 "R17": ["협력사", "공급 단가", "월 공급량", "납품처"],
}
for near, ws in heads.items():
    for w in ws:
        neg(w, [near])
# 머리글 행 통째로
for _ in range(25):
    cols = rng.sample([w for ws in heads.values() for w in ws] + ["No.", "번호", "연락처", "이메일", "부서", "직급", "등록일", "상태", "금액", "수량"], rng.randint(3, 7))
    neg(rng.choice([" | ", "\t", ", ", " / "]).join(cols), ["P-NAME"])

# 2. 문서 속성·생성기 문자열 (음성) — 내부 시스템 주소(R10)와 헷갈림
gens = ["Microsoft Excel Compatible / Openpyxl 3.1.5", "Microsoft Excel Compatible / Openpyxl 3.1.2", "openpyxl", "Openpyxl 3.0.10",
        "Microsoft Excel", "Microsoft Office Excel", "Microsoft Office Word", "Microsoft Macintosh Excel", "LibreOffice/7.6.4.1$MacOSX_AARCH64",
        "LibreOffice Calc", "Apache POI", "XlsxWriter 3.2.0", "python-docx", "python-pptx 0.6.23", "Google Sheets", "Numbers 14.1",
        "WPS Spreadsheets", "한컴오피스 한셀 2022", "Hancom Office Hanword", "Microsoft Excel 16.0", "Calc 24.2", "pandas 2.2.3",
        "Sheet1", "Worksheet", "시트1", "Calibri", "맑은 고딕", "Arial", "Office Theme", "Normal.dotm", "Default", "x14ac",
        "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "Application: Microsoft Excel Compatible / Openpyxl 3.1.5",
        "AppVersion 3.1", "dc:creator openpyxl", "Producer: macOS 버전 15.1 Quartz PDFContext", "Created with Aspose.Cells for .NET 24.1",
        "TableStyleMedium9", "General", "yyyy-mm-dd", "#,##0", "0.00%"]
for g in gens:
    neg(g, ["R10"])

# 3. 가명 토큰이 섞인 문장 (토큰은 음성, 남은 진짜 값은 양성)
toks = ["[이름_1]", "[이름_2]", "[이름_3]", "[주소_1]", "[주소_2]", "[민감정보_1]", "[민감정보_2]", "[기밀_1]", "[기밀_2]", "[사내정보_1]",
        "[전화번호_1]", "[이메일_1]", "[계좌번호_1]", "[식별번호_1]", "[사업자번호_1]", "[생년월일_1]"]
tmpl = [
 ("{이름} 님께 회신 메일 정중하게 써줘", "P-NAME"), ("{이름} 팀장이 요청한 보고서 목차 잡아줘", "P-NAME"),
 ("{이름}, {이름2} 두 분 미팅 잡아주세요", "P-NAME"), ("담당자 {이름} 매니저에게 일정 공유 부탁", "P-NAME"),
 ("{주소}로 샘플 보내드릴게요. 안내 문자 문구 만들어줘", "P-ADDR"), ("배송지: {주소} / 수령인: {이름}", "P-ADDR"),
 ("{주소} 근처 회의실 예약 메일 써줘", "P-ADDR"), ("고객 주소 {주소}, 연락처 {전화} 확인했습니다", "P-ADDR"),
 ("{민감} 내렸어. 공지문 톤 다듬어줘", "P-SENS"), ("{이름} 씨 {민감} 상태라 업무 조정 필요", "P-SENS"),
 ("{민감} 관련 면담 기록 요약해줘", "P-SENS"), ("{기밀} 건 진행 상황 정리해서 표로 만들어줘", "R18"),
 ("{기밀} 수치는 빼고 보고서 문장만 다듬어줘", "R03"), ("{사내} 접속이 안 된다는 문의 답변 써줘", "R10"),
 ("{이름}({생일}) 고객 문의: 환불 절차 안내 부탁", "P-NAME"), ("입금 계좌 {계좌}로 변경 요청 메일 회신 써줘", "P-NAME"),
 ("[이름_1]: 내일 회의 몇 시죠?\n[이름_2]: 10시요", "P-NAME"), ("{이메일}로 견적서 보내달라고 하네요", "P-NAME"),
 ("사업자번호 {사업자} 거래처 등록 서류 체크리스트 줘", "R06"), ("{기밀}, {기밀2} 두 항목은 가려진 상태로 공유됨", "R21"),
]
fill = lambda: {"이름": rng.choice(toks[:3]), "이름2": "[이름_4]", "주소": rng.choice(toks[3:5]), "민감": rng.choice(toks[5:7]),
                "기밀": rng.choice(toks[7:9]), "기밀2": "[기밀_3]", "사내": "[사내정보_1]", "전화": "[전화번호_1]", "이메일": "[이메일_1]",
                "계좌": "[계좌번호_1]", "사업자": "[사업자번호_1]", "생일": "[생년월일_1]"}
for t, near in tmpl:
    for _ in range(3):
        neg(t.format(**fill()), [near])
for t in toks:
    neg(t, ["P-NAME" if "이름" in t else "P-ADDR" if "주소" in t else "P-SENS" if "민감" in t else "R10"])
# 토큰 + 진짜 값이 같이 남은 경우
mixed = [("[이름_1] 팀장이 서하람 대리한테 넘긴 건이야", [["서하람", "P-NAME"]]),
 ("[주소_1] 말고 대전시 유성구 대학로 291 연구동 4층으로 보내줘", [["대전시 유성구 대학로 291 연구동 4층", "P-ADDR"]]),
 ("[이름_2] 씨와 구태오 씨 둘 다 다음 주 휴가", [["구태오", "P-NAME"]]),
 ("[민감정보_1] 이후 류은채 사원이 우울증 상담 받는 중", [["류은채", "P-NAME"], ["우울증 상담 받는 중", "P-SENS"]]),
 ("[기밀_1] 다음 단계로 노바틱스 지분 40% 인수 협상 진행", [["노바틱스 지분 40% 인수 협상", "R04"]]),
 ("수신 [이메일_1], 참조 맹지오 책임", [["맹지오", "P-NAME"]]),
 ("[이름_3] 부장 연봉 8,200만원으로 조정, 공지 문구 써줘", [["연봉 8,200만원", "R08"]]),
 ("[주소_2]에서 경기도 김포시 고촌읍 아라육로 152번길 8로 이사", [["경기도 김포시 고촌읍 아라육로 152번길 8", "P-ADDR"]]),
 ("[이름_1] 님 후임은 Olivia Grant 매니저입니다", [["Olivia Grant", "P-NAME"]]),
 ("[사내정보_1] 대신 git.corp-hanul.local:8443 으로 접속", [["git.corp-hanul.local:8443", "R10"]]),
 ("[이름_2]가 금요일에 맹장 수술 받는대", [["맹장 수술 받는대", "P-SENS"]]),
 ("[기밀_2] 건, 다음 달 3일 리브라 앱 출시로 확정", [["다음 달 3일 리브라 앱 출시", "R02"]])]
for t, s in mixed:
    pos(t, s)

# 4. 문맥 없는 회사명 셀 (거래처·협력사 열에 들어가는 값) — 양성
cos = ["한별정공", "누리온소재", "새벽빛로지스", "마루결전자", "도담하이텍", "솔빛케미칼", "아라테크놀로지", "청람기연", "모아솔루션즈", "하랑식품",
       "대은메디칼", "윤슬바이오", "가람정밀", "늘봄유통", "라온시스템", "이음엔지니어링", "다솜화학", "빛고을정밀", "푸른숲인터내셔널", "한빛모터스",
       "Celvara Inc.", "Northmere Ltd", "Orbis Tech", "(주)태성하이텍", "주식회사 은하소재", "미르테크(주)", "제이엠피앤씨", "에스와이산업", "코아넷", "보람기공"]
for c in cos[:12] + cos[20:23]:  # v3.1: 30→15 (문맥 없는 회사명 양성이 공개 회사명 오탐을 늘림)
    pos(c, [[c, rng.choice(["R06", "R17"])]])
# 회사명 셀의 함정: 공개된 대기업·일반 표현은 셀이어도 기밀 아님
for t in ["(주)", "주식회사", "거래처 없음", "미정", "-", "N/A", "해당 없음", "합계", "소계", "TOTAL", "신규 등록 예정", "기타", "개인", "법인", "본사"]:
    neg(t, ["R06"])

# 5. 계약 조건 셀 — 양성 (단가·물량·금액·결제 조건)
terms = ["개당 2,140원", "단가 kg당 4,120원", "장당 870원, 월 5만 장", "금형 2벌 9,600만원", "연간 공급 계약 18억", "월 정액 1,350만원",
         "연 유지보수 2억 7천", "선급금 30%, 잔금 납품 후 60일", "할인율 12%", "기본 단가 대비 8% 인하", "최소 발주 1만 2천 개", "건당 수수료 3.2%",
         "시간당 9만 5천원", "라이선스 1식 6,400만원", "운임 톤당 38,000원", "개발비 2억 1천만원 일시 지급", "MOQ 3,000 EA, USD 4.85/EA",
         "단가 1,920원 → 1,780원", "3년 약정 총 54억", "분기 리베이트 2.5%", "월 3만 개, 개당 640원", "회당 납품 1,200세트",
         "장비 임대 월 870만원", "계약금 1억, 중도금 2억", "결제 조건 월말 마감 90일 어음", "카톤당 12,300원", "연 단가 동결, 물량 20% 증량"]
for t in terms:
    rid = rng.choice(["R05", "R05", "R13", "R17"])
    pos(t, [[t, rid]])
# 계약 조건 열의 함정: 숫자 없는 상태값·일반 문구
for t in ["협의 중", "계약 완료", "갱신 예정", "표준 계약서 적용", "조건 별도 첨부", "견적 요청", "단가표 참조", "부가세 별도", "VAT 포함", "검토 필요"]:
    neg(t, ["R05"])

# 6. 이름·주소·민감정보 셀
for n in ["윤다결", "제갈하온", "Maya Lindqvist", "피수아 과장", "탁로운", "Ethan Cho 매니저", "어서진", "봉새별 대리", "남궁린", "채온유"]:
    nm = n.rsplit(" ", 1)[0] if n.endswith(("과장", "매니저", "대리")) else n
    pos(n, [[nm, "P-NAME"]])
for a in ["광주시 북구 첨단과기로 123, 2층", "충북 청주시 흥덕구 오송생명2로 187", "서울 성동구 아차산로 17길 48, B동 1102호",
          "울산광역시 남구 삼산로 274 5층", "제주시 첨단로 213-4", "경남 창원시 성산구 공단로 471번길 12"]:
    pos(a, [[a, "P-ADDR"]])
for s in ["정직 2개월", "견책", "감봉 3개월 징계", "당뇨 치료 중", "허리 수술 후 재활", "공황장애로 병가", "출산휴가 중", "알코올 의존 치료"]:
    sp = [[s, "P-SENS"]] + ([[s, "R07"]] if any(k in s for k in ("정직", "견책", "감봉")) else [])
    pos(s, sp)
for t in ["양호", "이상 없음", "정상 근무", "해당 사항 없음", "재직", "휴가", "출근", "건강검진 대상"]:
    neg(t, ["P-SENS"])

seen, rows = set(), []
for d in out:
    if d["text"] not in seen:
        seen.add(d["text"]); rows.append(d)
with open("data/train_raw/pii_cells.samples.jsonl", "w") as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(len(rows), "neg", sum(not d["spans"] for d in rows))
