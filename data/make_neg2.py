"""v3.1 추가 음성: 공개 회사·제품명 일반 문맥, 정형 번호(규칙 엔진 담당), 가명 토큰 + 동사 → pii_neg2.samples.jsonl"""
import json, random
rng = random.Random(11)
out = []
def neg(t, near): out.append({"text": t, "spans": [], "near": near})
def pos(t, spans): out.append({"text": t, "spans": spans})

# 1. 널리 알려진 공개 회사·서비스·제품명 (기밀 아님)
pubs = ["삼성전자", "LG전자", "SK하이닉스", "현대자동차", "네이버", "카카오", "쿠팡", "배달의민족", "토스", "당근", "AWS", "Azure", "Google Cloud",
        "Microsoft", "Apple", "OpenAI", "Slack", "Notion", "Jira", "Figma", "Salesforce", "Oracle", "SAP", "Zoom", "GitHub", "Docker", "Kubernetes",
        "아이폰 16", "갤럭시 S25", "맥북 에어", "엑셀", "파워포인트", "한글 문서", "구글 시트", "카카오톡", "유튜브", "인스타그램", "넷플릭스", "스타벅스", "이마트"]
ctx = ["{x} 주가가 오늘 왜 떨어졌는지 기사 요약해줘", "{x} 공식 블로그에 나온 신기능 정리해줘", "{x} 요금제 비교표 만들어줘",
       "{x} 사용법 초보자용으로 설명해줘", "{x} 채용 공고 보고 자기소개서 첫 문장 써줘", "작년 {x} 사업보고서에 나온 매출 구조 설명해줘",
       "{x} 같은 회사는 보통 어떤 협업 툴 써?", "{x} 고객센터에 보낼 문의 메일 써줘", "{x} 관련 공개 뉴스 3개만 찾아 요약해줘",
       "우리 팀 {x} 계정 비밀번호 초기화하는 방법 알려줘", "{x}랑 경쟁사 공개 정보 기준으로 비교해줘"]
for x in pubs:
    for t in rng.sample(ctx, 2):
        neg(t.format(x=x), ["R06"])

# 2. 정형 개인정보: 규칙 엔진 담당이라 SLM 은 표시하지 않는다 (이름은 표시)
def rrn(): return f"{rng.randint(70,99)}{rng.randint(1,12):02d}{rng.randint(1,28):02d}-{rng.choice('1234')}{rng.randint(100000,999999)}"
def card(): return " ".join(str(rng.randint(1000, 9999)) for _ in range(4))
def phone(): return f"010-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
def acct(): return f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(100000,999999)}"
def pw(): return rng.choice(["Qwer", "Abcd", "Pass", "Hello", "Zxcv", "Admin"]) + rng.choice("!@#$") + str(rng.randint(1000, 99999))
st = [("주민번호 {r}", "P-ADDR"), ("주민등록번호: {r} 확인 부탁", "P-SENS"), ("카드 {c}", "R10"), ("카드번호 {c} 유효기간 08/27", "R10"),
      ("연락처 {p}로 전화 주세요", "P-ADDR"), ("계좌 {a} (국민)", "P-ADDR"), ("비밀번호는 {w} 야", "R10"), ("초기 비번 {w}로 설정해둠", "R10"),
      ("{r} / {p}", "P-ADDR"), ("결제 카드 {c}, 승인번호 {n}", "R10"), ("사업자등록번호 {b}", "R06"), ("여권번호 M{n2}", "P-ADDR"),
      ("이메일 hello{n}@mail.com, 전화 {p}", "P-ADDR"), ("계정 admin / 비번 {w}", "R10")]
for _ in range(3):
    for t, near in st:
        neg(t.format(r=rrn(), c=card(), p=phone(), a=acct(), w=pw(), n=rng.randint(100000, 999999), n2=rng.randint(10000000, 99999999),
                     b=f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(10000,99999)}"), [near])
names = ["윤서아", "도재민", "Hannah Lee", "방소율", "추민혁", "옥다인"]
for n in names:
    t = f"{n} 고객 주민번호 {rrn()}, 연락처 {phone()} 확인했습니다"
    pos(t, [[n, "P-NAME"]])
    t = f"{n} 님 카드 {card()} 결제 취소 요청 메일 써줘"
    pos(t, [[n, "P-NAME"]])

# 3. 가명 토큰 + 동사·어미 (토큰 옆 말은 이름이 아니다)
verbs = ["내렸어", "받았대", "줬어", "했습니다", "올렸어요", "확정됐어", "통보함", "처리 완료", "결정했대", "전달했어요", "끝났어", "들어갔대"]
titles = ["대리", "과장", "팀장", "님", "씨", "책임", "매니저", ""]
for v in verbs:
    for _ in range(2):
        a = rng.choice(["[이름_1]", "[이름_2]", "[이름_3]"]); b = rng.choice(["[민감정보_1]", "[기밀_1]", "[민감정보_2]", "[사내정보_1]"])
        tt = rng.choice(titles)
        neg(f"{a} {tt} {b} {v}".replace("  ", " "), ["P-NAME"])
    neg(f"{rng.choice(['[이름_1]', '[이름_2]'])}에게 {v}", ["P-NAME"])

# 4. 같은 이름이 조사·단어를 사이에 두고 반복 → 이름마다 따로 (사이 문구까지 한 덩어리로 잡지 않게)
rep = ["{n}의 연락처를 {n}에게 다시 보내줘", "{n} 님 자료는 {n} 님 폴더에 넣어", "{n}가 쓴 보고서를 {n}가 직접 발표해",
       "{n}한테 받은 메일, {n}한테 회신해줘", "{n} 대리 휴가 신청서를 {n} 대리 팀장에게 올려", "{n}의 계정은 {n}만 쓰게 해줘",
       "{n}이 맡던 업무를 {m}에게 넘기고 {n}은 교육팀으로", "{n} 책임 일정이랑 {m} 책임 일정 겹치는지 봐줘"]
pool = ["이도현", "정하린", "최우진", "한소희", "임지호", "배수아", "신동건", "류하은", "권태윤", "Jason Kim", "오세아", "남궁별"]
for i, t in enumerate(rep):
    for k in range(3):
        n, m = pool[(i * 3 + k) % len(pool)], pool[(i * 3 + k + 5) % len(pool)]
        txt = t.format(n=n, m=m)
        pos(txt, [[n, "P-NAME"]] * txt.count(n) + ([[m, "P-NAME"]] if "{m}" in t else []))

# 5. 연락처·번호 뒤에 붙은 건강·민감 정보 → 번호는 표시하지 않고 민감 구절은 P-SENS
# (문장에 넣을 말, 구간) — 구간은 어미 앞까지
health = [("HIV 양성 판정 받은", "HIV 양성 판정"), ("B형 간염 보균자인", "B형 간염 보균자"), ("조현병 치료 중인", "조현병 치료 중"),
          ("항암치료 받는 중인", "항암치료 받는 중"), ("결핵 확진 받은", "결핵 확진"), ("임신 20주차인", "임신 20주차"),
          ("알코올 의존 치료 이력 있는", "알코올 의존 치료 이력"), ("우울증으로 입원했던", "우울증으로 입원"),
          ("뇌전증 약 복용 중인", "뇌전증 약 복용 중"), ("신장 투석 받는", "신장 투석")]
fmt = ["{p} {h} 고객 상담 메모 정리해줘", "고객번호 C-{k} / {p} / {h} 환자", "{p}({h} 분) 예약 변경 안내 문자 써줘",
       "연락처 {p}, 비고: {h} 상태"]
for h, core in health:
    for f in rng.sample(fmt, 2):
        txt = f.format(p=phone(), h=h, k=rng.randint(10000, 99999))
        pos(txt, [[core, "P-SENS"]])

# 6. 은행명·주문/거래 식별자·가명 토큰 주변 (고객사·주소 아님)
banks = ["신한", "국민", "우리", "하나", "농협", "기업", "카카오뱅크", "토스뱅크", "SC제일", "부산은행"]
ords = ["ABC", "ORD-20431", "PO-7781", "INV-0092", "TX93821", "주문 A-1123", "No.4471", "REQ-55", "KR-0091-22", "Z9"]
bfmt = ["주문번호 {o} {b} 계좌 [계좌번호_1]", "주문번호 {o} {b} 계좌 {a}", "{b} 계좌 {a}로 환불 부탁드립니다 (주문 {o})",
        "환불 계좌: {b} [계좌번호_1], 주문 {o}", "{o} 건 입금 확인 — {b}은행 {a}", "거래번호 {o} / {b} / [계좌번호_2]"]
for i, f in enumerate(bfmt):
    for k in range(4):
        neg(f.format(o=ords[(i + k * 3) % len(ords)], b=banks[(i * 2 + k) % len(banks)], a=acct()), ["R06"])

seen, rows = set(), []
for d in out:
    if d["text"] not in seen:
        seen.add(d["text"]); rows.append(d)
with open("data/train_raw/pii_neg2.samples.jsonl", "w") as f:
    for d in rows:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(len(rows), "neg", sum(not d["spans"] for d in rows))
