"""G 도메인(게임 운영 AI) 데이터 패키지 생성기.

단일 원본 game_db.json 에서 아래 파일을 모두 결정적으로(시드 고정) 만든다.
  ../corpus/*.md, ../corpus_manifest.json      6회차 코퍼스
  ../faq.json, ../faq_queries.json              1회차 FAQ 의미 검색기
  ../gen_prompts.json                           2회차 생성 설정 비교
  ../schema.json, ../scoring.yaml               5~7회차 출력 형식·채점 규칙
  ../test_seed.json, ../test_dataset.json       5회차 시드 10건(K4·R3·F3), 하네스 입력
  ../my_item_template.json                    5회차 추가 문항 20건 양식(K8·R7·F5)
  ../train_seed.jsonl                           7회차 학습 시드
  ../tools_spec.md                              8~10회차 Tool 후보
  solutions/data/G/test_pool.json               현장 보정·교체 풀(공개 저장소에는 올리지 않음)

실행: python data/G/build/make_game_pkg.py [--train-n 200] [--pool-n 20]

평가용(test)과 학습용(train)은 사실 단위(패치 변경 항목·상품·영웅)와 문장 틀을 서로 다르게 써서 겹치지 않게 한다.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                                   # data/G
REPO = PKG.parents[1]                               # starter_code
DB = json.loads((HERE / "game_db.json").read_text(encoding="utf-8"))
GAME = DB["meta"]["game"]
AS_OF = DB["meta"]["as_of"]

CATEGORIES = ["버그", "밸런스", "결제", "계정", "이벤트", "제재", "기타"]
ACTIONS = ["안내", "버그접수", "결제확인", "보안조치", "제재검토", "이관"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
VERSIONS = [p["version"] for p in DB["patches"]]


# ================================================================ 한국어 조사
def _has_batchim(word: str) -> bool | None:
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 != 0
        if ch == "%":
            return False                     # 퍼센트
        if ch.isdigit():
            return ch in "013678"            # 영·일·삼·육·칠·팔 받침 있음
        if ch.isalpha():
            return None
    return None


def _rieul(word: str) -> bool:
    ch = word.strip()[-1]
    return "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 == 8


PAIRS = {"이": ("이", "가"), "은": ("은", "는"), "을": ("을", "를"), "과": ("과", "와"), "으로": ("으로", "로")}


def josa(word: str, j: str) -> str:
    b = _has_batchim(word)
    if j == "으로":
        return word + ("로" if (b is False or (b and _rieul(word))) else "으로")
    a, c = PAIRS[j]
    return word + (a if b is not False else c)


def fill(tpl: str, **kw) -> str:
    """'{e:이} 약해요' → '서리칼날이 약해요'. 콜론 뒤는 조사."""
    def rep(m):
        key, _, j = m.group(1).partition(":")
        v = str(kw[key])
        return josa(v, j) if j else v
    return re.sub(r"\{(\w+(?::\w+)?)\}", rep, tpl)


# ================================================================ DB 조회·검증
HERO = {h["name"]: h for h in DB["heroes"]}
EQUIP = {e["name"]: e for e in DB["equipment"]}
PRODUCT = {p["name"]: p for p in DB["products"]}
EVENT = {e["name"]: e for e in DB["events"]}


def current_value(target: str, field: str):
    if target in HERO:
        h = HERO[target]
        if field in h["stats"]:
            return h["stats"][field]
        if field in ("계수", "보호막", "회복"):
            return h["skill"]["계수"]
        if field == "쿨타임":
            return h["skill"]["쿨타임"]
    if target in EQUIP:
        return EQUIP[target]["stats"].get(field)
    return None


def validate_db() -> list[dict]:
    """수치 변경 이력의 마지막 값이 현재 DB 값과 같은지 확인하고 수치 변경 목록을 돌려준다."""
    last: dict[tuple, dict] = {}
    numeric = []
    for p in DB["patches"]:
        for c in p["changes"]:
            if c["type"] in ("hero", "equip") and c["before"] is not None:
                c2 = dict(c, version=p["version"], date=p["date"])
                last[(c["target"], c["field"])] = c2
                numeric.append(c2)
    errs = [f"{t}.{f}: 이력 {c['after']} ≠ 현재 {current_value(t, f)}"
            for (t, f), c in last.items() if str(current_value(t, f)) != str(c["after"])]
    if errs:
        raise SystemExit("game_db.json 불일치:\n  " + "\n  ".join(errs))
    return numeric


# ================================================================ 코퍼스
ROLE_NOTE = {
    "탱커": "탱커는 전열에서 적의 공격을 받아내는 역할이며 방어력과 체력 수치 문의가 많다.",
    "딜러": "딜러는 피해를 담당하는 역할이며 공격력·스킬 계수·쿨타임 변경에 민감하다.",
    "서포터": "서포터는 회복·보호막·강화 효과를 담당하며 효과 수치와 쿨타임 문의가 많다.",
}
def doc_id_of(kind: str, key: str) -> str:
    return f"{kind}_{key.lower()}"


def history_of(name: str) -> list[str]:
    rows = []
    for p in DB["patches"]:
        for c in p["changes"]:
            if c["target"] == name:
                if c["before"] is not None:
                    rows.append(f"- {p['version']}({p['date']}): {c['field']} {c['before']} → {c['after']} ({c['kind']}). {c['note']}")
                else:
                    rows.append(f"- {p['version']}({p['date']}): {c['kind']}: {c['note']}")
    for b in DB["known_issues"]:
        if b["target"] == name:
            rows.append(f"- 알려진 문제({b['version']}): {b['symptom']}. 상태: {b['status']}")
    return rows or ["- 변경 이력 없음"]


def make_corpus() -> list[dict]:
    docs = []

    def add(kind, key, title, body):
        docs.append({"doc_id": doc_id_of(kind, key), "title": title, "text": body.strip() + "\n"})

    for h in DB["heroes"]:
        s, k = h["stats"], h["skill"]
        add("hero", h["id"], f"영웅 {h['name']}", f"""
# {h['name']} (영웅)

{GAME}의 {h['element']} 속성 {h['rarity']} {h['role']} 영웅이다. {h['since']}에 출시되었다.

## 기본 능력치 (v1.5 기준)
- 공격력 {s['공격력']}
- 방어력 {s['방어력']}
- 체력 {s['체력']}

## 스킬: {k['name']}
{k['effect']}. 계수 {k['계수']}, 쿨타임 {k['쿨타임']}.

## 운영 참고
{ROLE_NOTE[h['role']]} {h['element']} 속성 적에게는 상성에 따라 피해량이 달라진다. 스킬 수치 문의가 오면 이 문서의 v1.5 기준 값과 변경 이력을 함께 확인한다.

## 변경 이력
""" + "\n".join(history_of(h["name"])))

    for e in DB["equipment"]:
        stats = ", ".join(f"{a} {b}" for a, b in e["stats"].items())
        add("equip", e["id"], f"장비 {e['name']}", f"""
# {e['name']} (장비)

{e['slot']} 장비이다. 현재 능력치는 {stats}이다(v1.5 기준).
{"세트: " + e['set'] + " 세트 효과 대상이다." if e['set'] else "세트 효과가 없는 단일 장비이다."}

## 변경 이력
""" + "\n".join(history_of(e["name"])))

    for p in DB["patches"]:
        bal = [c for c in p["changes"] if c["kind"] in ("너프", "버프")]
        other = [c for c in p["changes"] if c["kind"] not in ("너프", "버프")]
        lines = [f"# {p['version']} 패치노트: {p['title']} ({p['date']})", "",
                 f"{p['date']} 정기 업데이트로 {GAME} {josa(p['version'], '이')} 적용되었다. "
                 "이번 패치노트의 수치 변경은 모두 의도된 조정이며, 변경에 따른 재화 보상은 없다.", ""]
        if bal:
            lines += ["## 밸런스 조정", ""]
            for c in bal:
                arrow = "상향" if c["kind"] == "버프" else "하향"
                kind_word = "영웅" if c["type"] == "hero" else "장비"
                lines += [f"### {c['target']}",
                          f"- {c['field']} {c['before']} → {c['after']} ({arrow})",
                          f"- 조정 이유: {c['note']}.",
                          "- 적용 범위: 이미 보유한 영웅·장비에도 동일하게 적용된다.",
                          "",
                          f"개발자 코멘트: {josa(c['target'], '은')} 지난 버전 동안 사용률과 전투 기록을 함께 살펴본 {kind_word}이다. "
                          f"{c['note']}{'이라는' if _has_batchim(c['note']) else '라는'} 판단에 따라 {josa(c['field'], '을')} {c['before']}에서 {josa(c['after'], '으로')} {arrow} 조정했다. "
                          "이번 변경 후 2주 동안 지표를 다시 확인하고, 필요하면 다음 정기 업데이트에서 추가로 조정한다.", ""]
        if other:
            lines += ["## 신규·시스템·버그 수정", ""]
            for c in other:
                val = f" {c['before']} → {c['after']}" if c["before"] else (f" {c['after']}" if c["after"] else "")
                lines += [f"- [{c['kind']}] {c['target']} {c['field']}{val}: {c['note']}."]
            lines.append("")
        lines += ["## 자주 묻는 질문", "",
                  "- 수치가 바뀐 영웅·장비에 대한 보상이 있나요? 패치노트로 공지한 밸런스 조정은 의도된 변경이므로 재화 보상을 하지 않는다(운영정책 POL-06).",
                  "- 이미 강화한 장비도 수치가 바뀌나요? 강화 단계와 관계없이 기본 수치가 바뀌고, 강화로 더해지는 값은 그대로 유지된다.",
                  "- 패치 후 게임이 실행되지 않으면 어떻게 하나요? 앱을 최신 버전으로 업데이트한 뒤에도 같으면 버그로 신고한다(운영정책 POL-05).", "",
                  "## 점검 안내", "",
                  f"- 업데이트 점검은 {p['date']} 06:00~08:00에 진행했다. 점검 보상으로 루멘석 100개를 우편으로 지급했다.", ""]
        bugs = [b for b in DB["known_issues"] if b["version"] == p["version"]]
        if bugs:
            lines += ["## 이 버전에서 확인된 문제", ""]
            lines += [f"- {b['target']}: {b['symptom']}. ({b['status']})" for b in bugs]
        add("patch", p["version"], f"{p['version']} 패치노트", "\n".join(lines))

    for ev in DB["events"]:
        rw = "\n".join(f"- {a}: {b}" for a, b in ev["rewards"])
        add("event", ev["id"], f"이벤트 {ev['name']}", f"""
# {ev['name']} (이벤트)

- 기간: {ev['period']}
- 비고: {ev['note']}

## 보상
{rw}
""")

    for pr in DB["products"]:
        add("product", pr["id"], f"상품 {pr['name']}", f"""
# {pr['name']} (상점 상품)

- 가격: {pr['price']:,}원
- 구성: {pr['content']}
- 구매 제한: {pr['limit'] or '없음'}

결제 후 상품은 우편함으로 지급된다. 지급되지 않으면 영수증 번호와 함께 고객센터로 문의한다.
""")

    for s in DB["system"]:
        add("system", s["id"], s["title"], f"# {s['title']}\n\n" + "\n".join(f"- {b}" for b in s["body"]))

    for po in DB["policies"]:
        add("policy", po["id"], f"운영정책 {po['id']} {po['title']}", f"""
# 운영정책 {po['id']}: {po['title']}

- 적용 대상: {po['rule']}
- 티켓 분류(category): {po['category']}
- 처리(action): {po['action']}
- 우선순위(priority): {po['priority']}
- 처리 메모: {po['detail']}
""")

    for n in DB["notices"]:
        body = n["body"]
        if body == "__KNOWN_ISSUES__":
            body = [f"[{b['version']}] {b['target']}: {b['symptom']}. 현재 상태: {b['status']}" for b in DB["known_issues"]]
        add("notice", n["id"].replace("notice_", ""), n["title"], f"# {n['title']}\n\n" + "\n".join(f"- {b}" for b in body))
    return docs


# ================================================================ FAQ (1회차)
FAQ = [
    ("성좌 소환 천장은 몇 회인가요?", "80회입니다. 80회 안에 SSR이 나오지 않으면 80회째에 SSR 1개가 확정 지급됩니다(v1.2부터, 이전에는 90회). SSR을 얻으면 횟수는 0으로 초기화됩니다.", "기타"),
    ("소환 1회에 루멘석이 얼마나 드나요?", "1회에 루멘석 150개, 10회 연속 소환에 1500개가 듭니다.", "기타"),
    ("SSR 확률은 얼마인가요?", "SSR 1.5%, SR 12%, R 86.5%입니다. 게임 내 '확률 정보'에서 확인할 수 있습니다.", "기타"),
    ("픽업 소환 천장은 일반 소환과 같이 세나요?", "아닙니다. 픽업 소환의 천장 횟수는 일반 성좌 소환과 따로 계산합니다.", "기타"),
    ("정기 점검은 언제인가요?", "매주 수요일 06:00~08:00에 진행합니다. 점검 중에는 접속할 수 없습니다.", "기타"),
    ("점검 보상은 무엇인가요?", "루멘석 100개를 우편으로 드립니다. 점검이 연장되면 1시간마다 루멘석 50개를 추가로 드립니다.", "기타"),
    ("결제했는데 상품이 안 들어왔어요.", "먼저 우편함을 확인해 주세요. 없으면 영수증 번호와 함께 문의해 주시면 24시간 안에 확인해 드립니다.", "결제"),
    ("환불은 어떻게 하나요?", "구매 후 7일 이내이고 사용하지 않은 상품만 청약철회(환불)할 수 있습니다.", "결제"),
    ("월간 성좌 패스는 무엇을 주나요?", "구매 즉시 루멘석 300개, 이후 30일간 매일 루멘석 100개를 드립니다. 동시에 1개만 보유할 수 있습니다.", "결제"),
    ("계정을 다른 기기로 옮기고 싶어요.", "기존 기기에서 설정 > 계정 연동을 마친 뒤, 새 기기에서 같은 계정으로 로그인하면 됩니다.", "계정"),
    ("계정이 도용된 것 같아요.", "바로 고객센터에 신고해 주세요. 계정을 잠그고 접속 기록을 확인해 드립니다.", "계정"),
    ("닉네임을 바꿀 수 있나요?", "설정 > 프로필에서 바꿀 수 있습니다. 첫 변경은 무료이고 이후에는 닉네임 변경권이 필요합니다.", "계정"),
    ("일일 던전은 하루에 몇 번 들어갈 수 있나요?", "하루 5회 입장할 수 있으며 매일 05:00에 초기화됩니다.", "기타"),
    ("영웅을 분해하면 되돌릴 수 있나요?", "분해는 되돌릴 수 없습니다. 분해하면 등급별 영웅 조각을 돌려받습니다.", "기타"),
    ("우편함 보상은 언제까지 보관되나요?", "우편은 받은 날부터 7일간 보관되며, 기한이 지나면 사라집니다.", "기타"),
    ("매크로를 쓰는 유저를 신고하고 싶어요.", "신고 대상 닉네임과 발견 시각을 함께 보내 주시면 제재 여부를 검토합니다.", "제재"),
    ("욕설하는 유저는 어떻게 신고하나요?", "채팅 메시지를 길게 눌러 신고하거나 닉네임과 시각을 고객센터로 보내 주세요.", "제재"),
    ("호박등 축제는 언제까지인가요?", "2026년 10월 21일부터 11월 10일까지 진행합니다.", "이벤트"),
    ("호박등 축제 7일차 보상을 못 받았어요.", "출석 7일차 보상(SSR 소환권 1장)이 지급되지 않는 문제를 확인했습니다. 운영팀에서 일괄 지급할 예정입니다.", "이벤트"),
    ("길드 레이드는 무엇인가요?", "v1.3에 추가된 주간 길드 레이드 '잿빛 거인'입니다. 길드원이 함께 보스를 공략합니다.", "기타"),
    ("게임이 자꾸 강제 종료돼요.", "앱을 다시 설치하거나 캐시를 지워 보세요. 계속되면 기기 정보와 함께 버그로 신고해 주세요. 잿빛 거인 3페이즈 종료 문제는 확인 중입니다.", "버그"),
    ("무료 루멘석과 유료 루멘석은 어떻게 다른가요?", "이벤트·보상으로 받은 루멘석은 무료 루멘석으로 따로 집계하며 환불 대상이 아닙니다.", "결제"),
    ("패치로 영웅이 약해졌는데 보상받을 수 있나요?", "패치노트로 공지된 밸런스 조정은 의도된 변경이라 보상 대상이 아닙니다.", "밸런스"),
    ("초보 모험가 꾸러미는 몇 번 살 수 있나요?", "계정당 1회만 구매할 수 있습니다.", "결제"),
    ("확률 정보는 어디서 보나요?", "게임 내 '확률 정보' 메뉴에서 모든 소환 상품의 등급별 확률을 볼 수 있습니다.", "기타"),
]

FAQ_QUERIES = [
    ("천장 찍으려면 몇 번 뽑아야 돼요?", 1, "표현 변형·은어"),
    ("수요일 아침에 접속이 안 되는데 왜 그래요?", 5, "표현 변형"),
    ("돈 냈는데 다이아가 안 들어왔어요", 7, "은어"),
    ("구매한 지 사흘 됐는데 돈 돌려받을 수 있나요", 8, "표현 변형"),
    ("폰 바꿨는데 내 캐릭터 그대로 가져가려면?", 10, "표현 변형"),
    ("누가 제 계정에 들어온 거 같아요", 11, "표현 변형"),
    ("일일던전 입장횟수 몇번임", 13, "띄어쓰기·말줄임"),
    ("매크로 유져 신고하고싶어요", 16, "오타"),
    ("너프당한 영웅 보상 없나요?", 23, "은어"),
    ("오늘 저녁 메뉴 추천해 줘", None, "무관 질문"),
]

GEN_PROMPTS = [
    {"id": "G-GP-1", "kind": "사실형", "prompt": "프로젝트 루멘 고객에게 정기 점검을 안내하는 문장을 2문장으로 써 줘. 점검은 매주 수요일 06:00~08:00이고 보상은 루멘석 100개야.",
     "look_for": "주어진 사실(요일·시각·보상)을 바꾸지 않는가"},
    {"id": "G-GP-2", "kind": "사실형", "prompt": "다음 운영정책을 한 문장으로 요약해 줘: 구매 후 7일 이내이고 사용하지 않은 상품의 환불 요청은 결제확인 P2, 7일이 지났거나 일부라도 사용한 상품은 환불 불가를 안내하고 P3.",
     "look_for": "7일·사용 여부 조건을 빠뜨리지 않는가"},
    {"id": "G-GP-3", "kind": "창작형", "prompt": "신규 영웅 '솔라'(불 속성 딜러) 출시를 알리는 공지 제목 3개를 지어 줘.",
     "look_for": "다양성과 어색함의 균형"},
    {"id": "G-GP-4", "kind": "창작형", "prompt": "업데이트 점검이 2시간 연장된 것을 사과하는 공지를 친근한 말투로 3문장으로 써 줘.",
     "look_for": "말투 일관성, 같은 문장 반복 여부"},
    {"id": "G-GP-5", "kind": "형식형", "prompt": "다음 문의를 분류해서 {\"category\": \"...\", \"priority\": \"P1~P4\"} 형식의 JSON 으로만 답해. 문의: 결제했는데 루멘석이 안 들어왔어요",
     "look_for": "JSON 외 문장이 섞이는가"},
    {"id": "G-GP-6", "kind": "형식형", "prompt": "다음 문장에서 게임 고유명사만 골라 JSON 배열로만 답해: 서리칼날 끼고 녹스로 잿빛 거인 잡는데 자꾸 튕겨요",
     "look_for": "배열 형식 준수, 누락·추가"},
]


# ================================================================ 문항 생성 (5·7회차)
FIELD_WORD = {"계수": "스킬 피해량", "쿨타임": "쿨타임", "보호막": "보호막", "공격력": "공격력", "방어력": "방어력",
              "체력": "체력", "치명 확률": "치명타 확률", "속도": "속도"}
NICKS = ["달빛곰", "새벽고양이", "푸른여우", "검은토끼", "별사탕", "모래늑대", "은하수", "구름빵", "초록고래", "번개참새"]

# 사실 단위 분할: 평가(test)와 학습(train)에 서로 다른 항목을 쓴다
BAL_TEST = {("v1.4", "서리칼날"), ("v1.4", "녹스"), ("v1.1", "세라핀"), ("v1.2", "성채 방패"), ("v1.3", "카이로"),
            ("v1.5", "이브렌"), ("v1.5", "용비늘 갑옷"), ("v1.2", "루가"), ("v1.4", "엘로디")}
BUG_TEST = {"BUG-01", "BUG-02"}
PROD_TEST = {"월간 성좌 패스", "루멘석 묶음(대)", "호박등 한정 꾸러미", "SSR 확정 소환권 꾸러미"}
HERO_TEST = {"세라핀", "녹스", "미르", "카이로", "엘로디", "하늘"}
EQUIP_TEST = {"서리칼날", "별빛 반지", "천둥 망치", "성채 방패", "화염의 지팡이"}

T = {  # 평가용 문장 틀
    "nerf": ["{rel} 이후로 {e} {f:이} 줄어든 것 같은데 버그 아닌가요? 보상해 주세요",
             "{e} {f} 왜 낮아졌어요? 공지 본 적 없는데요",
             "{e} 너프 먹은 거 맞아요? {f:이} 확 달라졌어요",
             "{e} 쓰는데 {f:이} 예전보다 확실히 약해요. 오류인지 확인 부탁드립니다"],
    "buff": ["{e} {f} 올라간 거 의도된 건가요? 버그면 나중에 롤백될까 봐요",
             "{rel} 이후 {e:이} 갑자기 세졌는데 {f} 바뀐 거 정상인가요?",
             "{e} {f:이} 좋아졌던데 혹시 오류인가요?"],
    "bug": {"BUG-01": ["녹스 궁 쿨이 40초라고 써 있는데 직접 재 보니 45초 걸려요",
                       "녹스 심연 강림 툴팁이랑 실제 쿨타임이 달라요. 너프된 건가요?"],
            "BUG-02": ["길드 레이드 잿빛 거인 3페이즈만 가면 게임이 꺼져요",
                       "잿빛 거인 잡다가 계속 튕겨서 레이드를 못 끝내요"]},
    "ceiling": ["성좌 소환 {n}번 돌렸는데 SSR이 아직 안 나왔어요. 천장 몇 번이에요?",
                "{n}연차째인데 SSR이 없네요. 몇 번 더 돌려야 확정인가요?"],
    "product_missing": ["{p} 결제했는데 오늘 받아야 할 게 안 들어왔어요",
                        "{p} 샀는데 우편함에 아무것도 없어요"],
    "event_info": ["호박등 축제 5일차 보상이 뭐예요?", "호박등 축제 출석 5일 채우면 뭐 받아요?"],
    "paydouble": ["{p} {n}번 결제됐는데 {m}번만 들어왔어요", "{p:을} 사려다 결제가 {n}번 됐어요. 받은 건 {m}번뿐이에요"],
    "sum": ["보름달 송편 이벤트 매일 루멘석 50개 주는 거, {n}일치가 안 들어왔어요",
            "송편 이벤트 하루 50개씩 주는 루멘석을 {n}일 동안 못 받았어요"],
    "crash": ["{h} 스킬만 쓰면 게임이 꺼져요", "{h:으로} 전투 들어가면 화면 멈추고 튕겨요"],
    "display": ["{q} 아이콘이 다른 장비랑 똑같이 보여요", "{q} 상세 정보 창 글자가 깨져요"],
    "refund": ["{n}일 전에 산 {p:을} 아직 안 썼는데 환불 가능할까요?", "{p} 구매한 지 {n}일 됐는데 안 쓰고 그대로예요. 환불해 주세요"],
    "macro": ["닉네임 {k} 유저가 매크로 돌리는 것 같아요. 하루 종일 같은 던전만 돌아요",
              "{k:이} 불법 프로그램 쓰는 거 같아요. 이동이 비정상적이에요"],
    "abuse": ["채팅에서 계속 욕하는 유저 신고합니다. 닉네임 {k}", "{k} 유저가 길드 채팅에서 막말을 해요"],
    "steal": ["모르는 기기에서 로그인했다는 알림이 왔고 루멘석이 다 없어졌어요",
              "제가 안 했는데 비밀번호 변경 메일이 왔어요. 영웅도 몇 개 사라졌고요"],
    "link": ["폰 바꾸려는데 계정 연동은 어떻게 옮겨요?", "닉네임 바꾸는 방법 알려 주세요"],
    "suggest": ["길드 채팅에 이모티콘 기능 넣어 주세요", "던전 자동 반복 횟수 좀 늘려 주세요"],
}
R = {  # 학습용 문장 틀(평가용과 다르게)
    "nerf": ["{e} {f} 하향됐던데 이거 의도된 건가요?", "업데이트하고 나서 {e} {f:이} 떨어졌어요. 확인해 주세요",
             "{e} 요즘 왜 이렇게 약해졌죠? {f} 바뀐 건가요", "{e} {f} 수치가 달라졌는데 공지된 내용인가요?",
             "패치 후 {e} 체감이 너무 별로예요. {f} 낮춘 거예요?"],
    "buff": ["{e} {f} 상향된 거 맞죠? 확인 부탁드려요", "{e} {f:이} 좋아진 것 같은데 공지 내용 맞나요?",
             "갑자기 {e:이} 좋아졌네요. {f} 올린 거 의도된 건가요?"],
    "bug": {"BUG-03": ["일일 던전 보상 버튼 두 번 눌렀더니 보상이 안 들어왔어요", "일일 던전 끝나고 보상 받기를 연타했는데 아무것도 못 받았어요"],
            "BUG-04": ["호박등 축제 7일 출석했는데 마지막 보상이 안 들어왔어요", "할로윈 출석 7일차 SSR 소환권을 못 받았어요"]},
    "ceiling": ["천장 횟수는 SSR 나오면 처음부터 다시 세나요?", "픽업 소환이랑 일반 소환 천장은 따로 쌓이나요?"],
    "product_missing": ["{p} 결제 완료 문자는 왔는데 상품이 안 보여요", "{p} 구매했는데 지급이 안 됐어요"],
    "event_info": ["출시 기념 출석 3일차 보상이 뭐였죠?", "여름 해변 축제 조개 몇 개 모아야 용비늘 갑옷으로 바꿔요?"],
    "paydouble": ["{p} 결제가 {n}번 빠져나갔는데 지급은 {m}번이에요", "{p} 중복 결제됐어요. {n}번 결제, {m}번 지급"],
    "sum": ["월간 성좌 패스 매일 100개씩 받는 건데 {n}일치가 비어 있어요", "송편 이벤트 루멘석(하루 50개) {n}일 치 못 받았어요 확인해 주세요"],
    "crash": ["{h} 궁 쓰면 앱이 종료돼요", "{h} 데리고 던전 들어가면 로딩에서 멈춰요"],
    "display": ["{q} 툴팁 수치가 겹쳐서 안 보여요", "{q} 이미지가 깨져서 나와요"],
    "refund": ["{p} 산 지 {n}일 됐고 안 썼어요. 청약철회 되나요?", "{n}일 전에 결제한 {p} 환불 신청합니다. 사용 안 했어요"],
    "refund_used": ["{p} 조금 쓰긴 했는데 환불 될까요?", "{p} 받은 거 일부 사용했는데 환불하고 싶어요"],
    "macro": ["{k} 매크로 의심됩니다. 확인 부탁해요", "{k} 핵 쓰는 것 같아요 순간이동해요", "{k:이} 버그 악용해서 보상 무한으로 받아요"],
    "abuse": ["{k:이} 저한테 계속 욕해요", "{k} 비매너 채팅 신고합니다"],
    "steal": ["접속해 보니 제 영웅이 분해돼 있어요. 저는 안 했어요", "해외에서 로그인 시도 알림이 계속 와요"],
    "link": ["게스트 계정을 구글 계정으로 연동하고 싶어요", "기기 변경했는데 로그인을 어떻게 해야 하나요"],
    "suggest": ["우편함 보관 기간 좀 늘려 주세요", "영웅 도감에 보이스 다시 듣기 기능 추가해 주세요"],
}
REL = ["지난 패치", "이번 업데이트", "최근 패치", "패치"]


def gold(category, action, priority, entity=None, patch=None, is_intended=None, fact=None) -> dict:
    return {"category": category, "entity": entity, "patch": patch, "is_intended": is_intended,
            "fact": fact, "action": action, "priority": priority}


def gen_items(split: str, rng: random.Random, numeric: list[dict]) -> dict[str, list[dict]]:
    """split='test' 또는 'train'. 유형별 후보 목록(중복 입력 없음)을 만든다."""
    P = T if split == "test" else R
    out = {"K": [], "R": [], "F": []}

    def push(t, family, text, g, docs):
        out[t].append({"type": t, "family": family, "input": text, "gold": g, "gold_doc_ids": docs})

    # K: 패치 수치 변경
    for c in numeric:
        if ((c["version"], c["target"]) in BAL_TEST) != (split == "test"):
            continue
        kind = "nerf" if c["kind"] == "너프" else "buff"
        for tpl in P[kind]:
            text = fill(tpl, e=c["target"], f=FIELD_WORD.get(c["field"], c["field"]), rel=rng.choice(REL))
            push("K", "balance", text,
                 gold("밸런스", "안내", "P3", c["target"], c["version"], True, f"{c['field']} {c['before']}→{c['after']}"),
                 [doc_id_of("patch", c["version"])])
    # K: 알려진 문제
    for b in DB["known_issues"]:
        if (b["id"] in BUG_TEST) != (split == "test"):
            continue
        for text in P["bug"][b["id"]]:
            if b["target"] == "호박등 축제":
                g = gold("이벤트", "이관", "P2", b["target"], b["version"], False, b["fact"])
                docs = [doc_id_of("notice", "known_issues"), doc_id_of("policy", "POL-07")]
            else:
                g = gold("버그", "버그접수", "P1" if b["blocking"] else "P2", b["target"], b["version"], False, b["fact"])
                docs = [doc_id_of("notice", "known_issues"), doc_id_of("policy", "POL-05")]
            push("K", "known_bug", text, g, docs)
    # K: 소환 천장
    if split == "test":
        for n in (61, 72, 79):
            for tpl in P["ceiling"]:
                push("K", "ceiling", fill(tpl, n=n), gold("기타", "안내", "P4", "성좌 소환", None, None, "천장 80회"),
                     [doc_id_of("system", "gacha"), doc_id_of("policy", "POL-09")])
    else:
        push("K", "ceiling", P["ceiling"][0], gold("기타", "안내", "P4", "성좌 소환", None, None, None),
             [doc_id_of("system", "gacha")])
        push("K", "ceiling", P["ceiling"][1], gold("기타", "안내", "P4", "녹스 픽업 소환", None, None, None),
             [doc_id_of("system", "gacha"), doc_id_of("event", "EV6")])
    # K: 상품 미지급(상품 구성 지식 필요)
    facts = {"월간 성좌 패스": "매일 루멘석 100개", "루멘석 묶음(대)": "루멘석 1200개", "호박등 한정 꾸러미": "루멘석 600개",
             "SSR 확정 소환권 꾸러미": "SSR 확정 소환권 1장, 루멘석 1500개", "루멘석 묶음(중)": "루멘석 330개",
             "루멘석 묶음(소)": "루멘석 60개", "성장 지원 패키지": "레벨 30·50 달성 시 루멘석 각 1500개",
             "초보 모험가 꾸러미": "SR 영웅 선택권 1장, 골드 100000"}
    for p in DB["products"]:
        if (p["name"] in PROD_TEST) != (split == "test"):
            continue
        for tpl in P["product_missing"]:
            push("K", "product_missing", fill(tpl, p=p["name"]),
                 gold("결제", "결제확인", "P1", p["name"], None, None, facts[p["name"]]),
                 [doc_id_of("product", p["id"]), doc_id_of("policy", "POL-01")])
    # K: 이벤트 보상 내용
    if split == "test":
        for text in P["event_info"]:
            push("K", "event_info", text, gold("이벤트", "안내", "P4", "호박등 축제", None, None, "5일차 골드 50000"),
                 [doc_id_of("event", "EV5"), doc_id_of("policy", "POL-07")])
    else:
        push("K", "event_info", P["event_info"][0], gold("이벤트", "안내", "P4", "출시 기념 출석", None, None, "3일차 골드 30000"),
             [doc_id_of("event", "EV1")])
        push("K", "event_info", P["event_info"][1], gold("이벤트", "안내", "P4", "여름 해변 축제", None, None, "조개 300개"),
             [doc_id_of("event", "EV2")])

    # R: 입력만으로 풀 수 있는 문항(추출·계산)
    prods = [p["name"] for p in DB["products"] if (p["name"] in PROD_TEST) == (split == "test")]
    pairs = [(2, 1), (3, 1), (3, 2), (4, 2)] if split == "test" else [(2, 1), (3, 1), (3, 2), (4, 1), (4, 3), (5, 2)]
    for p in prods:
        for i, (n, m) in enumerate(pairs):
            tpl = P["paydouble"][i % len(P["paydouble"])]
            push("R", "paydouble", fill(tpl, p=p, n=n, m=m),
                 gold("결제", "결제확인", "P1", p, None, None, f"결제 {n}회, 지급 {m}회"),
                 [doc_id_of("policy", "POL-01")])
    days = (3, 4, 6, 9) if split == "test" else (2, 5, 7, 8, 10, 11, 12)
    for n in days:
        for i, tpl in enumerate(P["sum"]):
            if split == "train" and i == 0:   # 월간 패스 틀
                push("R", "sum", fill(tpl, n=n), gold("결제", "결제확인", "P1", "월간 성좌 패스", None, None, f"루멘석 {100 * n}개"),
                     [doc_id_of("policy", "POL-01")])
            else:
                push("R", "sum", fill(tpl, n=n), gold("이벤트", "이관", "P2", "보름달 송편 이벤트", None, None, f"루멘석 {50 * n}개"),
                     [doc_id_of("policy", "POL-07")])
    heroes = [h for h in HERO if (h in HERO_TEST) == (split == "test")]
    for h in heroes:
        for tpl in P["crash"]:
            push("R", "crash", fill(tpl, h=h), gold("버그", "버그접수", "P1", h, None, False, None), [doc_id_of("policy", "POL-05")])
    equips = [e for e in EQUIP if (e in EQUIP_TEST) == (split == "test")]
    for q in equips:
        for tpl in P["display"]:
            push("R", "display", fill(tpl, q=q), gold("버그", "버그접수", "P2", q, None, False, None), [doc_id_of("policy", "POL-05")])

    # F: 운영정책 라벨 규약
    ndays = (2, 5, 7, 8, 10, 14) if split == "test" else (1, 3, 4, 6, 9, 12, 15, 20, 30)
    for p in prods:
        for i, n in enumerate(ndays):
            tpl = P["refund"][i % len(P["refund"])]
            ok = n <= 7
            push("F", "refund", fill(tpl, p=p, n=n),
                 gold("결제", "결제확인" if ok else "안내", "P2" if ok else "P3", p, None, None, f"구매 {n}일 경과"),
                 [doc_id_of("policy", "POL-02")])
        if split == "train":
            for tpl in P["refund_used"]:
                push("F", "refund_used", fill(tpl, p=p), gold("결제", "안내", "P3", p, None, None, None), [doc_id_of("policy", "POL-02")])
    nicks = NICKS[:5] if split == "test" else NICKS[5:]
    for i, k in enumerate(nicks):
        push("F", "macro", fill(P["macro"][i % len(P["macro"])], k=k), gold("제재", "제재검토", "P2"), [doc_id_of("policy", "POL-08")])
        push("F", "abuse", fill(P["abuse"][i % len(P["abuse"])], k=k), gold("제재", "제재검토", "P3"), [doc_id_of("policy", "POL-08")])
    for text in P["steal"]:
        push("F", "steal", text, gold("계정", "보안조치", "P1"), [doc_id_of("policy", "POL-03")])
    for text in P["link"]:
        push("F", "link", text, gold("계정", "안내", "P3"), [doc_id_of("policy", "POL-04")])
    for text in P["suggest"]:
        push("F", "suggest", text, gold("기타", "안내", "P4"), [doc_id_of("policy", "POL-09")])

    for t in out:   # 같은 입력 제거
        seen, uniq = set(), []
        for it in out[t]:
            if it["input"] not in seen:
                seen.add(it["input"])
                uniq.append(it)
        out[t] = uniq
    return out


SEED_PICK = {  # 시드 10건(K4·R3·F3): 문항 설계 의도가 드러나는 대표 문항. 키워드가 모두 들어간 첫 후보를 고름
    "K": [("balance", "서리칼날", "보상해"), ("balance", "세라핀", "너프 먹은"), ("known_bug", "녹스", "45초"), ("ceiling", "72", "천장")],
    "R": [("paydouble", "월간 성좌 패스", "결제"), ("sum", "6일", "송편"), ("crash", "미르", "꺼져")],
    "F": [("refund", "10일", "환불"), ("macro", "매크로", ""), ("steal", "모르는 기기", "")],
}


def pick_seed(cands: dict[str, list[dict]]) -> tuple[list[dict], dict[str, list[dict]]]:
    seed, rest = [], {t: list(v) for t, v in cands.items()}
    for t, wants in SEED_PICK.items():
        for fam, *keys in wants:
            it = next(i for i in rest[t] if i["family"] == fam and all(k in i["input"] for k in keys))
            rest[t].remove(it)
            seed.append(it)
    return seed, rest


def schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "G 게임 고객 문의 티켓",
        "description": f"가상 수집형 RPG '{GAME}'의 고객 문의를 읽고 상담 티켓 JSON 하나를 만든다.",
        "type": "object",
        "properties": {
            "category": {"enum": CATEGORIES, "description": "문의 분류"},
            "entity": {"type": ["string", "null"], "description": "문의에 나온 게임 고유명(영웅·장비·이벤트·상품·콘텐츠). 없으면 null. 신고 대상 유저 닉네임은 넣지 않는다"},
            "patch": {"enum": VERSIONS + [None], "description": "관련 패치 버전(변경·버그가 생긴 버전). 없으면 null"},
            "is_intended": {"type": ["boolean", "null"], "description": "공지된 의도된 변경이면 true, 버그면 false, 해당 없으면 null"},
            "fact": {"type": ["string", "null"], "description": "판단 근거가 되는 핵심 사실 한 줄. 수치 변경은 '항목 이전값→이후값' 형식. 없으면 null"},
            "action": {"enum": ACTIONS, "description": "처리 방법"},
            "priority": {"enum": PRIORITIES, "description": "우선순위(P1 가장 급함)"},
        },
        "required": ["category", "entity", "patch", "is_intended", "fact", "action", "priority"],
        "additionalProperties": False,
    }


SCORING_YAML = """# G 도메인 채점 규칙: scorer.py가 읽음
# match: exact(정규화 후 일치) · enum · bool · numeric(허용오차 tol) · numbers(정답 숫자가 출력에 모두 있으면 정답) · set_f1 · contains
# 정답이 null 이면 출력도 null(또는 빈 값)이어야 정답
# core: true 인 필드가 모두 맞으면 Exact Match(보조 지표)
fields:
  category:    {match: enum,   weight: 2, core: true}
  entity:      {match: exact,  weight: 1}
  patch:       {match: exact,  weight: 1}
  is_intended: {match: bool,   weight: 1}
  fact:        {match: numbers, weight: 1}
  action:      {match: enum,   weight: 2, core: true}
  priority:    {match: enum,   weight: 1, core: true}
"""


TOOLS_SPEC = f"""# G 도메인 Tool 후보 (8~10회차 인계용)

> 기준일 {AS_OF}. 시그니처는 제안이며 차승희 박사 회차에서 확정한다. 데이터 원본은 `build/game_db.json`.

| Tool | 입력 | 출력 | 용도 |
|---|---|---|---|
| `item_lookup(name)` | 영웅·장비·상품 이름 | 현재 수치·설명 JSON | 지식형(K) 문의의 사실 확인 |
| `patch_diff(version)` | 패치 버전(v1.0~v1.5) | 변경 항목 목록 | 밸런스 문의에서 의도된 변경 여부 판단 |
| `damage_calc(atk, mult, defense)` | 공격력, 스킬 계수(%), 방어력 | 예상 피해량 | 수치 계산(LLM 산술 오류 보완) |
| `drop_rate_sim(rate, n, trials)` | 확률, 시도 횟수, 시뮬레이션 횟수 | 1회 이상 획득 확률·분포 | 데이터 분석 Tool 예시(천장·확률 문의) |
| `ticket_create(ticket)` | 5~7회차 출력 JSON | 티켓 번호 | FastAPI 서비스 연동(9·10회차) |

`damage_calc` 예시 공식(가상 게임 규칙): 피해량 = 공격력 × 계수 ÷ 100 × (1000 ÷ (1000 + 방어력)), 소수점 버림.
"""


def team_template(seed: list[dict]) -> dict:
    alloc = {"K": 8, "R": 7, "F": 5}
    items, n = [], 1
    for t, k in alloc.items():
        for _ in range(k):
            items.append({"id": f"G-T-{n:02d}", "type": t, "input": "",
                          "gold": {f: None for f in schema()["required"]}, "gold_doc_ids": []})
            n += 1
    return {
        "domain": "G",
        "allocation": alloc,
        "rules": [
            "K(지식형): 코퍼스의 특정 문서를 봐야만 답할 수 있는 문의. gold_doc_ids 에 근거 문서 id 를 반드시 적는다",
            "R(추론형): 입력 문장 안에 판단 근거가 모두 있는 문의(추출·계산)",
            "F(형식·판단형): 운영정책(policy_*) 의 분류 규약을 따라야 맞힐 수 있는 문의",
            "gold 는 schema.json 을 통과해야 한다. 값이 없으면 null",
            "시드 문항과 같은 문장·같은 사실을 다시 쓰지 않는다(표현만 바꾼 문항 금지)",
            "실존 인물·실제 게임 이름·개인정보를 넣지 않는다",
        ],
        "example": seed[0],
        "items": items,
    }


def readme(n_docs: int, n_train: int) -> str:
    return f"""# G 도메인 데이터: 게임 운영 AI({GAME})

| 항목 | 내용 |
|---|---|
| 출처 | 자체 제작 가상 게임 DB(`build/game_db.json`). 실존 게임과 무관함 |
| 라이선스 | CC BY 4.0(자체 저작). 학생 저장소·포트폴리오에 데이터 포함 가능 |
| 기준일 | {AS_OF} |
| 생성 | `python data/G/build/make_game_pkg.py` (시드 고정, 다시 실행해도 같은 결과) |
| 개인정보 | 없음. 닉네임은 가상 |

## 파일

| 파일 | 회차 | 내용 |
|---|---|---|
| `faq.json`, `faq_queries.json` | 1 | FAQ {len(FAQ)}건, 시험 질의 {len(FAQ_QUERIES)}건(표현 변형·오타·은어·무관 질문) |
| `gen_prompts.json` | 2 | 생성 설정 비교용 프롬프트 6건(사실형·창작형·형식형 각 2) |
| `schema.json`, `scoring.yaml` | 5~7 | 출력 JSON 형식, 필드별 채점 규칙 |
| `test_seed.json` | 5 | 시드 10건(K4·R3·F3) |
| `my_item_template.json` | 5 | 직접 추가하는 문항 20건 양식(K8·R7·F5)과 작성 규칙 |
| `test_dataset.json` | 5~7 | `benchmark.py` 입력. 처음에는 시드 10건, 5회차에 팀 20건을 합쳐 30건 |
| `corpus/*.md`, `corpus_manifest.json` | 6 | 위키형 문서 {n_docs}건 |
| `train_seed.jsonl` | 7 | 학습 시드 {n_train}건(`input`·`gold` 쌍. 메시지 형식은 `build_train_data.py`가 평가와 같은 프롬프트로 만듦) |
| `tools_spec.md` | 8~10 | Tool 후보 |

## 평가셋과 학습셋 분리
- 패치 변경 항목·상품·영웅·장비·닉네임을 평가용과 학습용으로 나누고, 문장 틀도 따로 쓴다.
- `build_train_data.py`가 입력 문장 유사도(문자 3-gram 자카드)로 한 번 더 검사한다.

"""


# ================================================================ 실행
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=200)
    ap.add_argument("--pool-n", type=int, default=20, help="교체 풀 유형별 문항 수")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    numeric = validate_db()

    docs = make_corpus()
    cdir = PKG / "corpus"
    cdir.mkdir(exist_ok=True)
    for old in cdir.glob("*.md"):
        old.unlink()
    manifest = []
    for d in docs:
        (cdir / f"{d['doc_id']}.md").write_text(d["text"], encoding="utf-8")
        manifest.append({"doc_id": d["doc_id"], "title": d["title"], "file": f"corpus/{d['doc_id']}.md",
                         "source": "자체 제작 가상 게임 DB", "license": "CC BY 4.0", "as_of": AS_OF,
                         "chars": len(d["text"])})
    w = lambda name, obj: (PKG / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # noqa: E731
    w("corpus_manifest.json", manifest)

    w("faq.json", [{"id": f"F{i:02d}", "question": q, "answer": ans, "category": c} for i, (q, ans, c) in enumerate(FAQ, 1)])
    w("faq_queries.json", [{"qid": f"Q{i:02d}", "query": q, "gold_faq_id": (f"F{g:02d}" if g else None), "variant": v}
                           for i, (q, g, v) in enumerate(FAQ_QUERIES, 1)])
    w("gen_prompts.json", GEN_PROMPTS)
    w("schema.json", schema())
    (PKG / "scoring.yaml").write_text(SCORING_YAML, encoding="utf-8")

    test_c = gen_items("test", rng, numeric)
    seed, rest = pick_seed(test_c)
    for i, it in enumerate(seed, 1):
        it["id"] = f"G-S-{i:02d}"
    seed_out = [{k: it[k] for k in ("id", "type", "input", "gold", "gold_doc_ids")} for it in seed]
    w("test_seed.json", seed_out)
    w("test_dataset.json", seed_out)
    w("my_item_template.json", team_template(seed_out))

    pool = []
    for t in ("K", "R", "F"):
        cand = rest[t]
        rng.shuffle(cand)
        for j, it in enumerate(cand[: a.pool_n], 1):
            pool.append({"id": f"G-P-{t}{j:02d}", "type": t, "input": it["input"], "gold": it["gold"],
                         "gold_doc_ids": it["gold_doc_ids"], "family": it["family"]})
    pool_path = REPO / "solutions" / "data" / "G" / "test_pool.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_c = gen_items("train", rng, numeric)
    flat = [it for t in ("K", "R", "F") for it in train_c[t]]
    rng.shuffle(flat)
    # 유형 비율을 대략 K 30 · R 35 · F 35 로 맞춘다(후보가 모자라면 있는 만큼)
    want = {"K": round(a.train_n * 0.30), "R": round(a.train_n * 0.35)}
    want["F"] = a.train_n - want["K"] - want["R"]
    train, cnt = [], {"K": 0, "R": 0, "F": 0}
    for it in flat:
        if cnt[it["type"]] < want[it["type"]]:
            cnt[it["type"]] += 1
            train.append(it)
    for it in flat:                      # 모자라면 다른 유형으로 채움
        if len(train) >= a.train_n:
            break
        if it not in train:
            train.append(it)
    with open(PKG / "train_seed.jsonl", "w", encoding="utf-8") as f:
        for i, it in enumerate(train, 1):
            f.write(json.dumps({"id": f"G-L-{i:03d}", "type": it["type"], "input": it["input"], "gold": it["gold"]},
                               ensure_ascii=False) + "\n")

    (PKG / "tools_spec.md").write_text(TOOLS_SPEC, encoding="utf-8")
    (PKG / "README.md").write_text(readme(len(docs), len(train)), encoding="utf-8")

    test_inputs = {it["input"] for t in test_c.values() for it in t}
    overlap = [it["input"] for it in train if it["input"] in test_inputs]
    print(f"코퍼스 {len(docs)}건 ({sum(m['chars'] for m in manifest) / 1024:.0f}KB) · FAQ {len(FAQ)} · 질의 {len(FAQ_QUERIES)}")
    print(f"평가 후보 K{len(test_c['K'])} R{len(test_c['R'])} F{len(test_c['F'])} → 시드 {len(seed)} · 교체 풀 {len(pool)}")
    print(f"학습 후보 K{len(train_c['K'])} R{len(train_c['R'])} F{len(train_c['F'])} → 학습 시드 {len(train)} {cnt}")
    print(f"평가·학습 입력 중복 {len(overlap)}건")


if __name__ == "__main__":
    main()
