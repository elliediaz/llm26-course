"""L 도메인(알바 급여 정산) 데이터 패키지 생성기. 시드 고정이라 다시 실행해도 같은 결과가 나온다.

  python data/L/build/make_wage_pkg.py

만드는 것(data/L/): schema.json, scoring.yaml, test_seed.json, test_dataset.json(처음에는 시드와 같음),
my_item_template.json, corpus/*.md, corpus_manifest.json, train_seed.jsonl, tools_spec.md, build/wage_db.json
교체 풀(solutions/data/L/, 공개 저장소에는 올리지 않음): test_pool.json(현장 보정용)
모든 사업장·사례는 가상이며, 정답은 wage_rules.compute 로 계산한다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import wage_rules as W

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
SOL = HERE.parents[2] / "solutions" / "data" / "L"
AS_OF = W.AS_OF
LICENSE = "CC BY 4.0"
SOURCE = "자체 제작 가상 상권 DB"

# ------------------------------------------------------------------ 가상 사업장(한빛로)
# use: eval(평가 문항용) / train(학습 문항용). 상시근로자 수는 사업주 제외
WORKPLACES = [
    ("ws01", "모퉁이분식", "분식점", 3, 10320, 10500, "eval", "점심·저녁 홀 서빙과 포장"),
    ("ws02", "한빛베이커리", "제과점", 6, 10400, 10700, "eval", "아침 진열과 계산"),
    ("ws03", "달빛편의점", "편의점", 4, 10320, 10700, "eval", "심야 근무 있음(22시~다음 날 7시)"),
    ("ws04", "새벽PC방", "PC방", 5, 10000, 10600, "eval", "주말 야간 근무 있음"),
    ("ws05", "오름카페", "카페", 8, 10500, 10900, "eval", "평일 오전·오후 교대"),
    ("ws06", "바람세탁소", "세탁소", 2, 10320, 10700, "eval", "접수·수선 보조"),
    ("ws07", "소나무치킨", "치킨집", 7, 10320, 10800, "eval", "저녁·심야 배달 포장"),
    ("ws08", "별마루스터디카페", "스터디카페", 4, 10320, 10700, "eval", "야간 관리 근무 있음"),
    ("ws09", "하늘서점", "서점", 3, 10350, 10700, "eval", "주말 계산·진열"),
    ("ws10", "다온김밥", "김밥집", 5, 10200, 10700, "eval", "아침 조리 보조"),
    ("ws11", "나래꽃집", "꽃집", 2, 10500, 10800, "eval", "포장·배송"),
    ("ws12", "햇살코인노래방", "코인노래방", 4, 10320, 10650, "eval", "주말 야간 근무 있음"),
    ("ws13", "푸른정육점", "정육점", 6, 10600, 11000, "eval", "저녁 손질·포장"),
    ("ws14", "미소도시락", "도시락 전문점", 9, 10320, 10700, "eval", "평일 낮 조리·포장"),
    ("ws15", "봄날떡집", "떡집", 3, 10320, 10700, "train", "새벽 조리 보조"),
    ("ws16", "한결헬스장", "헬스장", 5, 10400, 10800, "train", "새벽·야간 안내 근무 있음"),
    ("ws17", "유리안경점", "안경점", 4, 10500, 10900, "train", "주말 접객"),
    ("ws18", "동그라미문구", "문구점", 2, 10100, 10700, "train", "평일 오후 계산"),
    ("ws19", "단비빙수", "빙수 가게", 6, 10320, 10600, "train", "주말 홀 서빙"),
    ("ws20", "가람순대국", "식당", 7, 10320, 10700, "train", "심야 영업 있음"),
    ("ws21", "노을횟집", "횟집", 4, 10300, 10700, "train", "저녁·심야 홀 서빙"),
    ("ws22", "솔방울베이글", "베이커리 카페", 5, 10320, 10750, "train", "오전 굽기·진열"),
    ("ws23", "은하수볼링장", "볼링장", 10, 10500, 10900, "train", "야간 카운터 근무 있음"),
    ("ws24", "초록채소가게", "채소 가게", 3, 10320, 10700, "train", "오전 입고·진열"),
]
WS = {w[0]: dict(zip(("id", "name", "industry", "size", "w2026", "w2027", "use", "work"), w)) for w in WORKPLACES}


def ws_doc(w: dict) -> str:
    small = "5인 미만" if w["size"] < 5 else "5인 이상"
    return (f"# {w['name']} (한빛로 가상 사업장)\n\n"
            f"한빛로에 있는 {w['industry']}이다. 사업주를 뺀 상시근로자는 {w['size']}명({small})이다.\n\n"
            f"| 항목 | 내용 |\n|---|---|\n| 업종 | {w['industry']} |\n"
            f"| 상시근로자 수(사업주 제외) | {w['size']}명({small}) |\n"
            f"| 약정 시급(2026년) | {w['w2026']:,}원 |\n| 약정 시급(2027년) | {w['w2027']:,}원(2027-01-01부터) |\n"
            f"| 주요 업무 | {w['work']} |\n\n"
            f"- 기준일 {AS_OF}. 실존 상호와 관계없는 가상 사업장임\n")


MW_DOCS = {
    "mw_2026": ("2026년 적용 최저임금", "# 2026년 적용 최저임금\n\n"
                "- 시간급 10,320원(2026-01-01~2026-12-31 적용)\n"
                "- 월 환산액 2,156,880원(주 소정근로 40시간, 유급 주휴 포함 월 209시간 기준)\n"
                "- 근거: 최저임금법 제8조·제10조, 고용노동부 고시\n"
                "- 업종 구분 없이 모든 사업장에 같은 금액을 적용함\n"),
    "mw_2027": ("2027년 적용 최저임금", "# 2027년 적용 최저임금\n\n"
                "- 시간급 10,700원(2027-01-01~2027-12-31 적용)\n"
                "- 월 환산액 2,236,300원(주 소정근로 40시간, 유급 주휴 포함 월 209시간 기준)\n"
                "- 근거: 최저임금법 제8조·제10조, 2026년 8월 고용노동부 고시\n"
                "- 업종 구분 없이 모든 사업장에 같은 금액을 적용함\n"),
}
RULE_DOCS = {
    "rule_minwage": ("최저임금 미달 판단", "# 최저임금 미달 판단(자체 작성 해설)\n\n"
                     "- 약정 시급이 그해 최저시급보다 낮으면 최저임금법 제6조 위반임. 근로자는 최저시급 기준으로 계산한 금액을 받아야 함\n"
                     "- 차액 = (최저시급 - 약정 시급) × 근로시간. 주휴수당도 최저시급 기준으로 다시 계산함\n"
                     "- 적용 연도는 근무한 날짜로 정함. 이 과정의 문항은 한 해 안의 근무만 다룸\n"),
    "rule_holiday": ("주휴수당 계산", "# 주휴수당 계산(자체 작성 해설)\n\n"
                     "- 근거: 근로기준법 제55조, 같은 법 시행령 제30조\n"
                     "- 1주 소정근로시간이 15시간 이상이고 그 주를 개근하면 유급 주휴일 하루분을 받음\n"
                     "- 1주 주휴수당 = (1주 소정근로시간 ÷ 40) × 8 × 시급. 소정근로시간이 40시간을 넘으면 40으로 계산(최대 8시간분)\n"
                     "- 1주 소정근로시간이 15시간 미만이면 주휴수당이 없음(근로기준법 제18조 제3항)\n"
                     "- 예: 주 18시간, 시급 10,700원 → 18 ÷ 40 × 8 × 10,700 = 38,520원\n"),
    "rule_premium": ("연장·야간 가산수당", "# 연장·야간 가산수당(자체 작성 해설)\n\n"
                     "- 근거: 근로기준법 제56조\n"
                     "- 1주 40시간을 넘는 연장근로, 22시~다음 날 6시의 야간근로에는 통상임금의 50%를 더 줌\n"
                     "- 이 과정의 문항은 하루 8시간 이하 근무만 다루고, 연장근로는 1주 40시간 초과분으로만 셈\n"
                     "- 상시근로자 5인 미만 사업장에는 가산수당 규정이 적용되지 않음(규칙 문서 rule_small 참고)\n"),
    "rule_small": ("5인 미만 사업장", "# 5인 미만 사업장(자체 작성 해설)\n\n"
                   "- 근거: 근로기준법 제11조 제2항, 같은 법 시행령 제7조 [별표 1]\n"
                   "- 상시근로자 4명 이하인 사업장은 연장·야간·휴일 가산수당(제56조)을 적용하지 않음\n"
                   "- 최저임금과 주휴수당은 5인 미만 사업장에도 적용됨\n"
                   "- 상시근로자 수에는 사업주를 넣지 않음\n"),
    "rule_hours": ("1주 소정근로시간", "# 1주 소정근로시간(자체 작성 해설)\n\n"
                   "- 소정근로시간은 근로자와 사용자가 정한 근로시간이며 1주 40시간을 넘을 수 없음(근로기준법 제50조)\n"
                   "- 이 과정의 정산 JSON에서 weekly_hours는 1주 실제 근로시간과 40 중 작은 값으로 적음\n"
                   "- 40시간을 넘는 시간은 연장근로로 보고 5인 이상 사업장이면 가산수당 대상임\n"),
    "law_summary": ("근거 조문 요지", "# 근거 조문 요지(자체 정리)\n\n"
                    "조문 원문은 국가법령정보센터(law.go.kr)에서 확인함. 아래는 이 과정 문항에 필요한 부분만 줄여 적은 것임\n\n"
                    "| 조문 | 요지 | 쓰이는 쟁점 |\n|---|---|---|\n"
                    "| 최저임금법 제6조 | 사용자는 최저임금 이상의 임금을 지급해야 함. 최저임금보다 적게 정한 부분은 무효이고 최저임금으로 정한 것으로 봄 | 최저임금미달 |\n"
                    "| 최저임금법 제8조·제10조 | 최저임금은 매년 정하고, 고용노동부 장관이 고시하며 다음 해 1월 1일부터 효력이 생김 | 적용 연도 판단 |\n"
                    "| 근로기준법 제55조 | 사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장해야 함 | 주휴미지급 |\n"
                    "| 근로기준법 시행령 제30조 | 주휴일은 1주 동안 소정근로일을 개근한 사람에게 줌 | 주휴미지급 |\n"
                    "| 근로기준법 제18조 제3항 | 4주 평균 1주 소정근로시간이 15시간 미만이면 주휴일 규정을 적용하지 않음 | 주휴 대상 여부 |\n"
                    "| 근로기준법 제50조 | 1주 근로시간은 휴게시간을 빼고 40시간을 넘을 수 없음 | 소정근로시간 |\n"
                    "| 근로기준법 제56조 | 연장근로와 야간근로(22시~다음 날 6시)에는 통상임금의 50% 이상을 더 줘야 함 | 가산미지급 |\n"
                    "| 근로기준법 제11조, 시행령 제7조 [별표 1] | 상시근로자 4명 이하 사업장에는 일부 규정만 적용됨. 제56조 가산수당은 적용 대상이 아님 | 5인 미만 판단 |\n"),
    "rule_format": ("정산 JSON 작성 규약", "# 정산 JSON 작성 규약(이 과정의 채점 기준)\n\n"
                    "- 금액은 원 단위 정수로 쓰고, 계산 중 생긴 원 미만은 버림\n"
                    "- 근무 시간·시급·사업장 규모(가산수당이 걸리는 경우) 중 하나라도 알 수 없으면 issue는 판단불가, 수치 필드는 모두 null\n"
                    "- 쟁점이 여럿이면 최저임금미달 > 주휴미지급 > 가산미지급 순서로 하나만 고름. unpaid_krw에는 모든 미지급액을 합침\n"
                    "- law_article은 대표 쟁점의 조문 하나: 최저임금미달 = 최저임금법 제6조, 주휴미지급 = 근로기준법 제55조, 가산미지급 = 근로기준법 제56조. 정상지급·판단불가는 null\n"
                    "- 법률 자문이 아니며, 실제 사안은 고용노동부 상담(국번 없이 1350)으로 확인함\n"),
}

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "L 알바 급여 정산",
    "description": ("가상 상권 '한빛로'의 아르바이트 근무 상황을 읽고 급여 정산 결과를 JSON 하나로 만든다. "
                    "금액은 원 단위 정수(원 미만 버림). 판단에 필요한 정보가 없으면 issue를 판단불가로, 수치 필드를 null로 둔다. 법률 자문이 아님."),
    "type": "object",
    "properties": {
        "issue": {"enum": ["최저임금미달", "주휴미지급", "가산미지급", "정상지급", "판단불가"],
                  "description": "대표 쟁점. 여럿이면 최저임금미달 > 주휴미지급 > 가산미지급 순서로 하나"},
        "small_workplace": {"type": ["boolean", "null"], "description": "상시근로자(사업주 제외) 5인 미만이면 true, 5인 이상이면 false, 모르면 null"},
        "weekly_hours": {"type": ["number", "null"], "description": "1주 소정근로시간. 1주 실제 근로시간과 40 중 작은 값"},
        "min_wage": {"type": ["integer", "null"], "description": "근무한 해에 적용되는 최저시급(원)"},
        "weekly_holiday_pay": {"type": ["integer", "null"], "description": "1주에 받아야 할 주휴수당(원). 15시간 미만이면 0"},
        "unpaid_krw": {"type": ["integer", "null"], "description": "근무 기간 전체의 미지급액 합계(원). 없으면 0"},
        "law_article": {"type": ["string", "null"], "description": "대표 쟁점의 근거 조문(예: '근로기준법 제55조'). 정상지급·판단불가는 null"},
    },
    "required": ["issue", "small_workplace", "weekly_hours", "min_wage", "weekly_holiday_pay", "unpaid_krw", "law_article"],
    "additionalProperties": False,
}

SCORING = """# L 도메인 채점 규칙: scorer.py가 읽음
# match: exact(정규화 후 일치) · enum · bool · numeric(허용오차 tol)
# 정답이 null 이면 출력도 null(또는 빈 값)이어야 정답
# core: true 인 필드가 모두 맞으면 Exact Match(보조 지표)
# 수치 필드 가중치 합 6/10: 점수의 60%가 회귀(숫자) 출력에서 나오는 회귀형 과제
fields:
  issue:              {match: enum,    weight: 2, core: true}
  small_workplace:    {match: bool,    weight: 1}
  weekly_hours:       {match: numeric, weight: 1, tol: 0}
  min_wage:           {match: numeric, weight: 1, tol: 0}
  weekly_holiday_pay: {match: numeric, weight: 1, tol: 10}
  unpaid_krw:         {match: numeric, weight: 3, tol: 10, core: true}
  law_article:        {match: exact,   weight: 1}
"""

# ------------------------------------------------------------------ 문장 만들기
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
Q_EVAL = ["덜 받은 돈이 있나요?", "제가 더 받아야 할 돈이 있으면 얼마인지 알려 주세요.", "이대로 받은 게 맞는지 궁금해요.",
          "혹시 못 받은 금액이 있을까요?", "정산이 제대로 된 건지 확인하고 싶어요."]


def night_overlap(s: int, e: int) -> int:
    """s시~e시(자정을 넘으면 e > 24로 표기) 중 22시~30시(다음 날 6시)와 겹치는 시간"""
    return max(0, min(e, 30) - max(s, 22))


def time_text(sc: dict) -> str:
    if sc.get("shift"):
        s, e = sc["shift"]
        return f"{s}시부터 {'다음 날 ' if e > 24 else ''}{e % 24 if e != 24 else 24}시까지"
    h = sc["hours"]
    return f"하루 {h:g}시간씩"


def case_of(sc: dict) -> dict:
    """문항 설정 → 규칙엔진 입력"""
    if sc.get("shift"):
        s, e = sc["shift"]
        hours, night = e - s, night_overlap(s, e)
    else:
        hours, night = sc.get("hours"), 0
    wage = sc.get("wage")
    size = sc.get("size")
    if sc.get("ws"):
        w = WS[sc["ws"]]
        wage = w[f"w{sc['year']}"] if wage is None else wage
        size = w["size"] if size is None else size
    if sc.get("missing") == "hours":
        hours = None
    if sc.get("missing") == "size":
        size = None
    return {"year": sc["year"], "weeks": sc["weeks"], "days": len(sc["days"]), "hours_per_day": hours,
            "night_per_day": night, "wage": wage, "size": size, "holiday_paid": sc.get("hol", False),
            "premium_paid": sc.get("prem", False)}


def hol_text(sc: dict, rng: random.Random) -> str:
    if sc.get("hol"):
        return rng.choice(["주휴수당은 받았어요", "주휴수당도 따로 받았어요"])
    return rng.choice(["주휴수당은 따로 없었어요", "주휴수당 얘기는 아예 없었어요", "주휴수당은 못 받았어요"])


def prem_text(sc: dict, rng: random.Random) -> str:
    c = case_of(sc)
    over = c["days"] * (c["hours_per_day"] or 0) > 40
    if not (c["night_per_day"] or over):
        return ""
    what = "밤늦게 일한 시간" if c["night_per_day"] else "주 40시간을 넘긴 시간"
    if sc.get("prem"):
        return f" {what}에 대한 수당도 더 받았어요."
    return f" {what}도 다른 시간과 같은 시급으로만 받았어요."


def cop(t: str) -> str:
    """'씩'으로 끝나면 '이었', '까지'로 끝나면 '였'"""
    return "이었" if t.endswith("씩") else "였"


def period_text(sc: dict) -> str:
    m, d = sc["start"]
    return f"{sc['year']}년 {m}월 {d}일부터 {sc['weeks']}주 동안"


def render_eval(sc: dict, rng: random.Random) -> str:
    days = "·".join(sc["days"])
    q = rng.choice(Q_EVAL)
    hol, prem = hol_text(sc, rng), prem_text(sc, rng)
    t = time_text(sc) if sc.get("missing") != "hours" else None
    if sc["type"] == "K":
        name = WS[sc["ws"]]["name"]
        v = sc.get("style", rng.randrange(3))
        if v == 0:
            return f"한빛로 '{name}'에서 {period_text(sc)} {days}요일마다 {t} 일했어요. 시급은 가게에서 정한 금액 그대로 받았고 {hol}.{prem} {q}"
        if v == 1:
            return f"{period_text(sc)} 한빛로에 있는 '{name}'에서 알바를 했습니다. 근무는 {days}요일 {t}{cop(t)}고, 시급은 사장님이 정한 대로 받았어요. {hol}.{prem} {q}"
        return f"'{name}'(한빛로) 알바생이에요. {period_text(sc)} {days}요일에 {t} 근무했어요. 시급은 처음 정한 그대로였고 {hol}.{prem} {q}"
    c = case_of(sc)
    ind = sc["industry"]
    size_t = f" 사장님을 빼고 직원은 {sc['size']}명이에요." if sc.get("missing") != "size" else ""
    mw_t = f" 그해 최저시급은 {W.MIN_WAGE[sc['year']]:,}원이라고 알고 있어요." if sc.get("show_mw", True) else ""
    if t is None:
        t_part = "근무 시간은 날마다 달라서 정확히 기억이 안 나요"
        return f"{ind}에서 {period_text(sc)} {days}요일에 일했어요. {t_part}. 시급은 {c['wage']:,}원이었고 {hol}.{size_t}{mw_t} {q}"
    v = sc.get("style", rng.randrange(3))
    if v == 0:
        return f"{ind}에서 {period_text(sc)} {days}요일마다 {t} 일했어요. 시급은 {c['wage']:,}원이었고 {hol}.{prem}{size_t}{mw_t} {q}"
    if v == 1:
        return f"{period_text(sc)} {ind} 알바를 했어요. 일한 요일은 {days}, {t}{cop(t)}고 시급 {c['wage']:,}원을 받았어요. {hol}.{prem}{size_t}{mw_t} {q}"
    return f"{ind} 아르바이트 정산 문의입니다. {period_text(sc)} {days}요일에 {t} 근무했고 시급은 {c['wage']:,}원이었어요. {hol}.{prem}{size_t}{mw_t} {q}"


def render_train(sc: dict, rng: random.Random) -> str:
    """학습 문항은 평가와 문장 틀을 다르게 씀(평가셋 누수 방지)"""
    days = ", ".join(sc["days"])
    t = time_text(sc) if sc.get("missing") != "hours" else "시간 기록 없음"
    c = case_of(sc)
    hol = "받음" if sc.get("hol") else "못 받음"
    extra = ""
    if c["night_per_day"] or (c["days"] * (c["hours_per_day"] or 0) > 40):
        extra = f" / 가산수당: {'받음' if sc.get('prem') else '시급만 받음'}"
    if sc["type"] == "K":
        head = f"근무지: 한빛로 {WS[sc['ws']]['name']}"
        wage = "시급: 가게 기준 시급"
        size = ""
    else:
        head = f"근무지: {sc['industry']}"
        wage = f"시급: {c['wage']:,}원"
        size = f" / 직원 수(사업주 제외): {sc['size']}명" if sc.get("missing") != "size" else ""
        if sc.get("show_mw", True):
            size += f" / 그해 최저시급 {W.MIN_WAGE[sc['year']]:,}원"
    v = rng.randrange(2)
    m, d = sc["start"]
    if v == 0:
        return (f"알바비 정산 좀 봐 주세요. {head} / 기간: {sc['year']}.{m}.{d}부터 {sc['weeks']}주 / 요일: {days} / "
                f"시간: {t} / {wage} / 주휴수당: {hol}{extra}{size}")
    return (f"{head}에서 일한 내역이에요. {sc['year']}년 {m}월 {d}일 시작, {sc['weeks']}주, 매주 {days}, {t}. "
            f"{wage}, 주휴수당 {hol}{extra}{size}. 받을 돈 계산 부탁해요.")


def gold_docs(sc: dict, gold: dict) -> list[str]:
    if sc["type"] == "K":
        return [sc["ws"], f"mw_{sc['year']}"]
    docs = []
    if gold["issue"] == "판단불가":
        docs.append("rule_format")
    elif gold["weekly_holiday_pay"] == 0 or (gold["weekly_hours"] or 0) >= 40:
        docs += ["rule_holiday", "rule_hours"]
    if gold["issue"] == "가산미지급" or (gold["small_workplace"] and case_of(sc)["night_per_day"]):
        docs += ["rule_premium", "rule_small"]
    return docs


def make_item(iid: str, sc: dict, rng: random.Random, train: bool = False) -> dict:
    gold = W.compute(case_of(sc))
    text = render_train(sc, rng) if train else render_eval(sc, rng)
    it = {"id": iid, "type": sc["type"], "input": text, "gold": gold}
    if not train:
        it["gold_doc_ids"] = gold_docs(sc, gold)
    return it


# ------------------------------------------------------------------ 시드 10건(K4·R3·F3)
SEED = [
    dict(type="K", ws="ws01", year=2027, start=(1, 4), weeks=4, days=["월", "수", "금"], hours=6, hol=False, style=0),
    dict(type="K", ws="ws07", year=2027, start=(2, 5), weeks=3, days=["금", "토", "일"], shift=(18, 24), hol=True, prem=False, style=1),
    dict(type="K", ws="ws05", year=2026, start=(9, 7), weeks=4, days=["월", "화", "수", "목", "금"], hours=4, hol=False, style=2),
    dict(type="K", ws="ws12", year=2027, start=(3, 6), weeks=2, days=["토", "일"], shift=(20, 26), hol=False, prem=False, style=0),
    dict(type="R", industry="편의점", size=6, year=2026, start=(10, 6), weeks=3, days=["화", "목", "토"], hours=5, wage=10000, hol=False, style=0),
    dict(type="R", industry="카페", size=9, year=2027, start=(1, 11), weeks=4, days=["월", "화", "수", "목", "금"], hours=6, wage=11000, hol=True, style=1),
    dict(type="R", industry="고깃집", size=5, year=2026, start=(8, 3), weeks=2, days=["수", "목", "금", "토"], shift=(17, 24), wage=10320, hol=True, prem=False, style=2),
    dict(type="F", industry="학원", size=4, year=2026, start=(9, 1), weeks=4, days=["화", "목"], hours=7, wage=10400, hol=False, style=0),
    dict(type="F", industry="물류센터", size=12, year=2026, start=(11, 2), weeks=2, days=["월", "화", "수", "목", "금", "토"], hours=8, wage=10500, hol=True, prem=False, style=1),
    dict(type="F", industry="동네 PC방", size=3, year=2027, start=(1, 18), weeks=3, days=["월", "화", "수"], shift=(22, 28), wage=10700, hol=True, prem=False, style=2),
]

# ------------------------------------------------------------------ 교체 풀 20건(K8·R7·F5, 공개 저장소에는 올리지 않음)
POOL = [
    dict(type="K", ws="ws02", year=2027, start=(2, 1), weeks=3, days=["월", "수", "금"], hours=5, hol=False),
    dict(type="K", ws="ws03", year=2026, start=(10, 5), weeks=4, days=["월", "화", "수"], shift=(22, 27), hol=True, prem=False),
    dict(type="K", ws="ws04", year=2027, start=(1, 25), weeks=2, days=["화", "목", "토", "일"], hours=5, hol=False),
    dict(type="K", ws="ws06", year=2026, start=(8, 10), weeks=4, days=["월", "수", "금", "토"], hours=4, hol=False),
    dict(type="K", ws="ws09", year=2027, start=(3, 13), weeks=4, days=["토", "일"], hours=6, hol=False),
    dict(type="K", ws="ws10", year=2026, start=(7, 6), weeks=3, days=["월", "화", "수", "목"], hours=6, hol=True),
    dict(type="K", ws="ws13", year=2027, start=(2, 15), weeks=2, days=["월", "화", "수", "목"], shift=(18, 23), hol=True, prem=False),
    dict(type="K", ws="ws14", year=2026, start=(9, 14), weeks=2, days=["월", "화", "수", "목", "금"], hours=8, hol=False),
    dict(type="R", industry="빵집", size=7, year=2026, start=(10, 12), weeks=4, days=["월", "수", "금"], hours=6, wage=10300, hol=True),
    dict(type="R", industry="찜질방", size=11, year=2027, start=(1, 4), weeks=2, days=["금", "토", "일"], shift=(20, 26), wage=10800, hol=True, prem=False),
    dict(type="R", industry="중식당", size=5, year=2027, start=(2, 8), weeks=4, days=["월", "화", "수", "목", "금"], hours=4, wage=10700, hol=False),
    dict(type="R", industry="놀이공원 매표소", size=20, year=2026, start=(5, 4), weeks=3, days=["토", "일"], hours=8, wage=10320, hol=True),
    dict(type="R", industry="버블티 가게", size=3, year=2027, start=(3, 1), weeks=4, days=["월", "화", "목", "금"], hours=5, wage=10500, hol=False),
    dict(type="R", industry="사진관", size=2, year=2026, start=(6, 1), weeks=2, days=["화", "수", "목", "금", "토"], hours=3.5, wage=10320, hol=False),
    dict(type="R", industry="택배 분류센터", size=30, year=2027, start=(1, 11), weeks=3, days=["월", "화", "수", "목", "금"], shift=(19, 25), wage=11000, hol=True, prem=True),
    dict(type="F", industry="식당", size=6, year=2026, start=(9, 7), weeks=4, days=["월", "화", "수", "목", "금"], wage=10500, hol=False, missing="hours"),
    dict(type="F", industry="편의점", size=None, year=2026, start=(11, 2), weeks=2, days=["금", "토", "일"], shift=(22, 28), wage=10320, hol=True, prem=False, missing="size"),
    dict(type="F", industry="빨래방", size=3, year=2027, start=(1, 4), weeks=4, days=["월", "화", "수"], hours=6, wage=10200, hol=False),
    dict(type="F", industry="동네 마트", size=4, year=2026, start=(8, 3), weeks=3, days=["월", "화", "수", "목", "금", "토"], hours=7.5, wage=10400, hol=True),
    dict(type="F", industry="입시학원", size=8, year=2027, start=(2, 1), weeks=4, days=["월", "수", "금"], hours=5, wage=10800, hol=False),
]

TRAIN_INDUSTRIES = ["주유소", "영화관 매점", "피자집", "세차장", "독서실", "약국 보조", "도서관 카페", "꽃가게", "족발집", "만두가게",
                    "키즈카페", "인형뽑기방", "코인세탁소", "제과점", "학교 앞 분식집", "대형마트 시식", "호프집", "스크린골프장"]


def random_train(rng: random.Random, n: int) -> list[dict]:
    out = []
    train_ws = [w for w in WS if WS[w]["use"] == "train"]
    for i in range(n):
        typ = rng.choices(["K", "R", "F"], weights=[4, 4, 3])[0]
        year = rng.choice([2026, 2027])
        nd = rng.choice([2, 2, 3, 3, 4, 5, 5, 6])
        days = sorted(rng.sample(DAYS, nd), key=DAYS.index)
        sc = dict(type=typ, year=year, start=(rng.randint(1, 12) if year == 2027 else rng.randint(3, 10), rng.randint(1, 25)),
                  weeks=rng.randint(1, 6), days=days, hol=rng.random() < 0.45, prem=rng.random() < 0.3)
        if rng.random() < 0.4:
            s = rng.choice([17, 18, 19, 20, 21, 22])
            sc["shift"] = (s, s + rng.choice([4, 5, 6]))
        else:
            sc["hours"] = rng.choice([3, 4, 4.5, 5, 6, 6, 7, 8])
        if typ == "K":
            sc["ws"] = rng.choice(train_ws)
        else:
            sc["industry"] = rng.choice(TRAIN_INDUSTRIES)
            sc["size"] = rng.choice([2, 3, 4, 5, 6, 8, 12, 25])
            mw = W.MIN_WAGE[year]
            sc["wage"] = rng.choice([mw - 320, mw - 120, mw, mw, mw + 180, mw + 480])
            if typ == "F" and rng.random() < 0.25:
                sc["missing"] = rng.choice(["hours", "size"]) if sc.get("shift") else "hours"
                if sc["missing"] == "size":
                    sc["size"] = None
        out.append(sc)
    return out


TEAM_RULES = [
    "K(지식형): 입력에는 한빛로 사업장 이름만 쓰고 시급·직원 수는 쓰지 않는다. gold_doc_ids에 사업장 문서(ws01~ws14)와 그해 최저임금 문서(mw_2026 또는 mw_2027)를 반드시 적는다",
    "R(추론형): 판단에 필요한 숫자(근무 기간·요일·시간·시급·직원 수·그해 최저시급)를 입력에 모두 쓴다. 곱셈·합산만으로 답이 나와야 한다",
    "F(형식·판단형): 15시간 미만 주휴 없음, 40시간 상한, 5인 미만 가산 미적용, 정보 부족 시 판단불가, 쟁점 순서 중 하나를 반드시 건드린다",
    "gold는 손으로 계산하지 않고 `python data/L/build/wage_rules.py --year … --weeks … --days … --hours … --wage … --size …`의 출력을 그대로 붙인다",
    "근무 기간은 한 해(2026년 또는 2027년) 안에서만 쓰고, 하루 근무는 8시간 이하로 한다",
    "사업장은 평가용 ws01~ws14만 쓴다(ws15~ws24는 학습용). 시드 문항과 같은 조건을 다시 쓰지 않는다",
    "실존 상호·실명·개인정보를 넣지 않는다. 본인이나 지인의 실제 근로 정보도 쓰지 않는다",
]

TOOLS_SPEC = f"""# L 도메인 Tool 후보(8~10회차)

> 기준일 {AS_OF}. 시그니처는 제안이며 차승희 박사 회차에서 확정한다. 데이터 원본은 `build/wage_db.json`, 계산 규칙은 `build/wage_rules.py`.

| Tool | 입력 | 출력 | 쓰임 |
|---|---|---|---|
| `calc_wage` | year, weeks, days, hours_per_day, night_per_day, wage, size, holiday_paid, premium_paid | 정산 JSON(issue·weekly_hours·min_wage·weekly_holiday_pay·unpaid_krw·law_article) | 회귀형 필드를 LLM이 직접 계산하지 않고 계산기에 맡김. R 유형에서 효과가 가장 클 것으로 예상 |
| `lookup_workplace` | 사업장 이름 | 업종, 상시근로자 수, 연도별 약정 시급 | K 유형: 입력에 없는 시급·규모 조회 |
| `get_min_wage` | year | 최저시급(원) | 2027년 고시처럼 모델이 모르는 값 조회 |
"""


def main() -> None:
    rng = random.Random(20261101)
    (DATA / "corpus").mkdir(exist_ok=True)
    SOL.mkdir(parents=True, exist_ok=True)
    db = {"as_of": AS_OF, "street": "한빛로(가상)", "min_wage": {str(k): v for k, v in W.MIN_WAGE.items()},
          "workplaces": list(WS.values())}
    (HERE / "wage_db.json").write_text(json.dumps(db, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = []
    for w in WS.values():
        md = ws_doc(w)
        (DATA / "corpus" / f"{w['id']}.md").write_text(md, encoding="utf-8")
        manifest.append({"doc_id": w["id"], "title": f"{w['name']}(가상 사업장)", "file": f"corpus/{w['id']}.md",
                         "source": SOURCE, "license": LICENSE, "as_of": AS_OF, "chars": len(md)})
    for did, (title, md) in {**MW_DOCS, **RULE_DOCS}.items():
        (DATA / "corpus" / f"{did}.md").write_text(md, encoding="utf-8")
        src = "최저임금 고시 요약(공공정보)" if did.startswith("mw_") else "자체 작성 해설"
        manifest.append({"doc_id": did, "title": title, "file": f"corpus/{did}.md", "source": src,
                         "license": LICENSE if did.startswith("rule_") else "공공누리 제1유형 요약", "as_of": AS_OF, "chars": len(md)})
    (DATA / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    (DATA / "schema.json").write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "scoring.yaml").write_text(SCORING, encoding="utf-8")

    seed = [make_item(f"L-S-{i:02d}", sc, rng) for i, sc in enumerate(SEED, 1)]
    for p in ("test_seed.json", "test_dataset.json"):
        (DATA / p).write_text(json.dumps(seed, ensure_ascii=False, indent=1), encoding="utf-8")

    cnt = {"K": 0, "R": 0, "F": 0}
    pool = []
    for sc in POOL:
        cnt[sc["type"]] += 1
        it = make_item(f"L-P-{sc['type']}{cnt[sc['type']]:02d}", sc, rng)
        it["family"] = sc["type"]
        pool.append(it)
    (SOL / "test_pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")

    alloc = {"K": 8, "R": 7, "F": 5}
    items, n = [], 0
    for t, k in alloc.items():
        for _ in range(k):
            n += 1
            items.append({"id": f"L-T-{n:02d}", "type": t, "input": "",
                          "gold": {k2: None for k2 in SCHEMA["properties"]}, "gold_doc_ids": []})
    tmpl = {"domain": "L", "allocation": alloc, "rules": TEAM_RULES, "example": seed[0], "items": items}
    (DATA / "my_item_template.json").write_text(json.dumps(tmpl, ensure_ascii=False, indent=1), encoding="utf-8")

    train = [make_item(f"L-L-{i:03d}", sc, rng, train=True) for i, sc in enumerate(random_train(rng, 180), 1)]
    with open(DATA / "train_seed.jsonl", "w", encoding="utf-8") as f:
        for it in train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (DATA / "tools_spec.md").write_text(TOOLS_SPEC, encoding="utf-8")

    from collections import Counter
    for name, arr in (("시드", seed), ("교체 풀", pool), ("학습 시드", train)):
        print(f"{name} {len(arr)}건 · 유형 {dict(Counter(x['type'] for x in arr))} · 쟁점 {dict(Counter(x['gold']['issue'] for x in arr))}")
    print(f"코퍼스 {len(manifest)}건(사업장 {len(WS)} · 최저임금 2 · 해설 {len(RULE_DOCS)})")


if __name__ == "__main__":
    main()
