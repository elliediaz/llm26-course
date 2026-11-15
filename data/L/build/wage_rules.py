"""L 도메인(알바 급여 정산) 규칙엔진: 근무 조건 → 정답 JSON. 문항의 정답은 모두 이 함수로 계산한다.

기준일 2026-11-01 고정. 이 과정의 계산 규약(법률 자문 아님)
  - 최저시급: 2026년 10,320원, 2027년 10,700원(최저임금법 제10조 고시). 근무 기간은 한 해 안에서만 다룸
  - 적용 시급 = max(약정 시급, 그해 최저시급)
  - 1주 소정근로시간 = min(1주 실제 근로시간, 40). 40시간을 넘는 시간은 연장근로
  - 주휴수당(근로기준법 제55조): 소정근로시간 15시간 이상·개근이면 (소정근로시간 ÷ 40) × 8 × 적용 시급, 원 단위 버림
  - 가산수당(근로기준법 제56조): 연장·야간(22시~06시) 근로에 적용 시급의 50%. 상시 5인 미만 사업장은 적용하지 않음(제11조)
  - 미지급액 = 근무 주 수 × (1주 받아야 할 돈 - 1주 받은 돈), 원 단위 버림
  - 대표 쟁점 순서: 최저임금미달 > 주휴미지급 > 가산미지급 > 정상지급. 판단에 필요한 정보가 없으면 판단불가, 수치 필드는 null
"""
from __future__ import annotations

import math
from fractions import Fraction as F

AS_OF = "2026-11-01"
MIN_WAGE = {2026: 10320, 2027: 10700}
ARTICLE = {
    "최저임금미달": "최저임금법 제6조",
    "주휴미지급": "근로기준법 제55조",
    "가산미지급": "근로기준법 제56조",
    "정상지급": None,
    "판단불가": None,
}


def compute(c: dict) -> dict:
    """c: year, weeks, days(요일 수), hours_per_day, night_per_day, wage(약정 시급), size(상시근로자 수 또는 None),
    holiday_paid, premium_paid, missing(None 또는 'hours'·'wage'·'size')"""
    size = c.get("size")
    small = None if size is None else size < 5
    missing = c.get("missing")
    actual = c["days"] * F(str(c["hours_per_day"])) if c.get("hours_per_day") is not None else None
    night = c["days"] * F(str(c.get("night_per_day", 0)))
    over = max(0, actual - 40) if actual is not None else 0
    # 5인 이상인지 몰라도 가산수당이 걸리지 않으면 판단할 수 있음
    needs_size = (night + over) > 0
    if missing or actual is None or c.get("wage") is None or (needs_size and size is None):
        return {"issue": "판단불가", "small_workplace": small, "weekly_hours": None, "min_wage": None,
                "weekly_holiday_pay": None, "unpaid_krw": None, "law_article": None}
    mw = MIN_WAGE[c["year"]]
    agreed = c["wage"]
    hourly = max(agreed, mw)
    sched = min(actual, 40)
    holiday_due = math.floor(sched * 8 * hourly / 40) if sched >= 15 else 0
    premium_due = 0 if small else F(1, 2) * hourly * (night + over)
    due = hourly * actual + holiday_due + premium_due
    holiday_paid = math.floor(sched * 8 * agreed / 40) if (c.get("holiday_paid") and sched >= 15) else 0
    premium_paid = F(1, 2) * agreed * (night + over) if (c.get("premium_paid") and not small) else 0
    paid = agreed * actual + holiday_paid + premium_paid
    unpaid = math.floor(c["weeks"] * (due - paid))
    if agreed < mw:
        issue = "최저임금미달"
    elif holiday_due > 0 and not c.get("holiday_paid"):
        issue = "주휴미지급"
    elif premium_due > 0 and not c.get("premium_paid"):
        issue = "가산미지급"
    else:
        issue = "정상지급"
    sched = int(sched) if sched == int(sched) else float(sched)
    return {"issue": issue, "small_workplace": small, "weekly_hours": sched, "min_wage": mw,
            "weekly_holiday_pay": holiday_due, "unpaid_krw": unpaid, "law_article": ARTICLE[issue]}


def explain(c: dict) -> str:
    """정답이 어떻게 나왔는지 한 줄 설명(해설·채점 확인용)"""
    g = compute(c)
    if g["issue"] == "판단불가":
        return "판단에 필요한 정보(근무 시간·시급·사업장 규모 중 하나)가 없어 판단불가"
    actual = c["days"] * c["hours_per_day"]
    actual = int(actual) if actual == int(actual) else actual
    hourly = max(c["wage"], g["min_wage"])
    s = f"주 {actual}시간(소정 {g['weekly_hours']}) · 적용 시급 {hourly:,} · 주휴 {g['weekly_holiday_pay']:,}"
    return s + f" · 미지급 {g['unpaid_krw']:,}원({c['weeks']}주) → {g['issue']}"


def main() -> None:
    """정답 계산기: 팀 문항의 gold를 손으로 계산하지 않고 이 명령으로 만든다."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description="알바 급여 정산 정답 계산(L 도메인 규칙엔진)")
    ap.add_argument("--year", type=int, required=True, choices=sorted(MIN_WAGE))
    ap.add_argument("--weeks", type=int, required=True, help="근무 주 수(개근 가정)")
    ap.add_argument("--days", type=int, required=True, help="1주 근무 일수")
    ap.add_argument("--hours", type=float, help="하루 근무 시간(모르면 생략 → 판단불가)")
    ap.add_argument("--night", type=float, default=0, help="하루 중 22시~06시 근무 시간")
    ap.add_argument("--wage", type=int, help="약정 시급(모르면 생략 → 판단불가)")
    ap.add_argument("--size", type=int, help="상시근로자 수(사업주 제외). 모르면 생략")
    ap.add_argument("--holiday-paid", choices=["yes", "no"], default="no", help="주휴수당을 받았는가")
    ap.add_argument("--premium-paid", choices=["yes", "no"], default="no", help="연장·야간 가산수당을 받았는가")
    a = ap.parse_args()
    c = {"year": a.year, "weeks": a.weeks, "days": a.days, "hours_per_day": a.hours, "night_per_day": a.night,
         "wage": a.wage, "size": a.size, "holiday_paid": a.holiday_paid == "yes", "premium_paid": a.premium_paid == "yes"}
    print(json.dumps(compute(c), ensure_ascii=False))
    print("# " + explain(c))


if __name__ == "__main__":
    main()
