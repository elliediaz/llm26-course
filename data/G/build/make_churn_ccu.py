"""1회차 기본 신경망 실습용 G 도메인 합성 데이터 두 가지를 만든다(가상 수집형 RPG '프로젝트 루멘').

1) churn.csv: 유저 20,000명의 최근 활동 12개 → 다음 14일 이탈 여부(분류) · 다음 7일 일평균 플레이 시간(회귀)
   실제 게임 운영 지표의 모양을 흉내 냄: 접속이 뜸하고 문의가 잦고 길드가 없으면 이탈이 늘고,
   결제는 소수에 몰리며, 플레이 시간은 오른쪽으로 긴 분포
2) ccu_hourly.csv: 2026년 1년치 시간별 동시 접속자 수(8,760행)
   하루 주기(저녁 21시 최고), 주말 상승, 매주 수요일 06~08시 정기 점검(0명), 업데이트·이벤트 직후 급증, 완만한 감소 추세

    python data/G/build/make_churn_ccu.py        # 같은 시드면 항상 같은 파일
"""
import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261116)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_churn(n=20000):
    e = rng.normal(0, 1, n)                                  # 숨은 몰입도(데이터에는 없음)
    whale = rng.random(n) < 0.04                             # 고액 결제 유저
    days_since_join = rng.integers(3, 900, n)
    login_days_14 = np.clip(np.round(7 + 3.2 * e + rng.normal(0, 1.8, n)), 0, 14).astype(int)
    avg_play_min = np.clip(np.exp(3.6 + 0.55 * e + rng.normal(0, 0.45, n)), 3, 600).round(1)
    level = np.clip(np.round(15 + 0.06 * days_since_join + 6 * e + rng.normal(0, 6, n)), 1, 120).astype(int)
    level_up_14 = np.clip(np.round(rng.poisson(np.maximum(0.2, 2.2 + 1.3 * e - level / 60))), 0, 30).astype(int)
    payer = (rng.random(n) < sigmoid(-1.2 + 0.7 * e)) | whale
    pay_30 = np.where(payer, np.round(np.exp(9.4 + 0.6 * e + 1.6 * whale + rng.normal(0, 0.8, n)), -2), 0).astype(int)
    guild = (rng.random(n) < sigmoid(-0.3 + 0.9 * e)).astype(int)
    friends = np.clip(np.round(np.exp(1.6 + 0.5 * e + 0.6 * guild + rng.normal(0, 0.6, n))), 0, 200).astype(int)
    tickets_30 = rng.poisson(np.where(e < -0.8, 0.9, 0.25) + 0.5 * (pay_30 > 50000))
    pvp_win = np.clip(0.48 + 0.04 * e + rng.normal(0, 0.09, n), 0.05, 0.95).round(3)
    last_login_gap = np.clip(np.round(rng.exponential(np.exp(0.8 - 1.0 * e))), 0, 30).astype(int)
    event_join_30 = rng.poisson(np.exp(0.4 + 0.6 * e))
    logit = (-1.25 - 1.15 * e - 0.55 * guild + 0.32 * tickets_30 + 0.11 * last_login_gap
             - 0.05 * login_days_14 - 0.35 * (pay_30 > 0) + 0.2 * (days_since_join < 30))
    churn = (rng.random(n) < sigmoid(logit)).astype(int)
    next7 = np.where(churn == 1, rng.exponential(4, n),
                     avg_play_min * np.exp(rng.normal(0.05 * e, 0.35, n)) * (1 + 0.04 * event_join_30))
    next7_play_min = np.clip(next7, 0, 720).round(1)
    cols = dict(days_since_join=days_since_join, login_days_14=login_days_14, avg_play_min=avg_play_min, level=level,
                level_up_14=level_up_14, pay_30=pay_30, guild=guild, friends=friends, tickets_30=tickets_30,
                pvp_win=pvp_win, last_login_gap=last_login_gap, event_join_30=event_join_30,
                churn_14d=churn, next7_play_min=next7_play_min)
    with open(OUT / "churn.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id"] + list(cols))
        for i in range(n):
            w.writerow([f"U{i + 1:05d}"] + [cols[k][i] for k in cols])
    print("churn.csv", n, "이탈 비율", churn.mean().round(3), "결제 유저", (pay_30 > 0).mean().round(3))


def make_ccu():
    start = dt.datetime(2026, 1, 1)
    hours = 24 * 365
    rows = []
    daily = [0.34, 0.25, 0.19, 0.15, 0.13, 0.14, 0.18, 0.24, 0.30, 0.34, 0.38, 0.42,
             0.50, 0.52, 0.50, 0.52, 0.56, 0.62, 0.72, 0.86, 0.96, 1.00, 0.88, 0.58]
    events = {dt.date(2026, 1, 28): 1.35, dt.date(2026, 3, 4): 1.25, dt.date(2026, 5, 6): 1.45,
              dt.date(2026, 7, 15): 1.55, dt.date(2026, 9, 23): 1.3, dt.date(2026, 11, 11): 1.4}
    level = 52000.0
    boost = 1.0
    for h in range(hours):
        t = start + dt.timedelta(hours=h)
        if t.date() in events and t.hour == 8:
            boost = events[t.date()]
        boost = 1 + (boost - 1) * 0.985                      # 업데이트 효과는 며칠에 걸쳐 줄어듦
        level *= 0.99997                                     # 완만한 감소 추세
        weekend = 1.18 if t.weekday() >= 5 else 1.0
        if t.weekday() == 4 and t.hour >= 18:
            weekend = 1.08                                   # 금요일 저녁
        v = level * daily[t.hour] * weekend * boost * math.exp(rng.normal(0, 0.035))
        maint = t.weekday() == 2 and 6 <= t.hour < 8         # 수요일 정기 점검
        rows.append((t.strftime("%Y-%m-%d %H:00"), t.weekday(), t.hour, int(maint), 0 if maint else int(v)))
    with open(OUT / "ccu_hourly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "weekday", "hour", "maintenance", "ccu"])
        w.writerows(rows)
    print("ccu_hourly.csv", len(rows), "최대", max(r[4] for r in rows))


if __name__ == "__main__":
    make_churn()
    make_ccu()
