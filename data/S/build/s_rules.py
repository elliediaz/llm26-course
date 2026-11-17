"""S 도메인 계산·표기 규칙. 생성기와 팀 문항 정답 만들기에 같이 씀

  python data/S/build/s_rules.py winrate --w 8 --l 5 --d 1          # 승률 0.615
  python data/S/build/s_rules.py gb --a 50-30 --b 46-33               # 게임차 3.5
  python data/S/build/s_rules.py avg --h 152 --ab 471                 # 타율 0.323
  python data/S/build/s_rules.py era --er 41 --ip 160.1               # 평균자책점 2.30 (160.1 = 160과 1/3이닝)
  python data/S/build/s_rules.py series --scores 3:2 1:4 6:5          # 시리즈 결과 2:1
  python data/S/build/s_rules.py kda --k 41 --d 12 --a 77             # KDA 9.83

표기 규약(기록실 규약 policy_format과 같음)
  승률·타율 : 소수 셋째 자리(반올림), 0으로 시작(0.615)
  평균자책점·KDA : 소수 둘째 자리(반올림)
  게임차 : 0.5 단위
  시리즈·세트 결과 : '이긴 횟수:진 횟수'를 묻는 쪽 기준으로(2:1)
"""
from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal


def rnd(x: float, nd: int) -> str:
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def winrate(w: int, l: int) -> str:
    """승률 = 승 ÷ (승 + 패). 무승부는 빼고 계산함"""
    return rnd(w / (w + l), 3) if w + l else "0.000"


def games_behind(a: tuple[int, int], b: tuple[int, int]) -> str:
    """앞선 팀 a(승, 패)와 뒤진 팀 b의 게임차 = ((a승 - b승) + (b패 - a패)) ÷ 2"""
    return str(((a[0] - b[0]) + (b[1] - a[1])) / 2)


def batting_avg(h: int, ab: int) -> str:
    return rnd(h / ab, 3) if ab else "0.000"


def ip_to_outs(ip: str | float) -> int:
    """이닝 표기 160.1 → 아웃 카운트 481 (소수 자리는 1/3이닝 단위)"""
    whole, _, frac = str(ip).partition(".")
    return int(whole) * 3 + (int(frac) if frac else 0)


def era(er: int, ip: str | float) -> str:
    outs = ip_to_outs(ip)
    return rnd(er * 27 / outs, 2) if outs else "0.00"


def series(scores: list[str]) -> str:
    """'우리 점수:상대 점수' 목록 → '이긴 경기:진 경기'"""
    w = l = 0
    for s in scores:
        a, b = (int(x) for x in s.split(":"))
        w += a > b
        l += a < b
    return f"{w}:{l}"


def kda(k: int, d: int, a: int) -> str:
    """KDA = (킬 + 어시스트) ÷ 데스. 데스가 0이면 데스를 1로 보고 계산함"""
    return rnd((k + a) / max(d, 1), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("winrate"); p.add_argument("--w", type=int, required=True); p.add_argument("--l", type=int, required=True); p.add_argument("--d", type=int, default=0)
    p = sub.add_parser("gb"); p.add_argument("--a", required=True, help="승-패"); p.add_argument("--b", required=True)
    p = sub.add_parser("avg"); p.add_argument("--h", type=int, required=True); p.add_argument("--ab", type=int, required=True)
    p = sub.add_parser("era"); p.add_argument("--er", type=int, required=True); p.add_argument("--ip", required=True)
    p = sub.add_parser("series"); p.add_argument("--scores", nargs="+", required=True)
    p = sub.add_parser("kda"); p.add_argument("--k", type=int, required=True); p.add_argument("--d", type=int, required=True); p.add_argument("--a", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "winrate":
        print(winrate(a.w, a.l))
    elif a.cmd == "gb":
        print(games_behind(tuple(map(int, a.a.split("-"))), tuple(map(int, a.b.split("-")))))
    elif a.cmd == "avg":
        print(batting_avg(a.h, a.ab))
    elif a.cmd == "era":
        print(era(a.er, a.ip))
    elif a.cmd == "series":
        print(series(a.scores))
    elif a.cmd == "kda":
        print(kda(a.k, a.d, a.a))


if __name__ == "__main__":
    main()
