"""1회차 소규모 Encoder-Decoder 학습 과제: 한글 수사 → 숫자 표기 합성 데이터.

  python data/common/numerals/make_numerals.py            # train.jsonl 2만 쌍, valid.jsonl 1천 쌍
  (함께 생성) long.jsonl                                  # 입력 16자 이상만 모은 시험셋(병목 확인용)

예
  삼만 오천이백 원          → 35200원
  이천이십육년 시월 이십육일  → 2026-10-26
  백이십삼 회               → 123회
  삼십오 퍼센트             → 35%

외부 데이터·라이선스 문제가 없다(규칙으로 생성). 띄어쓰기는 무작위로 흔든다.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIG = "영일이삼사오육칠팔구"
SMALL = [(1000, "천"), (100, "백"), (10, "십")]


def under_10000(n: int, rng: random.Random, p_drop: float = 0.9) -> str:
    """1~9999 → 한글 수사(예: 3200 → 삼천이백). p_drop 확률로 '일'을 생략한다(천·일천 둘 다 쓰임)."""
    out = ""
    for v, u in SMALL:
        d = n // v
        n %= v
        if d:
            one = d == 1 and rng.random() < p_drop
            out += ("" if one else DIG[d]) + u
    if n:
        out += DIG[n]
    return out


def sino(n: int, rng: random.Random) -> str:
    """0~9억 미만 정수 → 한글 수사. 만 단위로 띄어 쓰기도 한다."""
    if n == 0:
        return "영"
    parts = []
    eok, rest = divmod(n, 10**8)
    man, low = divmod(rest, 10**4)
    if eok:
        parts.append(under_10000(eok, rng) + "억")
    if man:
        parts.append(("" if man == 1 and rng.random() < 0.7 else under_10000(man, rng)) + "만")
    if low:
        parts.append(under_10000(low, rng))
    sep = " " if rng.random() < 0.5 else ""
    return sep.join(parts)


MONTH = {6: "유월", 10: "시월"}


def sample(rng: random.Random, max_amount: int = 9 * 10**8) -> tuple[str, str]:
    kind = rng.random()
    sp = lambda: " " if rng.random() < 0.6 else ""  # noqa: E731
    if kind < 0.45:                                   # 금액(짧은 것부터 억 단위까지 고르게)
        n = rng.choice([rng.randint(1, 999) * 10, rng.randint(1, 9999) * 100, rng.randint(1, 10**7),
                        rng.randint(10**7, max_amount)])
        return f"{sino(n, rng)}{sp()}원", f"{n}원"
    if kind < 0.70:                                   # 날짜
        y, m, d = rng.randint(1990, 2035), rng.randint(1, 12), rng.randint(1, 28)
        ms = MONTH.get(m, under_10000(m, rng, p_drop=1.0) + "월")
        return f"{sino(y, rng)}년{sp()}{ms}{sp()}{under_10000(d, rng, p_drop=1.0)}일", f"{y}-{m:02d}-{d:02d}"
    if kind < 0.88:                                   # 횟수·개수
        n = rng.randint(1, 9999)
        unit = rng.choice(["회", "개", "번째", "명"])
        return f"{sino(n, rng)}{sp()}{unit}", f"{n}{unit}"
    n = rng.randint(1, 100)                           # 퍼센트
    return f"{sino(n, rng)}{sp()}퍼센트", f"{n}%"


def make(n: int, seed: int, **kw) -> list[dict]:
    rng = random.Random(seed)
    return [dict(zip(("src", "tgt"), sample(rng, **kw))) for _ in range(n)]


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure(out_dir: Path = HERE, n_train: int = 20000, n_valid: int = 1000) -> None:
    """학습 스크립트가 호출. 파일이 없을 때만 만든다."""
    if not (out_dir / "train.jsonl").exists():
        write(out_dir / "train.jsonl", make(n_train, 1))
        write(out_dir / "valid.jsonl", make(n_valid, 2))
    if not (out_dir / "long.jsonl").exists():
        # 학습 분포 안에서 입력이 긴 것만 모은 시험셋(고정 길이 벡터 병목 확인용)
        rows = [r for r in make(8000, 3) if len(r["src"]) >= 16][:500]
        write(out_dir / "long.jsonl", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--valid", type=int, default=1000)
    a = ap.parse_args()
    for p in ("train.jsonl", "valid.jsonl", "long.jsonl"):
        (HERE / p).unlink(missing_ok=True)
    ensure(HERE, a.train, a.valid)
    for p in ("train.jsonl", "valid.jsonl", "long.jsonl"):
        rows = [json.loads(x) for x in open(HERE / p, encoding="utf-8")]
        lens = [len(r["src"]) for r in rows]
        print(f"{p:12s} {len(rows):6d}쌍  입력 길이 {min(lens)}~{max(lens)}  예: {rows[0]['src']} → {rows[0]['tgt']}")
