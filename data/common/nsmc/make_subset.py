"""NSMC(네이버 영화 리뷰 감성 말뭉치) 원본에서 실습용 부분 집합을 만든다.

원본: https://github.com/e9t/nsmc (CC0 1.0, 공개 도메인 기증). 학습 15만·시험 5만 건, 긍정 1·부정 0
실습용: 학습 30,000건·시험 10,000건(긍정·부정 같은 수), id 열 제외, 빈 리뷰 제외
    python data/common/nsmc/make_subset.py <ratings_train.txt> <ratings_test.txt>
"""
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or not parts[1].strip():
                continue
            rows.append((parts[1].strip(), int(parts[2])))
    return rows


def balanced(rows, n, seed):
    rnd = random.Random(seed)
    pos = [r for r in rows if r[1] == 1]
    neg = [r for r in rows if r[1] == 0]
    rnd.shuffle(pos)
    rnd.shuffle(neg)
    out = pos[: n // 2] + neg[: n // 2]
    rnd.shuffle(out)
    return out


def save(rows, name):
    with open(HERE / name, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["document", "label"])
        w.writerows(rows)
    print(name, len(rows))


if __name__ == "__main__":
    save(balanced(load(sys.argv[1]), 30000, 1), "train.tsv")
    save(balanced(load(sys.argv[2]), 10000, 2), "test.tsv")
