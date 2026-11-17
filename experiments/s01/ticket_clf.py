"""1회차 해보기 E 보조: 같은 티켓을 분류기와 회귀기로 판정. 규제 강도와 학습 건수를 바꿔 가며 봄

  python experiments/s01/ticket_clf.py                                # C 10, 학습 150건
  python experiments/s01/ticket_clf.py --C 0.01 0.1 1 10 100          # 규제 강도(작을수록 단순한 모델)
  python experiments/s01/ticket_clf.py --train-size 20 50 100 150     # 학습 건수
  전체 판(혼동행렬·TF-IDF 비교·결과 저장) : python src/ticket_classifier.py --features both

볼 것
  ① category(7개 중 하나)는 분류 → 정확도
  ② priority(P1~P4)는 순서가 있어 분류로도 회귀로도 풀 수 있음. 회귀는 MAE(평균 몇 단계 틀렸나)도 봄
  ③ 검증(학습과 같은 분포)과 시드 10건(5회차 LLM이 푸는 문항)의 점수 차이
"""
from __future__ import annotations

from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()          # 가중치 로딩 진행 막대 끄기

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
P = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C", type=float, nargs="+", default=[10.0])
    ap.add_argument("--train-size", type=int, nargs="+", default=[150])
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(ROOT / "data" / "G" / "train_seed.jsonl", encoding="utf-8")]
    random.Random(42).shuffle(rows)
    pool, valid = rows[:150], rows[150:]
    seed10 = json.load(open(ROOT / "data" / "G" / "test_seed.json", encoding="utf-8"))
    emb = SentenceTransformer("nlpai-lab/KURE-v1", device="cuda" if torch.cuda.is_available() else "cpu")
    enc = {id(d): emb.encode([r["input"] for r in d], normalize_embeddings=True) for d in (pool, valid, seed10)}
    y = lambda d, f: [r["gold"][f] for r in d]

    print("| 학습 건수 | C | category 검증 | category 시드10 | priority 분류 정확도 | priority 회귀 정확도(반올림) | priority 회귀 MAE |")
    print("|---|---|---|---|---|---|---|")
    for n in a.train_size:
        tr = pool[:n]
        Xt = enc[id(pool)][:n]
        for C in a.C:
            if len(set(y(tr, "category"))) < 2:
                continue
            clf = LogisticRegression(max_iter=3000, C=C).fit(Xt, y(tr, "category"))
            pt, pv = [P[p] for p in y(tr, "priority")], [P[p] for p in y(valid, "priority")]
            c = LogisticRegression(max_iter=3000, C=C).fit(Xt, pt)
            r = Ridge(alpha=1.0 / C).fit(Xt, pt)
            raw = r.predict(enc[id(valid)])
            print(f"| {n} | {C:g} | {accuracy_score(y(valid, 'category'), clf.predict(enc[id(valid)])):.3f} | "
                  f"{accuracy_score(y(seed10, 'category'), clf.predict(enc[id(seed10)])):.2f} | "
                  f"{accuracy_score(pv, c.predict(enc[id(valid)])):.3f} | {accuracy_score(pv, np.clip(np.round(raw), 1, 4)):.3f} | "
                  f"{mean_absolute_error(pv, raw):.3f} |")


if __name__ == "__main__":
    main()
