"""[따라 치기 T3] 같은 티켓을 분류기와 회귀기로 판정 (1회차 실습 2 ⑥, 11분)

실행: python typing/s01/t3_ticket_clf.py      (빈칸 ____을 채운 뒤)
확인: ① category(7개 중 하나) = 분류 → 정확도
      ② priority(P1~P4, 순서 있음)를 분류로도, 회귀(숫자 1~4)로도 풀어 정확도·MAE 비교
      ③ 검증(학습과 같은 분포)과 시드 10건(5회차 LLM이 풀 문항) 점수 차이
전체 판(혼동행렬·TF-IDF 비교·결과 저장): python src/ticket_classifier.py --features both
"""
import json
import random

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error

rows = [json.loads(x) for x in open("data/G/train_seed.jsonl", encoding="utf-8")]
random.Random(42).shuffle(rows)
train, valid = rows[:150], rows[150:]
seed10 = json.load(open("data/G/test_seed.json", encoding="utf-8"))

emb = SentenceTransformer("nlpai-lab/KURE-v1", device="cuda" if torch.cuda.is_available() else "cpu")
X = lambda d: emb.encode([r["input"] for r in d], normalize_embeddings=True)    # 문장 → 1024차원 벡터
Xt, Xv, Xs = X(train), X(valid), X(seed10)
y = lambda d, f: [r["gold"][f] for r in d]

# 1) 분류: category
clf = ____(max_iter=3000, C=10).fit(Xt, y(train, "category"))
print("category 검증 정확도:", accuracy_score(y(valid, "category"), clf.predict(Xv)))
print("category 시드10 정확도:", accuracy_score(y(seed10, "category"), clf.predict(Xs)))

# 2) priority: 분류(4클래스) vs 회귀(숫자 1개)
P = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
pt, pv = [P[p] for p in y(train, "priority")], [P[p] for p in y(valid, "priority")]
c = LogisticRegression(max_iter=3000, C=10).fit(Xt, pt)
r = ____(alpha=1.0).fit(Xt, pt)
raw = r.predict(Xv)                                     # 1.37, 2.81 … 연속값
rounded = np.clip(np.____(raw), 1, 4)
print("priority 분류 정확도:", accuracy_score(pv, c.predict(Xv)))
print("priority 회귀 정확도(반올림):", accuracy_score(pv, rounded), " MAE:", round(____(pv, raw), 3))
