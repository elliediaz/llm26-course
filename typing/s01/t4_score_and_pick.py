"""[따라 치기 T4] 검색 = 점수 매기기(연속값) + 고르기(분류) (1회차 해보기 D, 10분)

실행: python typing/s01/t4_score_and_pick.py      (빈칸 ____을 채운 뒤)
확인: ① 점수 25개는 연속값(코사인 유사도)
      ② 그중 가장 높은 것을 고르는 순간 'FAQ 25개 중 하나'를 고르는 분류가 됨
      ③ 기준값(THRESHOLD)을 두면 '해당 없음'이라는 26번째 클래스가 생김
"""
import json

import torch
from sentence_transformers import SentenceTransformer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
faq = json.load(open("data/G/faq.json", encoding="utf-8"))
emb = SentenceTransformer("nlpai-lab/KURE-v1", device=DEV)

F = emb.encode([f["question"] for f in faq], normalize_embeddings=True)    # (25, 1024)
for query in ["돈 냈는데 아이템이 안 와요", "오늘 점심 뭐 먹지"]:
    q = emb.encode([query], normalize_embeddings=True)[0]                  # (1024,)
    scores = ____                     # 정규화했으므로 내적 = 코사인 유사도(연속값 25개)
    best = scores.____()             # 가장 높은 점수 = 분류 결과
    print(f"\n[{query}] 점수 상위 3:", [(faq[i]["id"], round(float(scores[i]), 3)) for i in scores.argsort()[::-1][:3]])
    THRESHOLD = 0.5
    if scores[best] ____ THRESHOLD:
        print("→ 해당 없음")
    else:
        print("→", faq[best]["id"], faq[best]["question"])
