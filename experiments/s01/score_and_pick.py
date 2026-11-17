"""1회차 해보기 D 보조: 검색 = 점수 매기기(연속값) + 고르기(분류). 기준값을 바꾸면 '해당 없음' 판정이 어떻게 바뀌나

  python experiments/s01/score_and_pick.py                                  # 시험 질의 10건, 기준값 0.3~0.7
  python experiments/s01/score_and_pick.py --query "돈 냈는데 아이템이 안 와요" "오늘 점심 뭐 먹지"
  python experiments/s01/score_and_pick.py --thresholds 0.2 0.4 0.6 0.8
  python experiments/s01/score_and_pick.py --model BAAI/bge-m3               # 다른 임베딩 모델

볼 것
  ① FAQ 25개에 대한 점수 25개(코사인 유사도)는 연속값. 1등을 고르는 순간 25개 중 하나를 고르는 분류가 됨
  ② 기준값을 높이면 '해당 없음'이 늘어 무관 질문은 잘 거르지만 은어·오타 질문도 놓침. 낮추면 반대
  ③ 표의 '맞힘'(정답 FAQ를 1등으로 고름)과 '무관 거름'(정답이 없는 질의를 해당 없음으로 처리)이 서로 밀고 당김
"""
from __future__ import annotations

from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()          # 가중치 로딩 진행 막대 끄기

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nlpai-lab/KURE-v1")
    ap.add_argument("--query", nargs="*")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    a = ap.parse_args()
    faq = json.load(open(ROOT / "data" / "G" / "faq.json", encoding="utf-8"))
    emb = SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu")
    F = emb.encode([f["question"] for f in faq], normalize_embeddings=True)

    if a.query:
        for q in a.query:
            s = F @ emb.encode([q], normalize_embeddings=True)[0]
            order = np.argsort(-s)[:3]
            print(f"\n[{q}] 상위 3 : " + ", ".join(f"{faq[i]['id']} {s[i]:.3f}" for i in order))
            print("  → 기준값별 판정 : " + " · ".join(f"{t}: {'해당 없음' if s[order[0]] < t else faq[order[0]]['id']}" for t in a.thresholds))
        return

    qs = json.load(open(ROOT / "data" / "G" / "faq_queries.json", encoding="utf-8"))
    Q = emb.encode([q["query"] for q in qs], normalize_embeddings=True)
    S = Q @ F.T
    best, top = S.argmax(1), S.max(1)
    print(f"시험 질의 {len(qs)}건(표현 변형·오타·은어·무관 질문)\n")
    print("| 기준값 | 맞힘 | 무관 거름 | 놓침(해당 없음으로 잘못 처리) |\n|---|---|---|---|")
    for t in a.thresholds:
        hit = miss = rej_ok = n_none = 0
        for q, b, sc in zip(qs, best, top):
            gold = q.get("gold_faq_id")
            say = None if sc < t else faq[b]["id"]
            if gold is None:
                n_none += 1
                rej_ok += say is None
            else:
                hit += say == gold
                miss += say is None
        n_ans = len(qs) - n_none
        print(f"| {t} | {hit}/{n_ans} | {rej_ok}/{n_none} | {miss}/{n_ans} |")
    print("\n질의별 1등 점수 : " + ", ".join(f"{q['qid']} {sc:.2f}" for q, sc in zip(qs, top)))


if __name__ == "__main__":
    main()
