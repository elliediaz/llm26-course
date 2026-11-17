"""6회차 도전과제 ①: 키워드 검색(BM25) + 벡터 검색을 함께 쓰면 검색 품질이 오르는가.

  python src/hybrid_search.py                  # 벡터만 / BM25만 / 결합(RRF) Recall@k 비교
  python src/hybrid_search.py --strategy fixed # 다른 분할 전략의 인덱스로
  결과: results/retrieval_rrf.json(retrieval_sweep.py hybrid의 결과 파일과 이름을 나눔)

고유명사(서리칼날·녹스), 숫자(v1.4, 72회), 약어가 들어간 질문에서 키워드 검색이 강하다.
결합 방식: RRF(Reciprocal Rank Fusion): 두 순위표에서 1/(60+순위)를 더해 다시 줄 세운다(점수 척도가 달라도 됨).
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

import config
import retrieval
import utils


def tokenize(s: str) -> list[str]:
    """한국어 형태소 분석기 없이 쓰는 간단한 방법: 어절 + 글자 2-gram."""
    words = re.findall(r"[0-9A-Za-z.]+|[가-힣]+", s.lower())
    grams = [w[i:i + 2] for w in words if re.match(r"[가-힣]", w) for i in range(len(w) - 1)]
    return words + grams


class Hybrid:
    def __init__(self, retriever: retrieval.Retriever):
        self.vec = retriever
        self.chunks = retriever.chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def bm25_search(self, q: str, k: int):
        scores = self.bm25.get_scores(tokenize(q))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(top)]

    def search(self, q: str, k: int = 5, mode: str = "hybrid", depth: int = 30):
        if mode == "vector":
            return self.vec.search(q, k)
        if mode == "bm25":
            return self.bm25_search(q, k)
        fused: dict[str, float] = {}
        for hits in (self.vec.search(q, depth), self.bm25_search(q, depth)):
            for h in hits:
                fused[h["chunk_id"]] = fused.get(h["chunk_id"], 0.0) + 1.0 / (60 + h["rank"])
        by_id = {c["chunk_id"]: c for c in self.chunks}
        top = sorted(fused, key=lambda c: -fused[c])[:k]
        return [{**by_id[c], "score": round(fused[c], 4), "rank": r + 1} for r, c in enumerate(top)]


class _Adapter:
    """eval_retrieval이 쓰는 search(query, k) 형태로 맞춘다."""

    def __init__(self, h: Hybrid, mode: str):
        self.h, self.mode, self.strategy = h, mode, f"{h.vec.strategy}/{mode}"

    def search(self, q, k=5):
        return self.h.search(q, k, self.mode)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["fixed", "sentence"], help="기본: config.CHUNK_STRATEGY")
    a = ap.parse_args()
    h = Hybrid(retrieval.Retriever(a.strategy))
    items = utils.read_json(config.data_dir() / "test_dataset.json")
    rows = []
    for mode in ("vector", "bm25", "hybrid"):
        r = retrieval.eval_retrieval(_Adapter(h, mode), items)
        rows.append({"방식": mode, "Recall@1": r["recall@1"], "Recall@3": r["recall@3"], "Recall@5": r["recall@5"], "MRR": r["mrr"]})
    utils.print_table(rows)
    utils.write_json(utils.result_path("retrieval_rrf.json"), rows)
    print("저장: results/retrieval_rrf.json")


if __name__ == "__main__":
    main()
