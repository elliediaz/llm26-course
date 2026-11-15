"""6회차 도전과제 ②: 재순위화(Reranker): 1차 검색 후보를 질문과 함께 다시 읽어 순위를 고친다.

  python src/rerank.py                       # 1차 검색만 vs +재순위화: Recall@k, 질의당 추가 지연

Reranker는 질문과 문서를 한 번에 입력하는 Cross-Encoder 라 정확하지만 느리다.
늘어난 지연만큼 정확도가 올랐는지 따져 보는 것이 이 과제의 핵심(실무에서 늘 하는 판단).
"""
from __future__ import annotations

import time

import config
import retrieval
import utils


class Reranked:
    def __init__(self, retriever: retrieval.Retriever, model_name: str | None = None, depth: int = 20):
        from sentence_transformers import CrossEncoder
        import model_utils
        self.r, self.depth = retriever, depth
        self.ce = CrossEncoder(model_name or config.RERANK_MODEL, device=model_utils.device())
        self.strategy = f"{retriever.strategy}+rerank"
        self.extra_sec: list[float] = []

    def search(self, q: str, k: int = 5):
        cands = self.r.search(q, self.depth)
        t0 = time.perf_counter()
        scores = self.ce.predict([(q, c["text"]) for c in cands])
        self.extra_sec.append(time.perf_counter() - t0)
        order = sorted(range(len(cands)), key=lambda i: -scores[i])[:k]
        return [{**cands[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(order)]


def main() -> None:
    base = retrieval.Retriever()
    items = utils.read_json(config.data_dir() / "test_dataset.json")
    rr = Reranked(base)
    rows = []
    for name, ret in (("1차 검색", base), ("+재순위화", rr)):
        r = retrieval.eval_retrieval(ret, items)
        rows.append({"방식": name, "Recall@1": r["recall@1"], "Recall@3": r["recall@3"], "MRR": r["mrr"]})
    lat = utils.latency_stats(rr.extra_sec)
    rows[1]["추가 지연 p50(s)"] = lat["p50"]
    utils.print_table(rows)
    utils.write_json(utils.result_path("retrieval_rerank.json"), rows)


if __name__ == "__main__":
    main()
