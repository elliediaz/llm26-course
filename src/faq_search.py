"""1회차 실습 1: FAQ 의미 검색기. Encoder로 문장을 벡터로 바꾸고 코사인 유사도로 가장 가까운 FAQ를 찾는다.

  python src/faq_search.py                          # 시험 질의 10건 정확도(임베딩)
  python src/faq_search.py --method tfidf           # 비교: 글자 단위 TF-IDF(키워드 겹침 기반)
  python src/faq_search.py --q "돈 냈는데 아이템 안 와요"   # 한 문장 검색
  python src/faq_search.py --queries my_queries.json        # 직접 만든 질문 10건으로 측정(완료 기준: 8건 이상 1순위)
  python src/faq_search.py --threshold 0.45         # 과제 카드 B: 기준 미달이면 '해당 없음'

my_queries.json 형식: [{"query": "...", "gold_faq_id": "F07"}, {"query": "무관한 질문", "gold_faq_id": null}, ...]
결과: results/faq_eval.json (TF-IDF는 results/faq_eval_tfidf.json)
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter

import numpy as np

import config
import utils


# ---------------------------------------------------------------- 비교 기준: TF-IDF(신경망 이전 방식)
def char_ngrams(s: str, n: int = 2) -> list[str]:
    s = re.sub(r"\s+", " ", s.strip())
    return [s[i:i + n] for i in range(len(s) - n + 1)]


class TfidfIndex:
    """글자 2-gram TF-IDF. 문장을 '단어 주머니' 고정 길이 벡터로 만든다(어순·의미는 반영 못 함)."""

    def __init__(self, texts: list[str]):
        self.docs = [Counter(char_ngrams(t)) for t in texts]
        df = Counter(g for d in self.docs for g in d)
        n = len(texts)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        self.vocab = {g: i for i, g in enumerate(self.idf)}
        self.mat = np.stack([self.vec_counts(d) for d in self.docs])

    def vec_counts(self, cnt: Counter) -> np.ndarray:
        v = np.zeros(len(self.vocab), dtype="float32")
        for g, c in cnt.items():
            if g in self.vocab:
                v[self.vocab[g]] = c * self.idf[g]
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vec_counts(Counter(char_ngrams(t))) for t in texts])


# ---------------------------------------------------------------- 검색기
class FaqSearcher:
    def __init__(self, faq: list[dict], method: str = "embed", model_name: str | None = None, embedder=None):
        self.faq, self.method = faq, method
        questions = [f["question"] for f in faq]
        if method == "tfidf":
            self.enc = TfidfIndex(questions)
            self.mat = self.enc.mat
        else:
            import model_utils
            self.enc = embedder or model_utils.load_embedder(model_name)
            # [실습] 질문만 넣을지, 질문+답을 함께 넣을지 바꿔 보며 정확도 비교
            self.mat = np.asarray(self.enc.encode(questions, normalize_embeddings=True))

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.method == "tfidf":
            return self.enc.encode(texts)
        return np.asarray(self.enc.encode(texts, normalize_embeddings=True))

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        q = self.encode([query])[0]
        scores = self.mat @ q                 # 정규화했으므로 내적 = 코사인 유사도
        top = np.argsort(-scores)[:k]
        return [(self.faq[i], float(scores[i])) for i in top]


def evaluate(searcher: FaqSearcher, queries: list[dict], threshold: float | None) -> dict:
    rows, hit = [], 0
    for q in queries:
        res = searcher.search(q["query"])
        best, score = res[0]
        pred = best["id"] if threshold is None or score >= threshold else None
        gold = q.get("gold_faq_id")
        ok = pred == gold
        hit += ok
        rows.append({"질의": q["query"], "유형": q.get("variant", ""), "정답": gold, "1순위": best["id"],
                     "유사도": round(score, 3), "판정": "O" if ok else "X"})
    return {"accuracy": hit / len(queries), "hits": hit, "n": len(queries), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["embed", "tfidf"], default="embed")
    ap.add_argument("--model", default=None, help="임베딩 모델(기본 config.EMBED_MODEL)")
    ap.add_argument("--queries", help="직접 만든 질의 파일(기본: data/<DOMAIN>/faq_queries.json)")
    ap.add_argument("--q", help="한 문장만 검색")
    ap.add_argument("--threshold", type=float, help="이 값보다 낮으면 '해당 없음'(None)으로 처리")
    a = ap.parse_args()

    d = config.data_dir()
    faq = utils.read_json(d / "faq.json")
    with utils.Timer() as t:
        s = FaqSearcher(faq, a.method, a.model)
    print(f"[{a.method}] FAQ {len(faq)}건 색인 {t.sec:.1f}s")

    if a.q:
        for f, sc in s.search(a.q):
            print(f"  {sc:.3f}  {f['id']} {f['question']}\n         → {f['answer']}")
        return

    queries = utils.read_json(a.queries or d / "faq_queries.json")
    r = evaluate(s, queries, a.threshold)
    utils.print_table(r["rows"])
    print(f"\n1순위 적중 {r['hits']}/{r['n']} = {r['accuracy']:.0%}  (완료 기준: 10건 중 8건 이상)"
          + (f"  · 임계값 {a.threshold}" if a.threshold is not None else ""))
    name = "faq_eval.json" if a.method == "embed" else "faq_eval_tfidf.json"
    utils.write_json(utils.result_path(name), {
        "method": a.method, "model": a.model or (config.EMBED_MODEL if a.method == "embed" else "char-2gram TF-IDF"),
        "threshold": a.threshold, "measured_at": utils.now(), **r})


if __name__ == "__main__":
    main()
