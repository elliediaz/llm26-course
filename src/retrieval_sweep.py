"""6회차 검색 손잡이 실험: LLM 없이 검색만 바꿔 가며 Recall@k·MRR을 잼(개인 CPU 서버에서도 돌아감)

  python src/retrieval_sweep.py size 200 500 1000 2000                 # 고정 길이 조각 크기(겹침 100)
  python src/retrieval_sweep.py overlap 0 50 100 250 --size 500        # 겹침 크기
  python src/retrieval_sweep.py strategy fixed sentence                # 분할 방식
  python src/retrieval_sweep.py embed nlpai-lab/KURE-v1 BAAI/bge-m3 keyword   # 임베딩 모델과 키워드(BM25)
  python src/retrieval_sweep.py hybrid 0 0.3 0.5 0.7 1                 # 키워드 비중(0 벡터만 ~ 1 키워드만)
  python src/retrieval_sweep.py size 300 1000 --dataset data/G/probes.json   # 다른 질문 묶음(gold_doc_ids가 있는 문항만)

볼 것 : Recall@1·3·5(정답 문서가 상위 k 안에 든 비율), MRR(첫 정답 순위의 역수 평균), 조각 수, 인덱스 시간, 유형별 Recall@3
결과 : results/retrieval_<손잡이>.json, results/fig/retrieval_<손잡이>.png
메모 : 인덱스를 파일로 만들지 않고 메모리에서만 비교함. 고른 설정은 config.py에 적고 index_builder.py로 실제 인덱스를 만듦
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np

import config
import retrieval
import utils

KNOBS = {"size": "고정 길이 조각 크기(글자)", "overlap": "조각 겹침(글자)", "strategy": "분할 방식",
         "embed": "임베딩 모델", "hybrid": "키워드(BM25) 비중"}


def tokenize(s: str) -> list[str]:
    words = re.findall(r"[0-9A-Za-z.]+|[가-힣]+", s.lower())
    grams = [w[i:i + 2] for w in words if re.match(r"[가-힣]", w) for i in range(len(w) - 1)]
    return words + grams


class MemIndex:
    """메모리 인덱스. 벡터 점수와 BM25 점수를 비중 alpha로 섞을 수 있음(0이면 벡터만, 1이면 키워드만)"""

    def __init__(self, chunks, embedder=None, alpha: float = 0.0):
        self.chunks, self.alpha = chunks, alpha
        t0 = time.perf_counter()
        self.emb = None
        if embedder is not None and alpha < 1:
            self.embedder = embedder
            self.emb = np.asarray(embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                                                  batch_size=32, show_progress_bar=False), dtype="float32")
        self.bm25 = None
        if alpha > 0:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.build_sec = round(time.perf_counter() - t0, 2)

    def search(self, q: str, k: int = 20) -> list[dict]:
        score = np.zeros(len(self.chunks), dtype="float32")
        if self.emb is not None:
            v = self.emb @ np.asarray(self.embedder.encode([q], normalize_embeddings=True), dtype="float32")[0]
            score += (1 - self.alpha) * v
        if self.bm25 is not None:
            b = np.asarray(self.bm25.get_scores(tokenize(q)), dtype="float32")
            if b.max() > 0:
                b = b / b.max()
            score += self.alpha * b
        top = np.argsort(-score)[:k]
        return [{**self.chunks[i], "score": float(score[i]), "rank": r + 1} for r, i in enumerate(top)]


def evaluate(index: MemIndex, items) -> dict:
    class Wrap:
        strategy = "mem"

        def search(self, q, k=None):
            return index.search(q, k or 20)
    return retrieval.eval_retrieval(Wrap(), items)


def chunks_for(docs, strategy="fixed", size=500, overlap=100, max_chars=500):
    out = []
    for d in docs:
        pieces = retrieval.chunk_fixed(d["text"], size, overlap) if strategy == "fixed" else retrieval.chunk_sentence(d["text"], max_chars)
        out += [{"chunk_id": f"{d['doc_id']}#{i}", "doc_id": d["doc_id"], "title": d["title"], "text": t} for i, t in enumerate(pieces)]
    return out


def plot(rows, knob, path):
    import matplotlib.pyplot as plt
    utils.korean_font()
    xs = [str(r["값"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for key, c in (("Recall@1", "#c0622b"), ("Recall@3", "#8aa1c1"), ("Recall@5", "#9bb08f"), ("MRR", "#555555")):
        ax.plot(range(len(xs)), [r[key] for r in rows], "o-", color=c, label=key)
    ax.set_xticks(range(len(xs)), xs, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{KNOBS[knob]}에 따른 검색 품질", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knob", choices=list(KNOBS))
    ap.add_argument("values", nargs="+")
    ap.add_argument("--size", type=int, default=config.CHUNK["fixed"]["size"])
    ap.add_argument("--overlap", type=int, default=config.CHUNK["fixed"]["overlap"])
    ap.add_argument("--strategy", default="fixed", choices=["fixed", "sentence"])
    ap.add_argument("--embed", default=config.EMBED_MODEL)
    ap.add_argument("--dataset", help="기본: data/<도메인>/test_dataset.json(gold_doc_ids가 있는 문항만 셈)")
    ap.add_argument("--with-updates", action="store_true", help="corpus_updates/의 새 문서도 넣음")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    docs = retrieval.load_corpus(with_updates=a.with_updates)
    items = utils.read_json(a.dataset or config.data_dir() / "test_dataset.json")
    n_gold = sum(1 for it in items if it.get("gold_doc_ids"))
    print(f"[검색 손잡이] {KNOBS[a.knob]} : {' · '.join(a.values)} | 도메인 {config.DOMAIN} · 문서 {len(docs)}건 · 근거 문서가 있는 문항 {n_gold}건")
    import model_utils
    emb_cache = {}

    def embedder(name):
        if name == "keyword":
            return None
        if name not in emb_cache:
            emb_cache[name] = model_utils.load_embedder(name)
        return emb_cache[name]

    rows, detail = [], {}
    for v in a.values:
        size, overlap, strategy, emb_name, alpha = a.size, a.overlap, a.strategy, a.embed, 0.0
        if a.knob == "size":
            size, strategy = int(v), "fixed"
        elif a.knob == "overlap":
            overlap, strategy = int(v), "fixed"
        elif a.knob == "strategy":
            strategy = v
        elif a.knob == "embed":
            emb_name = v
            alpha = 1.0 if v == "keyword" else 0.0
        elif a.knob == "hybrid":
            alpha = float(v)
        chunks = chunks_for(docs, strategy, size, overlap, config.CHUNK["sentence"]["max_chars"])
        idx = MemIndex(chunks, embedder(emb_name), alpha)
        r = evaluate(idx, items)
        lens = [len(c["text"]) for c in chunks]
        row = {"값": v.split("/")[-1], "조각 수": len(chunks), "조각 길이(중앙)": sorted(lens)[len(lens) // 2],
               "Recall@1": r.get("recall@1"), "Recall@3": r.get("recall@3"), "Recall@5": r.get("recall@5"), "MRR": r.get("mrr"),
               "인덱스(초)": idx.build_sec, **{f"R@3 {t}": x for t, x in (r.get("recall@3_by_type") or {}).items()}}
        rows.append(row)
        detail[v] = r
        print(f"  {row['값']:>12s} | 조각 {len(chunks):4d}개 | R@1 {row['Recall@1']} · R@3 {row['Recall@3']} · MRR {row['MRR']} | {idx.build_sec}s")
    print()
    utils.print_table(rows)
    name = f"retrieval_{a.knob}" + (f"_{a.tag}" if a.tag else "")
    utils.write_json(utils.result_path(f"{name}.json"), {"knob": a.knob, "values": a.values, "measured_at": utils.now(),
                                                         "domain": config.DOMAIN, "n": n_gold, "table": rows, "detail": detail})
    fig = utils.fig_path(f"{name}.png")
    plot(rows, a.knob, fig)
    best = max(rows, key=lambda r: (r["Recall@3"] or 0, r["MRR"] or 0))
    print(f"\nRecall@3 기준으로는 [{best['값']}] 설정이 가장 높음. 조각 수(인덱스 크기)와 프롬프트 길이도 같이 볼 것")
    print(f"저장: results/{name}.json · 그림: {fig}")


if __name__ == "__main__":
    main()
