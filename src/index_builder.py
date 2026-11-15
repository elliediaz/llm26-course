"""6회차 실습 1: 코퍼스 적재 → 두 가지 분할 전략으로 Chunk 생성 → 인덱스 구축 → 검색 품질 비교.

  python src/index_builder.py --stats              # ③ 고정 길이 vs 문장 경계: 조각 수·길이 분포
  python src/index_builder.py                      # ④ 두 전략 모두 인덱스 구축 + Recall@k·MRR 비교
  python src/index_builder.py --strategy sentence  # 한 전략만
  python src/index_builder.py --backend faiss      # 벡터DB 교체(chroma 문제 시)

비교 후 고른 전략을 src/config.py의 CHUNK_STRATEGY에 적는다(이후 rag.py·benchmark.py가 그 인덱스를 씀).
"검색 먼저 재라": 정답 문서를 못 가져오면 프롬프트를 아무리 고쳐도 소용없다.
결과: index/<도메인>_<전략>/, results/chunk_stats.json, results/retrieval_compare.json, results/fig/chunk_len.png
"""
from __future__ import annotations

import argparse

import config
import model_utils
import retrieval
import utils


def plot_lengths(by_strategy: dict[str, list[int]]) -> None:
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, lens in by_strategy.items():
        ax.hist(lens, bins=20, alpha=0.6, label=f"{name} ({len(lens)}개)")
    ax.set(xlabel="조각 길이(글자)", ylabel="개수", title="분할 전략별 조각 길이 분포")
    ax.legend()
    fig.tight_layout()
    fig.savefig(utils.fig_path("chunk_len.png"), dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=["fixed", "sentence", "both"], default="both")
    ap.add_argument("--backend", choices=["chroma", "faiss", "numpy"], default=None)
    ap.add_argument("--stats", action="store_true", help="조각 통계만(인덱스는 만들지 않음)")
    a = ap.parse_args()
    docs = retrieval.load_corpus()
    total = sum(len(d["text"]) for d in docs)
    print(f"[코퍼스] 도메인 {config.DOMAIN} · 문서 {len(docs)}건 · {total:,}자")
    strategies = ["fixed", "sentence"] if a.strategy == "both" else [a.strategy]
    chunks = {s: retrieval.make_chunks(docs, s) for s in strategies}
    stats = {s: retrieval.chunk_stats(c) for s, c in chunks.items()}
    utils.print_table([{"전략": s, **v, "설정": config.CHUNK[s]} for s, v in stats.items()])
    utils.write_json(utils.result_path("chunk_stats.json"), stats)
    plot_lengths({s: [len(x["text"]) for x in c] for s, c in chunks.items()})
    ex = chunks[strategies[0]][3]
    print(f"\n조각 예({strategies[0]}): {ex['chunk_id']}\n{ex['text'][:300]}")
    if a.stats:
        return

    emb = model_utils.load_embedder()
    items = utils.read_json(config.data_dir() / "test_dataset.json")
    rows = []
    for s in strategies:
        with utils.section(f"인덱스 구축: {s}"):
            path = retrieval.build_index(chunks[s], emb, s, a.backend)
            print(f"  저장: {path}")
        r = retrieval.eval_retrieval(retrieval.Retriever(s, emb), items)
        rows.append({"전략": s, "조각 수": stats[s]["n_chunks"], "평가 문항": r["n"],
                     "Recall@1": r.get("recall@1"), "Recall@3": r.get("recall@3"), "Recall@5": r.get("recall@5"),
                     "MRR": r.get("mrr"), "유형별 R@3": r.get("recall@3_by_type")})
        utils.write_json(utils.result_path(f"retrieval_{s}.json"), r)
        if r.get("misses"):
            print("  상위 3위 안에 정답 문서가 없던 문항:")
            for m in r["misses"][:5]:
                print(f"    {m['id']}({m['type']}) 정답 순위 {m['first_hit']} · 검색 상위 {m['top3']}")
    print()
    utils.print_table(rows)
    utils.write_json(utils.result_path("retrieval_compare.json"), {"rows": rows, "measured_at": utils.now(),
                                                                   "embed_model": config.EMBED_MODEL})
    best = max(rows, key=lambda r: (r["Recall@3"] or 0, r["MRR"] or 0))
    print(f"\n→ 검색 품질이 높은 전략: {best['전략']}. src/config.py의 CHUNK_STRATEGY를 확인할 것(현재 {config.CHUNK_STRATEGY})")


if __name__ == "__main__":
    main()
