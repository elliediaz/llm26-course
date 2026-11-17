"""6회차 실습 2: 검색 기반 생성: 질문과 가까운 문서 조각을 찾아 프롬프트에 넣고, 답과 함께 근거 문서를 보여 준다.

  python src/rag.py --q "서리칼날 공격력 왜 줄었어요? 버그 아닌가요"     # 근거 문서와 함께 답변
  python src/rag.py --q "..." --no-rag                                   # 같은 질문, 검색 없이(비교)
  python src/rag.py --q "..." --show-prompt                              # 모델에 실제로 들어간 프롬프트 확인
  python src/rag.py --q "..." --search-only                              # LLM 없이 검색 결과만(개인 환경에서도 됨)
  python src/rag.py --eval-retrieval                                     # 현재 인덱스의 Recall@k·MRR
  python src/benchmark.py --mode rag                                     # 성능 기록표 두 번째 행

오답 분석(실습 ⑧): results/rag.json의 문항마다 retrieval_hit가 False 면 '검색 실패', True 인데 틀렸으면 '생성 실패'.
python src/rag.py --diagnose results/rag.json으로 분류표를 만든다 → 7회차 입력("검색으로 해결되지 않는 항목")
"""
from __future__ import annotations

import argparse
import json

import config
import model_utils
import prompt_utils
import retrieval
import scorer
import utils


def answer(q: str, model, tok, retriever, schema, use_rag=True, show_prompt=False) -> dict:
    hits = retriever.search(q) if use_rag else []
    ctx = retrieval.format_context(hits, tok) if hits else None
    msgs = prompt_utils.build_messages(q, schema, "instruction", ctx)
    if show_prompt:
        print("\n----- 프롬프트 -----")
        print(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **config.CHAT_TEMPLATE_KWARGS))
        print("--------------------")
    g = model_utils.generate(model, tok, msgs, config.GEN_EVAL)
    return {"answer": g["text"], "sources": [{"doc_id": h["doc_id"], "title": h["title"], "score": h["score"]} for h in hits],
            "latency": g["latency"], "n_in": g["n_in"]}


def diagnose(path: str) -> None:
    res = utils.read_json(path)
    rows = []
    for it in res["items"]:
        if it["exact"]:
            continue
        cause = "검색 실패" if it.get("retrieval_hit") is False else ("생성 실패" if it.get("retrieval_hit") else "검색 불필요(R·F 유형)")
        wrong = [f for f, s in it["fields"].items() if s < 1]
        rows.append({"id": it["id"], "유형": it["type"], "원인": cause, "틀린 필드": ",".join(wrong), "점수": it["score"]})
    utils.print_table(rows)
    from collections import Counter
    print("\n원인별:", dict(Counter(r["원인"] for r in rows)))
    md = ["# 검색으로 해결되지 않는 항목(6회차 → 7회차 입력)", "",
          "| id | 유형 | 원인 | 틀린 필드 |", "|---|---|---|---|"] + \
         [f"| {r['id']} | {r['유형']} | {r['원인']} | {r['틀린 필드']} |" for r in rows if r["원인"] != "검색 실패"]
    out = utils.result_path("rag_limits.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"저장: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--eval-retrieval", action="store_true")
    ap.add_argument("--diagnose", metavar="RESULT_JSON")
    ap.add_argument("--search-only", action="store_true", help="LLM을 올리지 않고 검색 결과만 봄")
    ap.add_argument("--strategy", choices=["fixed", "sentence"], help="기본: config.CHUNK_STRATEGY")
    ap.add_argument("--k", type=int, help="가져올 조각 수(기본: config.RAG_TOP_K)")
    a = ap.parse_args()
    if a.diagnose:
        diagnose(a.diagnose)
        return
    retriever = retrieval.Retriever(a.strategy)
    if a.k:
        config.RAG_TOP_K = a.k
    if a.eval_retrieval:
        r = retrieval.eval_retrieval(retriever, utils.read_json(config.data_dir() / "test_dataset.json"))
        print(json.dumps({k: v for k, v in r.items() if k != "misses"}, ensure_ascii=False, indent=2))
        return
    if not a.q:
        ap.error("--q 질문이 필요합니다")
    if a.search_only:
        for h in retriever.search(a.q):
            print(f"  {h['rank']}. {h['doc_id']} {h['title']} (유사도 {h['score']:.4f})")
            print("     " + h["text"][:160].replace("\n", " ") + ("…" if len(h["text"]) > 160 else ""))
        return
    model, tok = model_utils.load_llm()
    res = answer(a.q, model, tok, retriever, scorer.load_schema(), not a.no_rag, a.show_prompt)
    print(f"\n[답변] {res['answer']}")
    if res["sources"]:
        print("[근거 문서]")
        for s in res["sources"]:
            print(f"  - {s['doc_id']} {s['title']} (유사도 {s['score']})")
    print(f"[입력 토큰] {res['n_in']} · [지연] {res['latency']}s")


if __name__ == "__main__":
    main()
