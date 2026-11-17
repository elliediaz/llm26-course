"""5~7회차: 평가 하네스 실행기: 같은 Test Dataset·같은 채점 규칙·같은 생성 설정으로 구성만 바꿔 잰다.

  python src/benchmark.py                          # Baseline (5회차)
  python src/benchmark.py --runs 2                 # 같은 명령 2회 → 점수가 같아야 완료(재현성)
  python src/benchmark.py --strategy fewshot --name fmt_fewshot --no-scoreboard   # 형식 강제 방법 비교
  python src/benchmark.py --no-template --name no_template --no-scoreboard          # Chat Template 전후 비교
  python src/benchmark.py --prompt prompts/s05/주입방어.txt --name guard --no-scoreboard   # 팀 규칙을 덧붙여 비교
  설정 하나를 여러 값으로 바꿔 가며 비교하려면 sweep.py를 씀(모델을 한 번만 올림)
  python src/benchmark.py --mode rag                # 6회차: 검색 결과를 넣어 생성
  python src/benchmark.py --mode lora --adapter adapters/<팀>/<run>        # 7회차
  python src/benchmark.py --mode rag_lora --adapter adapters/<팀>/<run>    # 7회차

결과: results/<name>.json (문항별 출력·점수 포함) + results/scoreboard.md 한 행 추가
측정 원칙: 한 번에 하나만 바꾼다. 프롬프트·채점 규칙·GEN_EVAL을 바꾸면 Baseline부터 다시 잰다.
"""
from __future__ import annotations

import argparse
import json

import config
import model_utils
import prompt_utils
import scorer
import scoreboard
import utils

MODE_LABEL = {"base": "Baseline", "rag": "RAG", "lora": "LoRA", "rag_lora": "RAG+LoRA"}


def read_extra(path: str | None) -> str | None:
    """--prompt 파일(팀이 덧붙이는 규칙). 'base'나 빈 값이면 없음"""
    if not path or path == "base":
        return None
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        p = config.ROOT / path
    return p.read_text(encoding="utf-8")


def run_once(model, tok, items, schema, rules, a, retriever=None, quiet: bool = False) -> dict:
    """a에서 읽는 값: strategy, no_template, score_mode, (있으면) shots, extra, gen, top_k, context_tokens"""
    out_items, lat, rlog = [], [], []
    utils.vram_reset()
    gen = getattr(a, "gen", None) or config.GEN_EVAL
    for i, it in enumerate(items, 1):
        context, hits = None, []
        if retriever is not None:
            hits = retriever.search(it["input"], k=getattr(a, "top_k", None))
            import retrieval
            context = retrieval.format_context(hits, tok, max_tokens=getattr(a, "context_tokens", None))
        msgs = prompt_utils.build_messages(it["input"], schema, a.strategy, context,
                                           shots=getattr(a, "shots", None), extra=getattr(a, "extra", None))
        lp = prompt_utils.json_constraint(tok, model, schema) if a.strategy == "constrained" else None
        g = model_utils.generate(model, tok, msgs, gen, use_template=not a.no_template, logits_processor=lp)
        s = scorer.score_item(g["text"], it["gold"], rules, schema, a.score_mode)
        lat.append(g["latency"])
        rec = {"id": it["id"], "type": it.get("type"), "input": it["input"], "gold": it["gold"],
               "output": g["text"], **{k: s[k] for k in ("parsed", "json_ok", "fields", "score", "exact", "errors")},
               "latency": g["latency"], "n_in": g["n_in"], "n_out": g["n_out"]}
        if hits:
            ranking = [h["doc_id"] for h in hits]
            gold_docs = set(it.get("gold_doc_ids") or [])
            rec["retrieved"] = ranking
            rec["retrieval_hit"] = bool(gold_docs & set(ranking)) if gold_docs else None
            rlog.append(rec["retrieval_hit"])
        out_items.append(rec)
        mark = "O" if s["exact"] else ("△" if s["score"] > 0 else "X")
        if not quiet:
            print(f"  [{i:2d}/{len(items)}] {it['id']:10s} {mark} score={s['score']:.2f} json={'O' if s['json_ok'] else 'X'} {g['latency']:.2f}s")
    metrics = scorer.aggregate(out_items)
    metrics["latency"] = utils.latency_stats(lat)
    metrics["n_in_avg"] = round(sum(x["n_in"] for x in out_items) / len(out_items), 1)
    metrics["n_out_avg"] = round(sum(x["n_out"] for x in out_items) / len(out_items), 1)
    metrics["vram_peak_gb"] = utils.vram_peak_gb()
    hits_known = [x for x in rlog if x is not None]
    if hits_known:
        metrics["retrieval_hit_rate"] = round(sum(hits_known) / len(hits_known), 4)
    return {"metrics": metrics, "items": out_items}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(MODE_LABEL), default="base")
    ap.add_argument("--adapter", help="LoRA Adapter 폴더(lora·rag_lora 모드)")
    ap.add_argument("--strategy", choices=prompt_utils.STRATEGIES, default="instruction")
    ap.add_argument("--shots", type=int, help="예시 개수(기본: fewshot 2, 나머지 0)")
    ap.add_argument("--prompt", help="시스템 프롬프트에 덧붙일 규칙 파일(예: prompts/s05/주입방어.txt)")
    ap.add_argument("--no-template", action="store_true", help="Chat Template 없이 입력(비교 실험용)")
    ap.add_argument("--score-mode", choices=["loose", "strict"], default="loose")
    ap.add_argument("--dataset", help="기본: data/<DOMAIN>/test_dataset.json")
    ap.add_argument("--limit", type=int, help="앞에서 N건만(빠른 점검용: 기록표에는 남기지 않음)")
    ap.add_argument("--runs", type=int, default=1, help="반복 횟수(2 이상이면 재현성 검사)")
    ap.add_argument("--name", help="결과 파일 이름(기본: 모드 이름)")
    ap.add_argument("--note", default="", help="성능 기록표 비고")
    ap.add_argument("--no-scoreboard", action="store_true", help="성능 기록표에 남기지 않음(비교 실험용)")
    a = ap.parse_args()
    if a.mode in ("lora", "rag_lora") and not a.adapter:
        ap.error("lora·rag_lora 모드는 --adapter 경로가 필요합니다")

    a.extra = read_extra(a.prompt)
    utils.set_seed()
    ds_path = a.dataset or config.data_dir() / "test_dataset.json"
    items = utils.read_json(ds_path)
    if a.limit:
        items = items[: a.limit]
    schema, rules = scorer.load_schema(), scorer.load_rules()
    print(f"[하네스] 도메인 {config.DOMAIN} · {len(items)}문항 · 모드 {MODE_LABEL[a.mode]} · 방식 {a.strategy}"
          f"{' · 템플릿 없음' if a.no_template else ''}")

    model, tok = model_utils.load_llm(adapter=a.adapter if a.mode in ("lora", "rag_lora") else None)
    retriever = None
    if a.mode in ("rag", "rag_lora"):
        import retrieval
        retriever = retrieval.Retriever()

    runs = []
    for r in range(a.runs):
        print(f"\n▶ 실행 {r + 1}/{a.runs}")
        runs.append(run_once(model, tok, items, schema, rules, a, retriever))
    res = runs[-1]
    m = res["metrics"]
    if a.runs > 1:
        scores = [x["metrics"]["task_accuracy"] for x in runs]
        outs_same = all([i["output"] for i in x["items"]] == [i["output"] for i in runs[0]["items"]] for x in runs)
        m["reproducible"] = len(set(scores)) == 1 and outs_same
        m["run_scores"] = scores

    name = a.name or a.mode
    payload = {
        "name": name, "label": MODE_LABEL[a.mode], "measured_at": utils.now(), "dataset": str(ds_path),
        "config": {**config.summary(), "mode": a.mode, "adapter": a.adapter, "strategy": a.strategy,
                   "chat_template": not a.no_template, "score_mode": a.score_mode,
                   "shots": a.shots, "prompt": a.prompt},
        "metrics": m, "items": res["items"],
    }
    path = utils.write_json(utils.result_path(f"{name}.json"), payload)

    print("\n[결과]")
    lat = m["latency"]
    print(f"  Task Accuracy {m['task_accuracy']:.3f} · JSON 성공률 {m['json_success']:.3f} · Exact Match {m['exact_match']:.3f}")
    print(f"  유형별 {json.dumps(m['by_type'], ensure_ascii=False)}  필드별 {json.dumps(m['by_field'], ensure_ascii=False)}")
    print(f"  Latency p50 {lat['p50']}s · p95 {lat['p95']}s · VRAM 최대 {m['vram_peak_gb']}GB")
    if "retrieval_hit_rate" in m:
        print(f"  검색 적중(상위 {config.RAG_TOP_K}개에 정답 문서) {m['retrieval_hit_rate']:.3f}")
    if "reproducible" in m:
        print(f"  재현성: {'같음' if m['reproducible'] else '다름: 생성 설정·시드를 확인'} {m['run_scores']}")
    worst = sorted(res["items"], key=lambda x: x["score"])[:3]
    print("  낮은 점수 3건:", ", ".join(f"{w['id']}({w['score']:.2f})" for w in worst))
    print(f"  저장: {path}")
    if not a.no_scoreboard and not a.limit:
        scoreboard.add_row(payload, note=a.note)


if __name__ == "__main__":
    main()
