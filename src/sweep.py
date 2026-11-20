"""5~7회차 설정 비교 실험: 모델을 한 번만 올리고 설정 하나를 여러 값으로 바꿔 가며 같은 문항을 잼

  python src/sweep.py strategy plain instruction fewshot constrained --limit 10
  python src/sweep.py shots 0 1 2 4 8 --limit 10
  python src/sweep.py template on off --limit 10
  python src/sweep.py temperature 0 0.3 0.7 1.2 --repeat 3 --limit 10
  python src/sweep.py max_tokens 24 48 96 256 --limit 10
  python src/sweep.py prompt base prompts/s05/주입방어.txt prompts/s05/우리팀.txt --limit 10
  python src/sweep.py score_mode loose strict                  # 생성은 한 번, 채점만 두 가지
  python src/sweep.py mode base rag --limit 10                 # 6회차: 검색 결과를 넣기 전과 후
  python src/sweep.py top_k 1 2 4 8 --mode rag --limit 10      # 6회차
  python src/sweep.py context_tokens 300 800 1500 3000 --mode rag --limit 10   # 6회차
  python src/sweep.py adapter none adapters/<팀>/r4 adapters/<팀>/r16 --limit 10  # 7회차
  python src/sweep.py model skt/A.X-4.0-Light Qwen/Qwen3-8B --limit 10            # 모델 바꾸기(값마다 다시 올림)
  python src/sweep.py quant bf16 nf4 --limit 10                                   # 4비트로 줄였을 때 점수·속도·메모리

공통 옵션
  --dataset  다른 문항 파일(예: data/G/probes.json 대신 test_dataset.json이 기본)
  --limit    앞에서 N건만. 설정 비교 실험은 10건으로 방향을 보고, 고른 설정만 30건으로 다시 잼
  --repeat   같은 값을 여러 번(temperature처럼 달라지는 설정에서 평균·편차를 봄)
  --tag      결과 파일 이름 태그(같은 설정을 다른 조건에서 다시 돌릴 때)

결과: results/sweep_<설정>[_<태그>].json, results/fig/sweep_<설정>[_<태그>].png
측정 원칙: 설정 하나만 바꿈. 나머지는 config.py 값 그대로임. 기록표(scoreboard)에는 남기지 않음

메모: 10건 기준 한 문항이 0.1이므로 0.1 안팎 차이는 우연일 수 있음. 차이가 작으면 --limit을 늘리거나 --repeat로 다시 봄
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics as stat
from types import SimpleNamespace

import config
import model_utils
import scorer
import utils
from benchmark import read_extra, run_once

KNOBS = {
    "strategy": "출력 형식 강제 방식",
    "shots": "예시 개수",
    "template": "Chat Template 사용",
    "temperature": "생성 온도(0이면 탐욕)",
    "max_tokens": "최대 생성 토큰",
    "prompt": "덧붙인 규칙 파일",
    "score_mode": "채점 방식(느슨·엄격)",
    "mode": "검색 사용(6회차)",
    "top_k": "검색 문서 수(6회차)",
    "context_tokens": "참고 문서 토큰 상한(6회차)",
    "adapter": "LoRA Adapter(7회차)",
    "model": "모델",
    "quant": "가중치 정밀도",
}
RELOAD = {"adapter", "model", "quant"}      # 값마다 모델을 다시 올리는 설정


def setting(knob: str, value: str, base: SimpleNamespace) -> SimpleNamespace:
    a = copy.copy(base)
    if knob == "strategy":
        a.strategy = value
    elif knob == "shots":
        a.shots = int(value)
    elif knob == "template":
        a.no_template = value in ("off", "없음", "0", "false")
    elif knob == "temperature":
        t = float(value)
        a.gen = dict(config.GEN_EVAL) if t == 0 else {**config.GEN_EVAL, "do_sample": True, "temperature": t, "top_p": 1.0}
    elif knob == "max_tokens":
        a.gen = {**config.GEN_EVAL, "max_new_tokens": int(value)}
    elif knob == "prompt":
        a.extra = read_extra(value)
    elif knob == "top_k":
        a.top_k = int(value)
    elif knob == "context_tokens":
        a.context_tokens = int(value)
    return a


def label(value: str) -> str:
    return value.split("/")[-1].removesuffix(".txt") if "/" in value else value


def plot(rows: list[dict], knob: str, path) -> None:
    import matplotlib.pyplot as plt
    utils.korean_font()
    xs = [r["값"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    acc = [r["Task Accuracy"] for r in rows]
    err = [r.get("편차(표준편차)", 0) or 0 for r in rows]
    ax[0].bar(range(len(xs)), acc, 0.55, yerr=err, color="#8aa1c1", capsize=4, label="Task Accuracy")
    ax[0].plot(range(len(xs)), [r["JSON 성공률"] for r in rows], "o-", color="#c0622b", label="JSON 성공률")
    ax[0].set_xticks(range(len(xs)), xs, fontsize=8)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title(f"{KNOBS.get(knob, knob)}에 따른 점수", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[1].bar(range(len(xs)), [r["p50 지연(초)"] or 0 for r in rows], 0.55, color="#9bb08f", label="p50 지연")
    ax2 = ax[1].twinx()
    ax2.plot(range(len(xs)), [r["입력 토큰"] for r in rows], "s--", color="#555", label="입력 토큰")
    ax[1].set_xticks(range(len(xs)), xs, fontsize=8)
    ax[1].set_ylabel("초", fontsize=8)
    ax2.set_ylabel("토큰", fontsize=8)
    ax[1].set_title("비용: 지연과 입력 길이", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knob", choices=list(KNOBS))
    ap.add_argument("values", nargs="+")
    ap.add_argument("--mode", choices=["base", "rag"], default="base", help="rag면 검색 결과를 넣어 생성(6회차)")
    ap.add_argument("--strategy", default="instruction")
    ap.add_argument("--shots", type=int)
    ap.add_argument("--prompt", help="모든 값에 공통으로 덧붙일 규칙 파일")
    ap.add_argument("--dataset")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ds = a.dataset or config.data_dir() / "test_dataset.json"
    items = utils.read_json(ds)[: a.limit] if a.limit else utils.read_json(ds)
    schema, rules = scorer.load_schema(), scorer.load_rules()
    base = SimpleNamespace(strategy=a.strategy, no_template=False, score_mode="loose", shots=a.shots,
                           extra=read_extra(a.prompt), gen=None, top_k=None, context_tokens=None)
    print(f"[설정 비교 실험] {KNOBS[a.knob]} : {' · '.join(a.values)} | 도메인 {config.DOMAIN} · {len(items)}문항 · 반복 {a.repeat}")

    model = tok = None
    if a.knob not in RELOAD:
        model, tok = model_utils.load_llm()
    retriever = None
    if a.mode == "rag" or (a.knob == "mode" and "rag" in a.values):
        import retrieval
        retriever = retrieval.Retriever()

    rows, detail = [], {}
    shared_out = None
    for v in a.values:
        if a.knob in RELOAD:
            if model is not None:
                del model
                utils.free_gpu()
            if a.knob == "adapter":
                model, tok = model_utils.load_llm(adapter=None if v == "none" else v)
            elif a.knob == "model":
                model, tok = model_utils.load_llm(model_id=v)
            else:
                model, tok = model_utils.load_llm(quant=v)
        if a.knob == "score_mode":
            if shared_out is None:
                shared_out = run_once(model, tok, items, schema, rules, base, retriever if a.mode == "rag" else None, quiet=True)
            res = {"metrics": {}, "items": []}
            for it in shared_out["items"]:
                s = scorer.score_item(it["output"], it["gold"], rules, schema, v)
                res["items"].append({**it, **{k: s[k] for k in ("json_ok", "fields", "score", "exact")}})
            res["metrics"] = {**shared_out["metrics"], **scorer.aggregate(res["items"])}
            runs = [res]
        else:
            st = setting(a.knob, v, base)
            runs = []
            rt = retriever if (a.knob != "mode" and a.mode == "rag") or (a.knob == "mode" and v == "rag") else None
            for r in range(a.repeat):
                utils.set_seed(config.SEED + r)
                runs.append(run_once(model, tok, items, schema, rules, st, rt, quiet=True))
        accs = [x["metrics"]["task_accuracy"] for x in runs]
        m = runs[0]["metrics"]
        same = None
        if len(runs) > 1:
            outs = [[i["output"] for i in x["items"]] for x in runs]
            same = round(sum(len({o[k] for o in outs}) == 1 for k in range(len(items))) / len(items), 2)
        row = {"값": label(v), "Task Accuracy": round(stat.mean(accs), 3), "JSON 성공률": m["json_success"],
               **({"검색 적중": m["retrieval_hit_rate"]} if "retrieval_hit_rate" in m else {}),
               "Exact Match": m["exact_match"], **{f"{t}": round(x, 2) for t, x in m["by_type"].items()},
               "p50 지연(초)": m["latency"]["p50"], "입력 토큰": m.get("n_in_avg"), "출력 토큰": m.get("n_out_avg"),
               "VRAM(GB)": m.get("vram_peak_gb")}
        if a.repeat > 1:
            row["편차(표준편차)"] = round(stat.pstdev(accs), 3)
            row["같은 출력 비율"] = same
        rows.append(row)
        detail[v] = {"runs": [x["metrics"] for x in runs], "items": runs[-1]["items"]}
        print(f"  {label(v):>14s} | 점수 {row['Task Accuracy']:.3f} | JSON {row['JSON 성공률']:.2f} | p50 {row['p50 지연(초)']}s | 입력 {row['입력 토큰']}토큰"
              + (f" | 편차 {row['편차(표준편차)']} · 같은 출력 {same}" if a.repeat > 1 else ""))

    print()
    utils.print_table(rows)
    name = f"sweep_{a.knob}" + (f"_{a.tag}" if a.tag else "")
    utils.write_json(utils.result_path(f"{name}.json"),
                     {"knob": a.knob, "values": a.values, "measured_at": utils.now(), "dataset": str(ds), "n": len(items),
                      "repeat": a.repeat, "config": config.summary(), "table": rows, "detail": detail})
    fig = utils.fig_path(f"{name}.png")
    plot(rows, a.knob, fig)
    best = max(rows, key=lambda r: (r["Task Accuracy"], -(r["p50 지연(초)"] or 0)))
    print(f"\n점수만 보면 [{best['값']}] 설정이 가장 높음. 지연·입력 길이까지 보고 고를 것")
    print(f"저장: results/{name}.json · 그림: {fig}")


if __name__ == "__main__":
    main()
