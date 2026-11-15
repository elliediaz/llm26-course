"""2회차 휴식 직전 실행, 실습 2 ⑦에서 해석: 생성 설정 비교표. 같은 입력에 Temperature·Top-k·Top-p 조합을 바꿔 여러 번 생성해 본다.

  python src/gen_params_sweep.py                    # 기본 6개 조합 × 프롬프트 6건 × 3회
  python src/gen_params_sweep.py --repeats 10 --ids G-GP-5   # 도전과제 ①: 같은 설정 10회 반복 시 흔들림
  python src/gen_params_sweep.py --model Qwen/Qwen3-0.6B     # 작은 모델로 빠르게

표에서 볼 것
  다양성  같은 입력 n회 중 서로 다른 출력 수(1이면 매번 같음)
  JSON    형식형 프롬프트에서 JSON으로 읽히는 비율
결정
  평가(5~7회차)는 GEN_EVAL(greedy) 고정: 측정값이 흔들리면 개선을 판단할 수 없음
  서비스(8~10회차)는 팀이 정한 GEN_SERVICE → src/config.py에 기록하고 이유를 한 줄로 남김
결과: results/gen_sweep.json, results/gen_sweep.md
"""
from __future__ import annotations

import argparse
import json

import config
import model_utils
import scorer
import utils

COMBOS = {
    "greedy": {"do_sample": False},
    "T0.3·p0.9": {"do_sample": True, "temperature": 0.3, "top_p": 0.9},
    "T0.7·p0.9": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    "T0.7·k20": {"do_sample": True, "temperature": 0.7, "top_k": 20, "top_p": 1.0},
    "T1.0·p0.95": {"do_sample": True, "temperature": 1.0, "top_p": 0.95},
    "T1.3·p1.0": {"do_sample": True, "temperature": 1.3, "top_p": 1.0},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ids", nargs="*", help="특정 프롬프트 id 만")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    a = ap.parse_args()
    prompts = utils.read_json(config.data_dir() / "gen_prompts.json")
    if a.ids:
        prompts = [p for p in prompts if p["id"] in a.ids]
    model, tok = model_utils.load_llm(a.model)
    rows, samples = [], []
    for p in prompts:
        msgs = [{"role": "user", "content": p["prompt"]}]
        for name, gen in COMBOS.items():
            utils.set_seed()
            outs = [model_utils.generate(model, tok, msgs, {**gen, "max_new_tokens": a.max_new_tokens})
                    for _ in range(a.repeats)]
            texts = [o["text"] for o in outs]
            json_ok = sum(scorer.parse(t)[0] is not None or t.strip().startswith("[") for t in texts) / len(texts)
            rows.append({"프롬프트": p["id"], "유형": p["kind"], "설정": name, "다양성": len(set(texts)),
                         "평균 길이": round(sum(map(len, texts)) / len(texts)),
                         "JSON": round(json_ok, 2) if p["kind"] == "형식형" else None,
                         "평균 지연(s)": round(sum(o["latency"] for o in outs) / len(outs), 2)})
            samples.append({"id": p["id"], "setting": name, "outputs": texts})
            print(f"  {p['id']} {name:10s} 다양성 {len(set(texts))}/{a.repeats} · {texts[0][:50]!r}")
    print()
    md = utils.print_table(rows)
    print(f"\n현재 서비스 설정 GEN_SERVICE = {json.dumps(config.GEN_SERVICE, ensure_ascii=False)}")
    print("→ 팀 과제에 맞는 값과 이유 한 줄을 src/config.py의 GEN_SERVICE·GEN_SERVICE_REASON에 기록")
    utils.write_json(utils.result_path("gen_sweep.json"), {"model": a.model or config.MODEL_ID, "repeats": a.repeats,
                                                           "rows": rows, "samples": samples, "measured_at": utils.now()})
    utils.result_path("gen_sweep.md").write_text("# 생성 설정 비교표\n\n" + md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
