"""2회차 1교시 휴식 직전에 걸어 두고 실습 2 ⑥에서 해석: 생성 설정 비교표. 같은 입력에 Temperature·Top-k·Top-p 조합을 바꿔 여러 번 생성해 본다.

  python src/gen_params_sweep.py                    # 기본 6개 조합 × 프롬프트 6건 × 3회
  python src/gen_params_sweep.py --model Qwen/Qwen3-0.6B --ids G-GP-5 --repeats 10 --temperature 0.7 --top-p 0.9
                                                    # 해보기 G: 같은 설정 10회 반복 시 편차(기본 비교표는 그대로 둠)
  python src/gen_params_sweep.py --model Qwen/Qwen3-0.6B     # 작은 모델로 빠르게
  python src/gen_params_sweep.py --model Qwen/Qwen3-0.6B --kind 형식형 --temperature 0.5 --top-p 0.8
                                                    # 해보기 ②: 내 설정 하나를 greedy와 비교(결과는 gen_sweep_mine.md에 쌓임)

표에서 볼 것
  다양성        같은 입력 n회 중 서로 다른 출력 수(1이면 매번 같음)
  JSON(느슨)    형식형 프롬프트에서 코드 울타리(```json … ```)를 벗기면 JSON으로 읽히는 비율
  JSON만(엄격)  울타리·설명 없이 JSON만 낸 비율("JSON으로만 답해"를 그대로 지켰나, 5회차 엄격 채점과 같은 기준)
결정
  평가(5~7회차)는 GEN_EVAL(greedy) 고정: 측정값이 달라지면 개선을 판단할 수 없음
  서비스(8~10회차)는 각자 정한 GEN_SERVICE → src/config.py에 기록하고 이유를 한 줄로 남김
결과: results/gen_sweep.json, results/gen_sweep.md
"""
from __future__ import annotations

import argparse
import json
import re

import config
import model_utils
import scorer
import utils

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def json_check(text: str) -> tuple[bool, bool]:
    """(느슨, 엄격) 두 기준으로 JSON 여부를 판정. 객체({…})와 배열([…]) 모두 인정."""
    def ok(s: str) -> bool:
        try:
            json.loads(s)
            return True
        except ValueError:
            return False
    t = text.strip()
    m = _FENCE.search(t)
    loose = ok(m.group(1).strip() if m else t) or scorer.parse(t)[0] is not None
    return loose, ok(t)


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
    ap.add_argument("--kind", choices=["사실형", "창작형", "형식형"], help="이 유형의 프롬프트만")
    ap.add_argument("--temperature", type=float, help="내 설정: 지정하면 greedy와 이 설정 둘만 비교")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    a = ap.parse_args()
    combos = COMBOS
    if a.temperature is not None:
        mine = {"do_sample": True, "temperature": a.temperature, "top_p": a.top_p}
        if a.top_k:
            mine["top_k"] = a.top_k
        name = f"T{a.temperature}" + (f"·p{a.top_p}" if a.top_p < 1 else "") + (f"·k{a.top_k}" if a.top_k else "")
        combos = {"greedy": COMBOS["greedy"], name: mine}
    prompts = utils.read_json(config.data_dir() / "gen_prompts.json")
    if a.ids:
        prompts = [p for p in prompts if p["id"] in a.ids]
    if a.kind:
        prompts = [p for p in prompts if p["kind"] == a.kind]
    model, tok = model_utils.load_llm(a.model)
    rows, samples = [], []
    for p in prompts:
        msgs = [{"role": "user", "content": p["prompt"]}]
        for name, gen in combos.items():
            utils.set_seed()
            outs = [model_utils.generate(model, tok, msgs, {**gen, "max_new_tokens": a.max_new_tokens})
                    for _ in range(a.repeats)]
            texts = [o["text"] for o in outs]
            checks = [json_check(t) for t in texts]
            loose = sum(c[0] for c in checks) / len(texts)
            strict = sum(c[1] for c in checks) / len(texts)
            fmt = p["kind"] == "형식형"
            rows.append({"프롬프트": p["id"], "유형": p["kind"], "설정": name, "다양성": len(set(texts)),
                         "평균 길이": round(sum(map(len, texts)) / len(texts)),
                         "JSON(느슨)": round(loose, 2) if fmt else None, "JSON만(엄격)": round(strict, 2) if fmt else None,
                         "평균 지연(s)": round(sum(o["latency"] for o in outs) / len(outs), 2)})
            samples.append({"id": p["id"], "setting": name, "outputs": texts})
            print(f"  {p['id']} {name:10s} 다양성 {len(set(texts))}/{a.repeats} · {texts[0][:50]!r}")
            if a.temperature is not None:                   # 해보기 ②: 반복마다 무엇이 바뀌는지 직접 봄
                for k, t in enumerate(texts, 1):
                    print(f"      {k}회: {' '.join(t.split())[:90]}")
    print()
    md = utils.print_table(rows)
    print(f"\n현재 서비스 설정 GEN_SERVICE = {json.dumps(config.GEN_SERVICE, ensure_ascii=False)}")
    print("→ 내 과제에 맞는 값과 이유 한 줄을 src/config.py의 GEN_SERVICE·GEN_SERVICE_REASON에 기록")
    if a.temperature is not None:                       # 해보기: 기본 비교표를 덮어쓰지 않고 따로 쌓음
        log = utils.result_path("gen_sweep_mine.md")
        prev = log.read_text(encoding="utf-8") if log.exists() else "# 내 생성 설정 시험 기록(해보기 ②)\n"
        log.write_text(prev + f"\n## {utils.now()} · {a.model or config.MODEL_ID} · 반복 {a.repeats}\n\n" + md + "\n", encoding="utf-8")
        print(f"기록: {log}")
        return
    utils.write_json(utils.result_path("gen_sweep.json"), {"model": a.model or config.MODEL_ID, "repeats": a.repeats,
                                                           "rows": rows, "samples": samples, "measured_at": utils.now()})
    utils.result_path("gen_sweep.md").write_text("# 생성 설정 비교표\n\n" + md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
