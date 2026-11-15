"""모델 한 번 실행하기: 로딩 시간·응답·토큰 수·지연·VRAM을 한 번에 본다.

  python src/run_model.py --prompt "정기 점검 시간을 알려 줘"
  python src/run_model.py --model Qwen/Qwen3-0.6B --quant bf16 --prompt "..." --service   # GEN_SERVICE로 생성
  python src/run_model.py --prompt "..." --raw          # Chat Template 없이(비교용)
"""
from __future__ import annotations

import argparse

import config
import model_utils
import utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--quant", default=None, choices=["bf16", "nf4"])
    ap.add_argument("--service", action="store_true", help="GEN_SERVICE(팀 확정 설정)로 생성")
    ap.add_argument("--raw", action="store_true", help="Chat Template 없이 입력")
    a = ap.parse_args()
    utils.set_seed()
    utils.vram_reset()
    model, tok = model_utils.load_llm(a.model, a.quant)
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
    g = model_utils.generate(model, tok, msgs, config.GEN_SERVICE if a.service else config.GEN_EVAL, use_template=not a.raw)
    print(f"\n{g['text']}\n")
    print(f"입력 {g['n_in']}토큰 · 출력 {g['n_out']}토큰 · {g['latency']}s · {g['n_out'] / g['latency']:.1f} 토큰/초 · VRAM 최대 {utils.vram_peak_gb()}GB")


if __name__ == "__main__":
    main()
