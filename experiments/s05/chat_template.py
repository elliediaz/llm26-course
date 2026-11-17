"""5회차 사건 1: Chat Template이 실제로 무엇을 만드는지 눈으로 봄

  python experiments/s05/chat_template.py                                  # 기본 모델(config.MODEL_ID)
  python experiments/s05/chat_template.py --models Qwen/Qwen3-0.6B skt/A.X-4.0-Light   # 모델마다 특수 토큰이 다름
  python experiments/s05/chat_template.py --no-system                      # 시스템 프롬프트를 뺐을 때
  python experiments/s05/chat_template.py --no-gen-prompt                  # 'assistant 차례' 표시를 뺐을 때
  python experiments/s05/chat_template.py --generate                       # 템플릿 있음·없음으로 실제 생성
  python experiments/s05/chat_template.py --generate --text "그거 또 안 돼요"

볼 것
  ① 역할(system·user·assistant)이 특수 토큰으로 감싸여 글 한 덩어리가 됨
  ② 끝에 붙는 assistant 차례 표시가 없으면 모델이 '답을 시작할 자리'를 모름
  ③ --generate : 템플릿 없이 넣으면 문의를 이어 쓰거나 끝없이 늘어지는 일이 잦음(출력 토큰 수, 끝맺음 여부)
메모: 토큰 수와 출력은 모델마다 다름. 작은 모델(Qwen3-0.6B)은 CPU에서도 몇십 초 안에 돌아감
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import model_utils  # noqa: E402
import prompt_utils  # noqa: E402
import scorer  # noqa: E402


def show(model_id: str, msgs: list[dict], gen_prompt: bool) -> None:
    tok = model_utils.load_tokenizer(model_id)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen_prompt, **config.CHAT_TEMPLATE_KWARGS)
    plain = "\n\n".join(m["content"] for m in msgs)
    n_t, n_p = len(tok(text)["input_ids"]), len(tok(plain)["input_ids"])
    specials = sorted({t for t in tok.all_special_tokens if t in text})
    print(f"\n===== {model_id} =====")
    print(text if len(text) < 1600 else text[:700] + "\n   ...(중략)...\n" + text[-500:])
    print(f"\n템플릿 적용 {n_t}토큰 · 내용만 이어 붙임 {n_p}토큰 · 늘어난 토큰 {n_t - n_p}")
    print(f"쓰인 특수 토큰 : {', '.join(specials) if specials else '(없음)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[config.MODEL_ID])
    ap.add_argument("--text", default="결제했는데 루멘석이 안 들어왔어요")
    ap.add_argument("--no-system", action="store_true")
    ap.add_argument("--no-gen-prompt", action="store_true")
    ap.add_argument("--generate", action="store_true", help="첫 모델로 템플릿 있음·없음 생성 비교")
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()

    schema = scorer.load_schema()
    msgs = prompt_utils.build_messages(a.text, schema, "instruction")
    if a.no_system:
        msgs = msgs[1:]
    for mid in a.models:
        show(mid, msgs, not a.no_gen_prompt)

    if a.generate:
        model, tok = model_utils.load_llm(model_id=a.models[0])
        gen = {**config.GEN_EVAL, "max_new_tokens": a.max_new}
        print("\n템플릿 있음·없음 생성 비교(탐욕 생성)")
        for use in (True, False):
            g = model_utils.generate(model, tok, msgs, gen, use_template=use)
            obj, why = scorer.parse(g["text"])
            ended = "끝맺음" if g["n_out"] < a.max_new else "최대 길이에서 잘림"
            print(f"\n[{'템플릿 있음' if use else '템플릿 없음'}] 출력 {g['n_out']}토큰({ended}) · {g['latency']}초 · JSON {'읽힘' if obj else '못 읽음(' + why + ')'}")
            print("  " + g["text"][:400].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
