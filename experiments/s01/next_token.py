"""1회차 실습 E 보조: LLM은 '다음 토큰 고르기' 분류기. 출력층 크기, 확률 분포, 한 토큰씩 고르는 과정을 봄

  python experiments/s01/next_token.py                                   # 기본 문장
  python experiments/s01/next_token.py --text "환불은 구매 후 며칠"         # 내 문장
  python experiments/s01/next_token.py --steps 25 --topk 3                # 더 길게, 후보 3개씩 표시
  python experiments/s01/next_token.py --text "The capital of Korea is"   # 영어는 분포가 어떻게 다른가

볼 것
  ① 어휘 크기 = 출력층 크기 = 클래스 수(Qwen3 약 15만)
  ② 각 단계의 1등 확률. 1등이 압도적인 단계와 후보가 갈리는 단계가 섞여 있음
  ③ 생성 = 분류를 반복한 것. 1등만 고르면(탐욕) 매번 같은 글이 나옴
"""
from __future__ import annotations

from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()          # 가중치 로딩 진행 막대 끄기

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--text", default="결제했는데 아이템이 안")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()   # T4는 bfloat16 미지원
    print(f"어휘 크기(클래스 수) {model.config.vocab_size:,} · 출력층 {model.lm_head}")

    x = tok(a.text, return_tensors="pt").input_ids.to(dev)
    print(f"\n단계 | 1등(확률) | 다른 후보")
    for step in range(1, a.steps + 1):
        with torch.no_grad():
            p = model(x).logits[0, -1].float().softmax(-1)
        top = p.topk(a.topk)
        cand = [(tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
        print(f"{step:4d} | {cand[0][0]!r} ({cand[0][1]}) | " + ", ".join(f"{t!r} {v}" for t, v in cand[1:]))
        x = torch.cat([x, top.indices[:1].view(1, 1)], dim=1)
    print("\n이어 쓴 글:", tok.decode(x[0], skip_special_tokens=True))
    print("확률 합(마지막 단계):", round(p.sum().item(), 4))


if __name__ == "__main__":
    main()
