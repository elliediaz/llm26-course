"""2회차 M4 보조: 같은 확률 분포에서 토큰을 고르는 방법 = 생성 설정. 온도·Top-k·Top-p를 바꿔 뽑아 봄

  python experiments/s02/sampling.py                                         # 기본 설정 6가지 × 30번 뽑기
  python experiments/s02/sampling.py --text "환불 규정에 따르면"               # 내 문장
  python experiments/s02/sampling.py --settings t=0 t=0.7 t=0.7,p=0.5 t=2.0,k=3 --draws 50

설정 쓰는 법 : t=온도, k=상위 k개만, p=누적 확률 p까지만. 쉼표로 이음(예: t=1.0,p=0.9). t=0은 탐욕(항상 1등)
볼 것
  ① 온도가 높을수록, p가 클수록 '서로 다른 토큰 수'가 늘어남. 탐욕은 항상 1
  ② k·p는 꼬리(드문 후보)를 잘라 높은 온도에서도 엉뚱한 토큰을 막음
  ③ 분포 자체(모델의 분류 결과)는 하나. 바뀌는 것은 '고르는 규칙'뿐
"""
from __future__ import annotations

from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()          # 가중치 로딩 진행 막대 끄기

import argparse
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse(s: str) -> dict:
    out = {"temperature": 1.0, "top_k": 0, "top_p": 1.0}
    for part in s.split(","):
        k, v = part.split("=")
        out[{"t": "temperature", "k": "top_k", "p": "top_p"}[k.strip()]] = float(v) if k.strip() != "k" else int(v)
    return out


def pick(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature == 0:
        return logits.argmax()
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_k:
        kth = torch.topk(probs, top_k).values[-1]
        probs[probs < kth] = 0
    if top_p < 1.0:
        s, i = torch.sort(probs, descending=True)
        drop = s.cumsum(0) - s >= top_p
        probs[i[drop]] = 0
    return torch.multinomial(probs / probs.sum(), 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--text", default="이번 패치에서 가장 크게 달라진 점은")
    ap.add_argument("--settings", nargs="+", default=["t=0", "t=0.3", "t=1.0", "t=1.5", "t=1.0,k=5", "t=1.0,p=0.9"])
    ap.add_argument("--draws", type=int, default=30)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        logits = model(**tok(a.text, return_tensors="pt").to(dev)).logits[0, -1].float()
    p = logits.softmax(-1)
    top = p.topk(5)
    print(f"'{a.text}' 다음 토큰 상위 5 : " + ", ".join(f"{tok.decode([i])!r} {v:.3f}" for v, i in zip(top.values.tolist(), top.indices.tolist())))
    torch.manual_seed(42)
    print(f"\n| 설정 | 서로 다른 토큰 수(/{a.draws}) | 많이 나온 것 |\n|---|---|---|")
    for st in a.settings:
        picks = [tok.decode([pick(logits.clone(), **parse(st)).item()]) for _ in range(a.draws)]
        print(f"| {st} | {len(set(picks))} | {Counter(picks).most_common(3)} |")


if __name__ == "__main__":
    main()
