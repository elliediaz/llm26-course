"""2회차 M2 보조: Self-Attention을 직접 계산하고, 마스크·나누기(√d)·차원을 바꿔 가며 주의 가중치가 어떻게 바뀌나 봄

  python experiments/s02/self_attention.py                     # 토큰 6개, 차원 8, causal mask, √d로 나눔
  python experiments/s02/self_attention.py --no-mask           # 미래 토큰도 봄(인코더 방식)
  python experiments/s02/self_attention.py --dim 512 --no-scale  # 차원이 크면 나누기를 빼는 순간 한 칸에 쏠림
  python experiments/s02/self_attention.py --tokens 12 --seed 3

볼 것
  ① 점수표는 (토큰 수 × 토큰 수). mask가 있으면 대각선 위쪽이 0
  ② 행마다 합이 1: 각 토큰이 어느 토큰을 얼마나 볼지 나눠 가짐(부드러운 분류)
  ③ 행별 엔트로피(퍼짐 정도). √d로 나누지 않으면 차원이 클수록 엔트로피가 0에 가까워짐(한 토큰만 봄)
"""
from __future__ import annotations

import argparse
import math

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    T, d = a.tokens, a.dim
    x = torch.randn(1, T, d)
    Wq, Wk, Wv = (torch.nn.Linear(d, d, bias=False) for _ in range(3))
    q, k, v = Wq(x), Wk(x), Wv(x)
    s = q @ k.transpose(-2, -1)
    if not a.no_scale:
        s = s / math.sqrt(d)
    if not a.no_mask:
        mask = torch.ones(T, T).tril().bool()
        s = s.masked_fill(~mask, float("-inf"))
    w = s.softmax(-1)
    y = w @ v
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=140)
    print(f"설정 : 토큰 {T} · 차원 {d} · mask {'없음' if a.no_mask else 'causal'} · √d 나누기 {'안 함' if a.no_scale else '함'}")
    print("주의 가중치(행: 만드는 토큰, 열: 보는 토큰)\n", w[0].detach())
    ent = -(w[0] * (w[0] + 1e-12).log()).sum(-1)
    print("행 합 :", [round(v, 2) for v in w[0].sum(-1).tolist()])
    print("행별 엔트로피(클수록 고르게 봄) :", [round(abs(v), 2) for v in ent.tolist()])
    print("행별 최대 가중치 :", [round(v, 2) for v in w[0].max(-1).values.tolist()])
    print("출력 모양 :", tuple(y.shape))


if __name__ == "__main__":
    main()
