"""1회차 시연(트랜스포머 이전 구조의 한계): 첫 숫자 기억하기 과제로 RNN·LSTM·어텐션을 비교한다.

과제    숫자 L개가 한 줄로 들어오고, 정답은 맨 처음 숫자(0~9 중 하나 고르기 = 10칸 분류)
        예) L=5  : 7 3 9 1 4 → 정답 7  /  L=60 : 7 3 9 … (60개) → 정답 7
        숫자는 무작위라 맨 앞 하나를 끝까지 들고 가야만 맞힘. 찍으면 0.1
RNN     한 칸씩 읽으며 상태 하나를 고침. 멀리 떨어진 정보는 곱셈이 거듭되며 희미해짐(기울기 소실)
LSTM    문(gate)으로 '기억할 것'을 따로 들고 감. 실제 리뷰에서는 RNN보다 오래 버티지만(실습 C) 학습이 짧으면 이 과제에서도 길이 20부터 무너지고, 여전히 한 칸씩 차례로 계산
어텐션  모든 칸이 맨 앞 칸을 한 번에 직접 봄(트랜스포머의 핵심 부품). 길이와 상관없이 거리가 1

  python src/seq_limits.py                     # 길이 5·10·20·40, 세 구조 각 1,000스텝(개인 CPU 서버 약 1~2분)
  python src/seq_limits.py --steps 3000        # 더 오래 학습하면 LSTM이 버티는 길이가 늘어나나(해보기)
결과: results/seq_limits.json, results/fig/seq_limits.png
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_tools as T
import utils


class Recurrent(nn.Module):
    def __init__(self, cell: str, d: int = 64):
        super().__init__()
        self.emb = nn.Embedding(10, d)
        self.rnn = (nn.RNN if cell == "rnn" else nn.LSTM)(d, d, batch_first=True)
        self.head = nn.Linear(d, 10)

    def forward(self, x):                      # [N,L] → [N,L,d] → 마지막 상태 [N,d] → [N,10]
        out, _ = self.rnn(self.emb(x))
        return self.head(out[:, -1])


class Attn(nn.Module):
    def __init__(self, d: int = 64, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(10, d)
        self.pos = nn.Embedding(max_len, d)    # 위치 정보(어텐션은 순서를 모르므로 따로 넣음)
        self.layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.0)
        self.head = nn.Linear(d, 10)

    def forward(self, x):                      # 마지막 칸이 모든 칸(맨 앞 포함)을 한 번에 봄
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        return self.head(self.layer(h)[:, -1])


def batch(n, L, dev, g=None):
    x = torch.randint(0, 10, (n, L), generator=g)
    return x.to(dev), x[:, 0].to(dev)


def run(kind, L, steps, dev):
    utils.set_seed()
    m = (Attn() if kind == "attn" else Recurrent(kind)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = batch(128, L, dev)
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        x, y = batch(2000, L, dev, torch.Generator().manual_seed(1))
        acc = (m(x).argmax(-1) == y).float().mean().item()
    return round(acc, 3), round(time.perf_counter() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="*", default=[5, 10, 20, 40])
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()
    dev = T.device()
    names = {"rnn": "RNN", "lstm": "LSTM", "attn": "어텐션(트랜스포머 층 1개)"}
    rows, res = [], {}
    for kind in names:
        res[kind] = {}
        for L in a.lengths:
            acc, sec = run(kind, L, a.steps, dev)
            res[kind][L] = {"acc": acc, "sec": sec}
            print(f"  {names[kind]:<22} 길이 {L:>4}  정확도 {acc:.3f}  ({sec}초)", flush=True)
        rows.append({"구조": names[kind], **{f"길이 {L}": res[kind][L]["acc"] for L in a.lengths}})
    print("\n'첫 숫자 기억하기' 정확도(찍으면 0.1)")
    utils.print_table(rows)
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for kind, mk in zip(names, "osD"):
        ax.plot(a.lengths, [res[kind][L]["acc"] for L in a.lengths], marker=mk, label=names[kind])
    ax.axhline(0.1, color="gray", ls="--", lw=1)
    ax.text(a.lengths[0], 0.13, "찍기 0.1", ha="left", color="gray", fontsize=9)
    ax.set_xlabel("입력 길이(맨 앞 숫자까지의 거리)")
    ax.set_ylabel("정확도")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"첫 숫자 기억하기: 같은 {a.steps}스텝 학습", fontsize=11)
    fig.tight_layout()
    p = utils.fig_path("seq_limits.png")
    fig.savefig(p, dpi=130)
    print("그림:", p)
    utils.write_json(utils.result_path("seq_limits.json"), {"args": vars(a), "result": res, "measured_at": utils.now()})


if __name__ == "__main__":
    main()
