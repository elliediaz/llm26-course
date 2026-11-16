"""2회차 복습 실습 R: OR는 직선 하나로 나뉘고 XOR는 나뉘지 않는다

  python src/xor_demo.py                    # AND·OR·XOR × 세 모델(약 10초)
  python src/xor_demo.py --hidden 2         # 은닉 뉴런 수를 줄여 보기
  python src/xor_demo.py --act tanh         # 활성화 함수 바꾸기(relu·tanh·sigmoid)

모델 세 가지
  단층          입력 2 → 출력 2                       직선 하나로 나눔
  2층·활성화 X  입력 2 → 은닉 8 → 출력 2              층만 쌓으면 여전히 직선 하나
  2층·활성화 O  입력 2 → 은닉 8 → ReLU → 출력 2       공간을 접어 XOR도 나눔
결과: results/xor_demo.json, results/fig/xor_boundary.png
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

import utils

PROBLEMS = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}
CORNERS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def make_data(labels, n_per=60, noise=0.12, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([c + rng.normal(0, noise, (n_per, 2)) for c in CORNERS]).astype(np.float32)
    y = np.repeat(labels, n_per)
    return torch.tensor(x), torch.tensor(y)


def build(kind, hidden, act):
    if kind == "단층":
        return nn.Linear(2, 2)
    if kind == "2층·활성화 X":
        return nn.Sequential(nn.Linear(2, hidden), nn.Linear(hidden, 2))
    return nn.Sequential(nn.Linear(2, hidden), ACTS[act](), nn.Linear(hidden, 2))


def train(model, x, y, steps=600, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    return round(acc, 3), round(loss.item(), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--act", default="relu", choices=list(ACTS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    utils.set_seed(a.seed)

    kinds = ["단층", "2층·활성화 X", f"2층·{a.act}"]
    rows, grid = [], {}
    for pname, labels in PROBLEMS.items():
        x, y = make_data(labels, seed=a.seed)
        for kind in kinds:
            m = build(kind, a.hidden, a.act)
            acc, loss = train(m, x, y)
            rows.append({"문제": pname, "모델": kind, "정확도": acc, "손실": loss})
            grid[(pname, kind)] = (m, x, y)
    utils.print_table(rows)
    utils.write_json(utils.result_path("xor_demo.json"), {"args": vars(a), "rows": rows})

    import matplotlib.pyplot as plt
    utils.korean_font()
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    xs = np.linspace(-0.5, 1.5, 200, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs)
    pts = torch.tensor(np.c_[gx.ravel(), gy.ravel()])
    for i, pname in enumerate(PROBLEMS):
        for j, kind in enumerate(kinds):
            m, x, y = grid[(pname, kind)]
            with torch.no_grad():
                zz = m(pts).argmax(1).numpy().reshape(gx.shape)
            ax = axes[i, j]
            ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], colors=["#dbe4f0", "#f3d9c4"])
            ax.scatter(x[:, 0], x[:, 1], c=["#2b5c9e" if v == 0 else "#c0622b" for v in y.numpy()], s=8)
            acc = next(r["정확도"] for r in rows if r["문제"] == pname and r["모델"] == kind)
            ax.set_title(f"{pname} · {kind}  정확도 {acc:.2f}", fontsize=10)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    fig.tight_layout()
    out = utils.fig_path("xor_boundary.png")
    fig.savefig(out, dpi=130)
    print("그림:", out)


if __name__ == "__main__":
    main()
