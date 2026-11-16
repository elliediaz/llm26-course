"""1회차 실습 B: 같은 이미지 분류를 Dense(MLP)와 CNN으로 풀어 비교한다.

데이터  Fashion-MNIST(Zalando, MIT): 28×28 흑백 옷 사진 10종, 학습 6만·시험 1만  →  data/common/fashion_mnist/
문제    10종 중 하나 고르기(분류). 출력층 10칸 + 교차 엔트로피
MLP     [N,1,28,28] → 펼치기 [N,784] → Linear 128 → ReLU → Linear 10           (픽셀 위치 관계를 버림)
CNN     [N,1,28,28] → Conv 3×3 16장 → ReLU → MaxPool → [N,16,14,14]
                    → Conv 3×3 32장 → ReLU → MaxPool → [N,32,7,7] → 펼치기 [N,1568] → Linear 128 → Linear 10
        (작은 창을 밀며 같은 무늬를 어디서든 찾음. 가까운 픽셀끼리의 관계를 살림)

  python src/nn_cnn.py                               # MLP·CNN 각 2에폭(개인 CPU 서버 약 15초)
  python src/nn_cnn.py --model cnn --channels 32 64 --epochs 3 --lr 0.002
  python src/nn_cnn.py --target group                # 10종 → 4묶음(상의·하의/원피스·신발·가방)으로 바꿔 풀기(해보기 B)
결과: results/nn_cnn.json, results/fig/nn_cnn_pred.png(예측 예시), results/fig/nn_cnn_maps.png(첫 층 특징 맵)
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import nn_tools as T
import utils

GROUPS = {"상의": [0, 2, 4, 6], "하의·원피스": [1, 3], "신발": [5, 7, 9], "가방": [8]}


class MLP(nn.Module):
    def __init__(self, n_out=10, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, n_out=10, ch=(16, 32), hidden=128, dropout=0.0):
        super().__init__()
        c1, c2 = ch
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 28×28 → 14×14
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 14×14 → 7×7
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c2 * 7 * 7, hidden), nn.ReLU(),
                                        *([nn.Dropout(dropout)] if dropout else []), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.classifier(self.features(x))


def to_group(y):
    m = torch.zeros(10, dtype=torch.long)
    for g, (_, idx) in enumerate(GROUPS.items()):
        m[idx] = g
    return m[y]


def run(kind, data, a, names):
    xtr, ytr, xte, yte = data
    n_out = len(names)
    model = MLP(n_out) if kind == "mlp" else CNN(n_out, tuple(a.channels), dropout=a.dropout)
    print(f"\n[{kind.upper()}] 모델 모양")
    T.shape_trace(model, xtr[:a.batch])
    tr = DataLoader(TensorDataset(xtr, ytr), batch_size=a.batch, shuffle=True)
    te = DataLoader(TensorDataset(xte, yte), batch_size=1000)

    def ev(m):
        out, yy = T.predict(m, te)
        return {"acc": round((out.argmax(1) == yy).float().mean().item(), 4)}
    hist = T.fit(model, tr, nn.CrossEntropyLoss(), a.epochs, a.lr, ev, kind.upper())
    out, yy = T.predict(model, te)
    pred = out.argmax(1)
    per = {names[c]: round((pred[yy == c] == c).float().mean().item(), 3) for c in range(n_out)}
    conf = torch.zeros(n_out, n_out, dtype=torch.long)
    for t, p in zip(yy, pred):
        conf[t, p] += 1
    pairs = sorted(((conf[i, j].item(), names[i], names[j]) for i in range(n_out) for j in range(n_out) if i != j), reverse=True)[:3]
    print("  종류별 정확도:", ", ".join(f"{k} {v}" for k, v in per.items()))
    print("  가장 많이 헷갈린 쌍:", "; ".join(f"{t}→{p} {n}건" for n, t, p in pairs))
    return model, {"params": sum(p.numel() for p in model.parameters()), "history": hist, "final_acc": hist[-1]["acc"],
                   "train_sec": hist[-1]["sec"], "per_class": per, "confused": [f"{t}→{p} {n}" for n, t, p in pairs]}, pred


def plot_pred(xte, yte, pred, names):
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 8, figsize=(13, 3.8))
    for k, ax in enumerate(axes.flat):
        ax.imshow(xte[k, 0], cmap="gray_r")
        ok = pred[k] == yte[k]
        ax.set_title(f"{names[pred[k]]}" + ("" if ok else f"\n(정답 {names[yte[k]]})"), fontsize=9,
                     color="#1F3B8C" if ok else "#C8281E")
        ax.axis("off")
    fig.suptitle("CNN 예측(빨간색은 틀림)", fontsize=11)
    fig.tight_layout()
    p = utils.fig_path("nn_cnn_pred.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


@torch.no_grad()
def plot_maps(model, x):
    utils.korean_font()
    import matplotlib.pyplot as plt
    model.eval().cpu()
    conv1 = model.features[0]
    maps = torch.relu(conv1(x[:1]))[0]           # [16,28,28]: 필터 16장이 각자 찾은 무늬
    n = min(8, maps.shape[0])
    fig, axes = plt.subplots(2, n + 1, figsize=(1.6 * (n + 1), 4.0))
    axes[0, 0].imshow(x[0, 0], cmap="gray_r")
    axes[0, 0].set_title("입력 28×28", fontsize=9)
    axes[1, 0].axis("off")
    for k in range(n):
        axes[0, k + 1].imshow(conv1.weight[k, 0], cmap="RdBu_r")
        axes[0, k + 1].set_title(f"필터 {k + 1}(3×3)", fontsize=8)
        axes[1, k + 1].imshow(maps[k], cmap="viridis")
        axes[1, k + 1].set_title("특징 맵 28×28", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("첫 합성곱 층: 위는 학습된 필터, 아래는 그 필터로 훑은 결과(밝을수록 무늬가 강함)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.6)
    p = utils.fig_path("nn_cnn_maps.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["mlp", "cnn", "both"], default="both")
    ap.add_argument("--target", choices=["class10", "group"], default="class10")
    ap.add_argument("--channels", type=int, nargs=2, default=[16, 32])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, help="학습 이미지 앞에서 N장만(빠른 확인용)")
    a = ap.parse_args()
    utils.set_seed()
    xtr, ytr, xte, yte = T.load_fashion(a.limit)
    names = T.FASHION_CLASSES
    if a.target == "group":
        ytr, yte, names = to_group(ytr), to_group(yte), list(GROUPS)
    print(f"데이터: 학습 {len(xtr):,}장 · 시험 {len(xte):,}장, 이미지 모양 {tuple(xtr.shape[1:])}, 클래스 {len(names)}개: {', '.join(names)}")
    res, cnn = {}, None
    for kind in (["mlp", "cnn"] if a.model == "both" else [a.model]):
        model, r, pred = run(kind, (xtr, ytr, xte, yte), a, names)
        res[kind] = r
        if kind == "cnn":
            cnn = model
            print("그림:", plot_pred(xte, yte, pred, names))
            print("그림:", plot_maps(cnn, xte))
    if len(res) == 2:
        utils.print_table([{"모델": k.upper(), "파라미터": r["params"], "시험 정확도": r["final_acc"], "학습 초": r["train_sec"]}
                           for k, r in res.items()])
    print("저장:", utils.write_json(utils.result_path("nn_cnn.json"), {"args": vars(a), "classes": names, "result": res}))


if __name__ == "__main__":
    main()
