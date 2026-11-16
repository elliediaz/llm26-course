"""1회차 시연(트랜스포머 이전 구조의 한계): VAE(변분 오토인코더). 옷 사진을 숫자 몇 개로 압축했다가 다시 그리는 생성 모델.

데이터  Fashion-MNIST(실습 B와 같은 옷 사진, 학습 6만·시험 1만)  →  data/common/fashion_mnist/
구조    인코더  [N,1,28,28] → 펼치기 [N,784] → Linear 256 → ReLU → 평균 μ [N,z]·흩어짐 logσ² [N,z]
        뽑기    z = μ + σ × 무작위 수   (같은 사진도 매번 조금씩 다른 z: 비슷한 사진이 가까이 모이게 함)
        디코더  [N,z] → Linear 256 → ReLU → Linear 784 → sigmoid → [N,1,28,28]
손실    재구성(원본 픽셀과 얼마나 같은가) + KL(z가 한쪽에 몰리지 않게 고르게 펴 두는 벌점)
잘 되는 것  비슷한 사진 다시 그리기, 두 사진 사이를 부드럽게 잇기, 비슷한 옷끼리 잠재 공간에 모임
안 되는 것  무늬·글자 같은 세부는 흐려짐(사진 한 장을 숫자 z개에 담음), 길이가 바뀌는 글을 다른 글로 바꾸는 일은 못 함

  python src/nn_vae.py                 # z 2개, 3에폭(개인 CPU 서버 약 10초)
  python src/nn_vae.py --z 16          # 담을 숫자를 늘리면 덜 흐려지나(해보기)
  python src/nn_vae.py --epochs 10     # 오래 학습하면
결과: results/nn_vae.json, results/fig/nn_vae_recon.png(원본·재구성), nn_vae_interp.png(사이 잇기),
      nn_vae_sample.png(무작위 z로 그리기), nn_vae_latent.png(z가 2개일 때 잠재 공간 지도)
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import nn_tools as T
import utils


class VAE(nn.Module):
    def __init__(self, z: int = 2, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, z)                 # 평균
        self.logvar = nn.Linear(hidden, z)             # 흩어짐(로그 분산)
        self.dec = nn.Sequential(nn.Linear(z, hidden), nn.ReLU(), nn.Linear(hidden, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # 다시 매개변수화: 뽑기를 미분 가능하게
        return self.decode(z), mu, logvar


def losses(x, recon, mu, logvar):
    rec = F.binary_cross_entropy(recon, x, reduction="sum") / len(x)          # 사진 한 장당 재구성 손실
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)    # 사진 한 장당 KL
    return rec, kl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=int, default=2, help="사진 한 장을 담을 숫자 개수(잠재 차원)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="학습 사진 수 줄이기")
    a = ap.parse_args()
    utils.set_seed()
    xtr, ytr, xte, yte = T.load_fashion(a.limit)
    dev = T.device()
    m = VAE(a.z).to(dev)
    print(f"[VAE] 학습 {len(xtr):,}장 · 시험 {len(xte):,}장 · 잠재 숫자 z {a.z}개 · {dev}")
    T.shape_trace(m.enc, xtr[:8].to(dev))
    print(f"  → 평균 μ·흩어짐 logσ² 각 (8, {a.z}) → z (8, {a.z}) → 디코더 → (8, 1, 28, 28)")
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    loader = DataLoader(TensorDataset(xtr), batch_size=128, shuffle=True)
    t0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        m.train()
        tr = tk = n = 0
        for (xb,) in loader:
            xb = xb.to(dev)
            recon, mu, logvar = m(xb)
            rec, kl = losses(xb, recon, mu, logvar)
            loss = rec + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += rec.item() * len(xb)
            tk += kl.item() * len(xb)
            n += len(xb)
        print(f"  에폭 {ep}/{a.epochs}  재구성 {tr / n:.1f}  KL {tk / n:.1f}  ({time.perf_counter() - t0:.1f}초)", flush=True)

    m.eval()
    with torch.no_grad():
        xt = xte.to(dev)
        mu, logvar = m.encode(xt)
        recon = m.decode(mu)
        mae = (recon - xt).abs().mean().item()                         # 픽셀 평균 차이(0~1)
        base = (xtr.mean(0, keepdim=True).to(dev) - xt).abs().mean().item()   # '평균 사진 한 장으로 찍기' 기준선
        per_cls = {T.FASHION_CLASSES[c]: round((recon[yte.to(dev) == c] - xt[yte.to(dev) == c]).abs().mean().item(), 3)
                   for c in range(10)}
    print(f"\n시험 사진 재구성 픽셀 오차 {mae:.3f}(0~1, 기준선 '평균 사진 한 장' {base:.3f})")
    print("종류별 재구성 오차(클수록 흐리게 그림):", ", ".join(f"{k} {v}" for k, v in sorted(per_cls.items(), key=lambda kv: kv[1])))

    figs = draw(m, xte, yte, mu.cpu(), a.z)
    utils.write_json(utils.result_path("nn_vae.json"), {
        "args": vars(a), "test_pixel_mae": round(mae, 4), "baseline_mean_image_mae": round(base, 4),
        "by_class_mae": per_cls, "train_sec": round(time.perf_counter() - t0, 1), "figures": figs, "measured_at": utils.now()})
    print("결과:", utils.result_path("nn_vae.json"))


@torch.no_grad()
def draw(m, xte, yte, mu, z):
    import matplotlib.pyplot as plt
    utils.korean_font()
    dev = T.device()
    out = []
    # 1) 원본과 재구성
    idx = [int((yte == c).nonzero()[0]) for c in range(10)]
    rec = m.decode(mu[idx].to(dev)).cpu()
    fig, axs = plt.subplots(2, 10, figsize=(12, 2.8))
    for j, i in enumerate(idx):
        axs[0, j].imshow(xte[i, 0], cmap="gray_r")
        axs[1, j].imshow(rec[j, 0], cmap="gray_r")
        axs[0, j].set_title(T.FASHION_CLASSES[int(yte[i])], fontsize=9)
    for ax in axs.flat:
        ax.axis("off")
    fig.suptitle(f"위: 원본 · 아래: z {z}개로 압축했다가 다시 그린 사진", fontsize=11)
    out.append(save(fig, "nn_vae_recon.png"))
    # 2) 두 사진 사이 잇기(운동화 → 앵클부츠, 티셔츠 → 바지)
    fig, axs = plt.subplots(2, 9, figsize=(11, 2.8))
    for r, (a_, b_) in enumerate([(7, 9), (0, 1)]):
        za, zb = mu[idx[a_]], mu[idx[b_]]
        for k, t in enumerate(torch.linspace(0, 1, 9)):
            img = m.decode(((1 - t) * za + t * zb).unsqueeze(0).to(dev)).cpu()[0, 0]
            axs[r, k].imshow(img, cmap="gray_r")
            axs[r, k].axis("off")
        axs[r, 0].set_title(T.FASHION_CLASSES[a_], fontsize=9)
        axs[r, 8].set_title(T.FASHION_CLASSES[b_], fontsize=9)
    fig.suptitle("두 사진의 z 사이를 9칸으로 나눠 그림: 중간 모양이 부드럽게 이어짐", fontsize=11)
    out.append(save(fig, "nn_vae_interp.png"))
    # 3) 무작위 z로 그리기
    g = torch.Generator().manual_seed(0)
    imgs = m.decode(torch.randn(32, z, generator=g).to(dev)).cpu()
    fig, axs = plt.subplots(4, 8, figsize=(8, 4.2))
    for k, ax in enumerate(axs.flat):
        ax.imshow(imgs[k, 0], cmap="gray_r")
        ax.axis("off")
    fig.suptitle("무작위 z로 새로 그린 사진: 옷 모양은 나오지만 흐릿함", fontsize=11)
    out.append(save(fig, "nn_vae_sample.png"))
    # 4) 잠재 공간 지도(z=2)
    if z == 2:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sc = ax.scatter(mu[:3000, 0], mu[:3000, 1], c=yte[:3000], cmap="tab10", s=4)
        for c in range(10):
            cx, cy = mu[yte == c].median(0).values.tolist()
            ax.text(cx, cy, T.FASHION_CLASSES[c], fontsize=9, ha="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        ax.set_title("시험 사진 3,000장의 z(정답은 학습에 쓰지 않음): 비슷한 옷끼리 모임", fontsize=10)
        out.append(save(fig, "nn_vae_latent.png"))
    return out


def save(fig, name):
    p = utils.fig_path(name)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("그림:", p)
    return str(p.name)


if __name__ == "__main__":
    main()
