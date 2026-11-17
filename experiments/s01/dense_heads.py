"""1회차 실습 A 보조: 몸통은 같고 출력층만 다름. 같은 유저 데이터로 분류(이탈)와 회귀(다음 주 플레이 시간)를 같이 학습

  python experiments/s01/dense_heads.py                         # 은닉층 64·32, ReLU, 5에폭
  python experiments/s01/dense_heads.py --act none              # 활성화를 빼면(층을 쌓아도 선형)
  python experiments/s01/dense_heads.py --hidden 8              # 몸통을 아주 좁게
  python experiments/s01/dense_heads.py --hidden 256 128 64     # 몸통을 넓고 깊게
  python experiments/s01/dense_heads.py --epochs 1 --lr 0.1     # 학습률을 크게

볼 것
  ① 층별 출력 모양: 마지막 줄만 다름(분류 [N,2], 회귀 [N,1])
  ② 손실 함수: 분류는 교차 엔트로피, 회귀는 평균제곱오차
  ③ 몸통·활성화·학습률을 바꿀 때 이탈 정확도와 플레이 시간 MAE(분)가 같이 움직이는지
메모: 이탈 비율이 약 28%라 전부 '유지'로 찍어도 정확도 0.72. 그보다 높아야 의미가 있음
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import nn_tools as T  # noqa: E402

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def body(n_in: int, hidden: list[int], act: str) -> list[nn.Module]:
    layers, d = [], n_in
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if act != "none":
            layers.append(ACTS[act]())
        d = h
    return layers


def load(limit: int):
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "G" / "churn.csv").sample(frac=1, random_state=0).head(limit)
    X = df.drop(columns=["user_id", "churn_14d", "next7_play_min"]).to_numpy(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return torch.tensor(X), torch.tensor(df["churn_14d"].to_numpy()), torch.tensor(df["next7_play_min"].to_numpy(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--act", choices=list(ACTS) + ["none"], default="relu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=20000)
    a = ap.parse_args()
    torch.manual_seed(0)

    X, yc, yr = load(a.limit)
    n_tr = int(len(X) * 0.8)
    cls = nn.Sequential(*body(X.shape[1], a.hidden, a.act), nn.Linear(a.hidden[-1], 2))
    reg = nn.Sequential(*body(X.shape[1], a.hidden, a.act), nn.Linear(a.hidden[-1], 1))
    print("[분류 모델]")
    T.shape_trace(cls, X[:8])
    print("[회귀 모델]")
    T.shape_trace(reg, X[:8])

    for name, model, y, loss_fn in (("분류", cls, yc, nn.CrossEntropyLoss()), ("회귀", reg, yr, nn.MSELoss())):
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        for ep in range(a.epochs):
            perm = torch.randperm(n_tr)
            for i in range(0, n_tr, 256):
                b = perm[i:i + 256]
                out = model(X[b])
                loss = loss_fn(out, y[b]) if name == "분류" else loss_fn(out.squeeze(1), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            out = model(X[n_tr:])
            if name == "분류":
                acc = (out.argmax(1) == yc[n_tr:]).float().mean().item()
                p = torch.softmax(out[:1], 1)[0]
                print(f"\n[분류] 검증 정확도 {acc:.3f} (전부 유지로 찍으면 {1 - yc[n_tr:].float().mean().item():.3f}) · 첫 유저 [유지, 이탈] 확률 {[round(v, 3) for v in p.tolist()]}")
            else:
                mae = (out.squeeze(1) - yr[n_tr:]).abs().mean().item()
                base = (yr[n_tr:] - yr[:n_tr].mean()).abs().mean().item()
                print(f"[회귀] 검증 MAE {mae:.1f}분 (평균값으로 찍으면 {base:.1f}분)")
    n = sum(p.numel() for p in cls.parameters())
    print(f"\n설정 : 은닉 {a.hidden} · 활성화 {a.act} · 에폭 {a.epochs} · 학습률 {a.lr} · 분류 모델 파라미터 {n:,}개")


if __name__ == "__main__":
    main()
