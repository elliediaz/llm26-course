"""1회차 실습 A: Dense(완전연결) 신경망으로 같은 표 데이터를 분류와 회귀로 푼다.

데이터  data/G/churn.csv: 가상 RPG 유저 20,000명의 최근 활동 12개(접속일수·플레이 시간·결제액·길드 여부·문의 수 …)
문제    분류: 다음 14일 이탈(1)·유지(0)            출력층 2칸 + 교차 엔트로피
        회귀: 다음 7일 일평균 플레이 시간(분)      출력층 1칸 + 평균제곱오차
몸통    입력 12 → Linear 64 → ReLU → Linear 32 → ReLU  (두 문제가 똑같이 씀. 다른 것은 출력층과 손실뿐)

  python src/nn_dense.py                         # 분류·회귀 둘 다(CPU 서버 약 20초)
  python src/nn_dense.py --task cls --hidden 128 64 --lr 0.003 --epochs 30
  python src/nn_dense.py --task reg --drop pay_30 guild     # 입력 열을 빼 보기(해보기 A)
  python src/nn_dense.py --task cls --threshold 0.3          # 이탈 판정 기준값을 낮춰 더 많이 찾기(해보기 A)
결과: results/nn_dense.json, results/fig/nn_dense.png
"""
from __future__ import annotations

import argparse
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
import nn_tools as T
import utils

FEATURES = ["days_since_join", "login_days_14", "avg_play_min", "level", "level_up_14", "pay_30", "guild",
            "friends", "tickets_30", "pvp_win", "last_login_gap", "event_join_30"]
LOG_COLS = {"avg_play_min", "pay_30", "friends", "days_since_join"}       # 오른쪽으로 긴 분포는 log로 펴서 넣음


class DenseNet(nn.Module):
    """몸통은 같고 머리(출력층)만 과제에 따라 다르다."""

    def __init__(self, n_in: int, hidden: list[int], n_out: int, dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)          # 분류: 2칸(이탈·유지 점수), 회귀: 1칸(숫자)

    def forward(self, x):
        return self.head(self.body(x))


def load(drop: list[str]):
    with open(config.data_dir("G") / "churn.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    feats = [c for c in FEATURES if c not in drop]
    x = torch.tensor([[float(r[c]) for c in feats] for r in rows])
    for j, c in enumerate(feats):
        if c in LOG_COLS:
            x[:, j] = torch.log1p(x[:, j])
    y_cls = torch.tensor([int(r["churn_14d"]) for r in rows])
    y_reg = torch.tensor([float(r["next7_play_min"]) for r in rows])
    n_tr = int(len(rows) * 0.8)
    mu, sd = x[:n_tr].mean(0), x[:n_tr].std(0) + 1e-6          # 표준화: 열마다 평균 0·표준편차 1
    x = (x - mu) / sd
    return feats, x, y_cls, y_reg, n_tr


def run_cls(x, y, n_tr, a):
    model = DenseNet(x.shape[1], a.hidden, 2, a.dropout)
    print("\n[분류] 모델 모양")
    T.shape_trace(model, x[:a.batch])
    tr = DataLoader(TensorDataset(x[:n_tr], y[:n_tr]), batch_size=a.batch, shuffle=True)
    te = DataLoader(TensorDataset(x[n_tr:], y[n_tr:]), batch_size=1024)

    def scores(p_churn, yy, thr):
        pred = (p_churn >= thr).long()                      # 이탈 확률이 기준값 이상이면 '이탈'로 판정
        tp = ((pred == 1) & (yy == 1)).sum().item()
        prec = tp / max(1, (pred == 1).sum().item())
        rec = tp / max(1, (yy == 1).sum().item())
        return {"acc": round((pred == yy).float().mean().item(), 4), "recall": round(rec, 3), "precision": round(prec, 3)}

    def ev(m):
        out, yy = T.predict(m, te)
        return scores(torch.softmax(out, 1)[:, 1], yy, a.threshold)
    hist = T.fit(model, tr, nn.CrossEntropyLoss(), a.epochs, a.lr, ev, "분류")
    base = round(1 - y[n_tr:].float().mean().item(), 4)
    print(f"  기준선(모두 '유지'로 찍기) 정확도 {base} → 모델 {hist[-1]['acc']} (판정 기준값 {a.threshold})")
    out, yy = T.predict(model, te)
    p_churn = torch.softmax(out, 1)[:, 1]
    print("  시험 유저 5명의 이탈 확률:", [round(p, 3) for p in p_churn[:5].tolist()])
    table = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        r = scores(p_churn, yy, thr)
        table.append({"기준값": thr, "정확도": r["acc"], "재현율": r["recall"], "정밀도": r["precision"],
                      "이탈 판정 수": int((p_churn >= thr).sum())})
    print("  기준값을 바꾸면(같은 모델, 다시 학습하지 않음). 재현율 = 실제 이탈 유저 중 찾아낸 비율, 정밀도 = 이탈로 판정한 유저 중 실제 이탈 비율")
    utils.print_table(table)
    return {"history": hist, "baseline_acc": base, "final": hist[-1], "threshold": a.threshold, "by_threshold": table}


def run_reg(x, y, n_tr, a):
    model = DenseNet(x.shape[1], a.hidden, 1, a.dropout)
    print("\n[회귀] 모델 모양")
    T.shape_trace(model, x[:a.batch])
    mu, sd = y[:n_tr].mean(), y[:n_tr].std()
    yn = (y - mu) / sd                                         # 목표값도 표준화해서 학습, 평가 때 되돌림
    tr = DataLoader(TensorDataset(x[:n_tr], yn[:n_tr]), batch_size=a.batch, shuffle=True)
    te = DataLoader(TensorDataset(x[n_tr:], yn[n_tr:]), batch_size=1024)

    def ev(m):
        out, yy = T.predict(m, te)
        p, t = out.squeeze(1) * sd + mu, yy * sd + mu
        mae = (p - t).abs().mean().item()
        r2 = 1 - ((p - t) ** 2).sum().item() / ((t - t.mean()) ** 2).sum().item()
        return {"mae_min": round(mae, 2), "r2": round(r2, 3)}
    loss = lambda out, yy: nn.functional.mse_loss(out.squeeze(1), yy)  # noqa: E731
    hist = T.fit(model, tr, loss, a.epochs, a.lr, ev, "회귀")
    base = round((y[n_tr:] - y[:n_tr].mean()).abs().mean().item(), 2)
    print(f"  기준선(모두 평균값으로 찍기) 평균 오차 {base}분 → 모델 {hist[-1]['mae_min']}분")
    return {"history": hist, "baseline_mae": base, "final": hist[-1]}


def plot(res):
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(res), figsize=(5.2 * len(res), 3.6))
    axes = axes if len(res) > 1 else [axes]
    for ax, (k, r) in zip(axes, res.items()):
        h = r["history"]
        ax.plot([e["epoch"] for e in h], [e["loss"] for e in h], marker="o", label="학습 손실")
        ax.set_title("분류(교차 엔트로피)" if k == "cls" else "회귀(평균제곱오차, 표준화)")
        ax.set_xlabel("에폭")
        ax2 = ax.twinx()
        key = "acc" if k == "cls" else "mae_min"
        ax2.plot([e["epoch"] for e in h], [e[key] for e in h], color="#C8281E", marker="s", label=key)
        ax2.set_ylabel("시험 정확도" if k == "cls" else "시험 평균 오차(분)")
    fig.tight_layout()
    p = utils.fig_path("nn_dense.png")
    fig.savefig(p, dpi=130)
    print("그림:", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["cls", "reg", "both"], default="both")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--drop", nargs="*", default=[], help="뺄 입력 열 이름")
    ap.add_argument("--threshold", type=float, default=0.5, help="분류: 이탈 확률이 이 값 이상이면 이탈로 판정")
    a = ap.parse_args()
    utils.set_seed()
    feats, x, y_cls, y_reg, n_tr = load(a.drop)
    print(f"데이터: 유저 {len(x):,}명(학습 {n_tr:,} · 시험 {len(x) - n_tr:,}), 입력 열 {len(feats)}개: {', '.join(feats)}")
    res = {}
    if a.task in ("cls", "both"):
        res["cls"] = run_cls(x, y_cls, n_tr, a)
    if a.task in ("reg", "both"):
        res["reg"] = run_reg(x, y_reg, n_tr, a)
    plot(res)
    out = {"args": vars(a), "features": feats, "result": {k: {kk: v for kk, v in r.items() if kk != "history"} | {"history": r["history"]} for k, r in res.items()}}
    print("저장:", utils.write_json(utils.result_path("nn_dense.json"), out))


if __name__ == "__main__":
    main()
