"""1회차 실습 C: 순서가 있는 데이터를 RNN과 LSTM으로 읽는다.

기본 과제(분류)  NSMC 네이버 영화 리뷰(CC0) 3만 건 → 긍정(1)·부정(0)          data/common/nsmc/
  [N,T] 글자 번호 → Embedding [N,T,64] → RNN 또는 LSTM [N,T,128] → 마지막 상태 [N,128] → Linear [N,2]
  RNN은 앞 내용을 상태 하나에 덮어쓰며 읽어 긴 문장 앞부분을 잊기 쉽고, LSTM은 문(gate)으로 남길 것을 고른다
바꿔 풀기(회귀)  --task ccu: 가상 RPG 시간별 동시 접속자 수 → 다음 1시간 접속자 수   data/G/ccu_hourly.csv
  [N,48,4] 지난 48시간(접속자·시각·요일·점검) → LSTM → [N,1] 숫자

  python src/nn_rnn.py                                   # 감성 분류 RNN vs LSTM(CPU 서버 약 3분)
  python src/nn_rnn.py --cell lstm --predict "배우들 연기는 좋았는데 결말이 너무 허무하다"
  python src/nn_rnn.py --task ccu                        # 같은 LSTM으로 회귀 문제(해보기 C)
결과: results/nn_rnn.json(또는 nn_rnn_ccu.json), results/fig/nn_rnn_len.png, runs/nn_rnn_<cell>.pt
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

BUCKETS = [(1, 20), (21, 40), (41, 80), (81, 999)]


class SeqClassifier(nn.Module):
    def __init__(self, vocab: int, cell: str, emb: int = 64, hidden: int = 128, n_out: int = 2):
        super().__init__()
        self.cell = cell
        self.embed = nn.Embedding(vocab, emb, padding_idx=0)
        rnn = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, ids, lens):
        h = self.rnn(self.embed(ids))[0]                       # [N,T,H]: 모든 시점의 상태
        last = h[torch.arange(len(ids)), (lens - 1).clamp(min=0)]  # [N,H]: 문장 끝(패딩 전) 시점의 상태
        return self.head(last)


class SeqRegressor(nn.Module):
    def __init__(self, n_feat: int, cell: str, hidden: int = 64):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).squeeze(1)    # 마지막 시점 상태 → 숫자 1개


# ---------------------------------------------------------------- 감성 분류
def sentiment(a):
    tr, te = T.load_nsmc(a.limit)
    vocab = T.CharVocab([t for t, _ in tr])
    xtr, ltr = vocab.encode([t for t, _ in tr], a.max_len)
    xte, lte = vocab.encode([t for t, _ in te], a.max_len)
    ytr, yte = torch.tensor([y for _, y in tr]), torch.tensor([y for _, y in te])
    print(f"데이터: 학습 {len(tr):,} · 시험 {len(te):,}건, 글자 사전 {len(vocab):,}개, 최대 길이 {a.max_len}자")
    print(f"예: {tr[0][0]!r} → 번호 {xtr[0, :8].tolist()} … (라벨 {tr[0][1]})")
    trl = DataLoader(TensorDataset(xtr, ltr, ytr), batch_size=a.batch, shuffle=True)
    tel = DataLoader(TensorDataset(xte, lte, yte), batch_size=1000)
    res = {}
    for cell in (["rnn", "lstm"] if a.cell == "both" else [a.cell]):
        utils.set_seed()
        model = SeqClassifier(len(vocab), cell, hidden=a.hidden)
        print(f"\n[{cell.upper()}] 모델 모양")
        T.shape_trace(model, xtr[:a.batch], ltr[:a.batch])

        def ev(m):
            out, yy = T.predict(m, tel)
            return {"acc": round((out.argmax(1) == yy).float().mean().item(), 4)}
        hist = T.fit(model, trl, nn.CrossEntropyLoss(), a.epochs, a.lr, ev, cell.upper())
        out, yy = T.predict(model, tel)
        ok = out.argmax(1) == yy
        by_len = {}
        for lo, hi in BUCKETS:
            m = (lte >= lo) & (lte <= hi)
            by_len[f"{lo}~{hi if hi < 999 else ''}자"] = round(ok[m].float().mean().item(), 3) if m.any() else None
        print("  길이별 정확도:", by_len)
        res[cell] = {"params": sum(p.numel() for p in model.parameters()), "final_acc": hist[-1]["acc"],
                     "train_sec": hist[-1]["sec"], "by_length": by_len, "history": hist}
        config.PATHS["runs"].mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "itos": vocab.itos, "cell": cell, "hidden": a.hidden, "max_len": a.max_len},
                   config.PATHS["runs"] / f"nn_rnn_{cell}.pt")
    plot_len(res)
    if len(res) == 2:
        utils.print_table([{"구조": k.upper(), "파라미터": r["params"], "시험 정확도": r["final_acc"],
                            **{f"{b}": v for b, v in r["by_length"].items()}, "학습 초": r["train_sec"]} for k, r in res.items()])
    print("저장:", utils.write_json(utils.result_path("nn_rnn.json"), {"args": vars(a), "result": res}))


def plot_len(res):
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    keys = list(next(iter(res.values()))["by_length"])
    w = 0.38
    for i, (cell, r) in enumerate(res.items()):
        ax.bar([k + (i - 0.5) * w for k in range(len(keys))], [r["by_length"][b] or 0 for b in keys], w, label=cell.upper())
    ax.set_xticks(range(len(keys)), keys)
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("시험 정확도")
    ax.set_title("리뷰 길이별 감성 분류 정확도")
    ax.legend()
    fig.tight_layout()
    p = utils.fig_path("nn_rnn_len.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("그림:", p)


@torch.no_grad()
def predict_text(a):
    path = config.PATHS["runs"] / f"nn_rnn_{a.cell if a.cell != 'both' else 'lstm'}.pt"
    if not path.exists():
        raise SystemExit(f"학습한 모델이 없음: {path}\n  먼저 python src/nn_rnn.py로 학습하거나(약 1분), 급하면 bash ref/s01/use_ref.sh")
    ck = torch.load(path, weights_only=False)
    vocab = T.CharVocab([])
    vocab.itos = ck["itos"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    model = SeqClassifier(len(vocab), ck["cell"], hidden=ck["hidden"])
    model.load_state_dict(ck["state"])
    model.eval()
    for text in a.predict:
        ids, lens = vocab.encode([text], ck["max_len"])
        p = torch.softmax(model(ids, lens), 1)[0]
        unk = sum(1 for c in text if c not in vocab.stoi)
        print(f"[{ck['cell'].upper()}] {text!r} → 긍정 {p[1]:.3f} · 부정 {p[0]:.3f}" + (f" (처음 보는 글자 {unk}개)" if unk else ""))


# ---------------------------------------------------------------- 동시 접속자 회귀
def ccu(a):
    with open(config.data_dir("G") / "ccu_hourly.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    v = torch.tensor([float(r["ccu"]) for r in rows])
    scale = v.max().item()
    feats = torch.stack([v / scale,
                         torch.tensor([float(r["hour"]) for r in rows]) / 23,
                         torch.tensor([float(r["weekday"]) for r in rows]) / 6,
                         torch.tensor([float(r["maintenance"]) for r in rows])], 1)        # [8760,4]
    W = a.window
    X = torch.stack([feats[i - W:i] for i in range(W, len(v))])                            # [M,48,4]
    Y = v[W:] / scale                                                                       # [M]
    cut = int(len(X) * 0.85)                                                                # 앞 85% 학습, 뒤(11~12월) 시험
    print(f"데이터: 시간별 동시 접속자 {len(v):,}시간, 입력 창 {W}시간 × 특징 4개 → 다음 1시간 접속자")
    trl = DataLoader(TensorDataset(X[:cut], Y[:cut]), batch_size=a.batch, shuffle=True)
    tel = DataLoader(TensorDataset(X[cut:], Y[cut:]), batch_size=1000)
    naive = (v[cut + W - 24:len(v) - 24] - v[cut + W:]).abs().mean().item()             # 어제 같은 시각 값으로 찍기
    res = {}
    for cell in (["rnn", "lstm"] if a.cell == "both" else [a.cell]):
        utils.set_seed()
        model = SeqRegressor(4, cell, hidden=a.hidden)
        print(f"\n[{cell.upper()}] 모델 모양")
        T.shape_trace(model, X[:a.batch])

        def ev(m):
            out, yy = T.predict(m, tel)
            return {"mae": round(((out - yy).abs().mean() * scale).item(), 0)}
        hist = T.fit(model, trl, nn.MSELoss(), a.epochs, a.lr, ev, cell.upper())
        res[cell] = {"final_mae": hist[-1]["mae"], "history": hist}
    print(f"\n기준선(어제 같은 시각 값) 평균 오차 {naive:,.0f}명 → " + ", ".join(f"{k.upper()} {r['final_mae']:,.0f}명" for k, r in res.items()))
    print("저장:", utils.write_json(utils.result_path("nn_rnn_ccu.json"), {"args": vars(a), "naive_mae": round(naive), "result": res}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["sentiment", "ccu"], default="sentiment")
    ap.add_argument("--cell", choices=["rnn", "lstm", "both"], default="both")
    ap.add_argument("--hidden", type=int, help="상태 크기(기본: 감성 128, 접속자 64)")
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--window", type=int, default=48, help="ccu: 입력으로 쓸 지난 시간 수")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, help="학습 리뷰 앞에서 N건만(빠른 확인용)")
    ap.add_argument("--predict", nargs="+", help="학습해 둔 모델로 내 문장 감성 예측")
    a = ap.parse_args()
    utils.set_seed()
    if a.predict:
        predict_text(a)
    elif a.task == "ccu":
        if a.epochs == 2:
            a.epochs = 5
        a.hidden = a.hidden or 64
        ccu(a)
    else:
        a.hidden = a.hidden or 128
        sentiment(a)


if __name__ == "__main__":
    main()
