"""1회차 기본 신경망 실습(Dense·CNN·RNN·LSTM) 공통 도구.

  shape_trace(model, x)   층마다 출력 모양·파라미터 수를 표로 찍는다(데이터가 층을 지나며 어떻게 바뀌는지)
  fit(...)                 분류·회귀 공통 학습 루프. 손실 함수만 바꿔 끼운다
  load_fashion()           Fashion-MNIST(IDX gz) → 텐서
  CharVocab                글자 단위 토크나이저(NSMC 리뷰용)
"""
from __future__ import annotations

import gzip
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config

DATA = config.PATHS["data"]


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 모양 추적
def shape_trace(model: nn.Module, *inputs) -> list[dict]:
    """forward를 한 번 돌리며 맨 아래 층(Linear·Conv2d·ReLU·LSTM …)마다 출력 모양을 기록해 표로 출력한다."""
    rows, hooks = [], []

    def hook(name, mod):
        def f(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            shape = tuple(o.shape) if hasattr(o, "shape") else "-"
            rows.append({"층": name, "종류": mod.__class__.__name__, "출력 모양": str(shape).replace(" ", ""),
                         "파라미터": sum(p.numel() for p in mod.parameters())})
        return f

    for n, m in model.named_modules():
        if n and not list(m.children()):
            hooks.append(m.register_forward_hook(hook(n, m)))
    model.eval()
    with torch.no_grad():
        model(*inputs)
    for h in hooks:
        h.remove()
    first = inputs[0]
    print(f"입력 모양: {tuple(first.shape)}  (N = 묶음 크기)")
    w = max(len(r["층"]) for r in rows) + 2
    for r in rows:
        print(f"  {r['층']:<{w}}{r['종류']:<12}{r['출력 모양']:<22}{r['파라미터']:>10,}")
    print(f"  전체 파라미터 {sum(p.numel() for p in model.parameters()):,}")
    return rows


# ---------------------------------------------------------------- 학습 루프
def fit(model, train_loader, loss_fn, epochs: int, lr: float, eval_fn=None, label: str = "", log_every: int = 0):
    """분류든 회귀든 같은 루프: 순전파 → 손실 → 역전파 → 갱신. 다른 것은 loss_fn 하나뿐."""
    dev = device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, t0, step = [], time.perf_counter(), 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in train_loader:
            *xs, y = [b.to(dev) for b in batch]
            out = model(*xs)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            step += 1
            if log_every and step % log_every == 0:
                print(f"    [{label}] 스텝 {step} 손실 {loss.item():.4f}", flush=True)
        rec = {"epoch": ep, "loss": round(tot / n, 4), "sec": round(time.perf_counter() - t0, 1)}
        if eval_fn:
            rec.update(eval_fn(model))
        hist.append(rec)
        if ep == 1 or ep == epochs or ep % 5 == 0 or epochs <= 5:
            print(f"  [{label}] 에폭 {ep}/{epochs}  " + "  ".join(f"{k} {v}" for k, v in rec.items() if k != "epoch"), flush=True)
    return hist


@torch.no_grad()
def predict(model, loader):
    dev = device()
    model.eval()
    outs, ys = [], []
    for batch in loader:
        *xs, y = [b.to(dev) for b in batch]
        outs.append(model(*xs).float().cpu())
        ys.append(y.cpu())
    return torch.cat(outs), torch.cat(ys)


# ---------------------------------------------------------------- 데이터
FASHION_CLASSES = ["티셔츠", "바지", "스웨터", "드레스", "코트", "샌들", "셔츠", "운동화", "가방", "앵클부츠"]


def _idx(path: Path, offset: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=offset)


def load_fashion(limit: int | None = None):
    """Fashion-MNIST(학습 6만·시험 1만, 28×28 흑백). 반환: (x_train, y_train, x_test, y_test), x는 [N,1,28,28] 0~1"""
    d = DATA / "common" / "fashion_mnist"
    xtr = _idx(d / "train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
    ytr = _idx(d / "train-labels-idx1-ubyte.gz", 8)
    xte = _idx(d / "t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)
    yte = _idx(d / "t10k-labels-idx1-ubyte.gz", 8)
    if limit:
        xtr, ytr = xtr[:limit], ytr[:limit]
        xte, yte = xte[: max(1000, limit // 6)], yte[: max(1000, limit // 6)]
    f = lambda a: torch.tensor(a, dtype=torch.float32) / 255.0  # noqa: E731
    return f(xtr), torch.tensor(ytr, dtype=torch.long), f(xte), torch.tensor(yte, dtype=torch.long)


def load_nsmc(limit: int | None = None):
    """NSMC 부분 집합(학습 3만·시험 1만, 긍정 1·부정 0). 반환: (train_rows, test_rows), 행은 (문장, 라벨)"""
    d = DATA / "common" / "nsmc"

    def read(name):
        rows = []
        with open(d / name, encoding="utf-8") as f:
            next(f)
            for line in f:
                text, label = line.rstrip("\n").rsplit("\t", 1)
                rows.append((text, int(label)))
        return rows
    tr, te = read("train.tsv"), read("test.tsv")
    if limit:
        tr, te = tr[:limit], te[: max(1000, limit // 3)]
    return tr, te


class CharVocab:
    """글자 하나를 토큰 하나로 본다. 자주 나온 글자 max_size 개 + <pad>(0)·<unk>(1)."""

    def __init__(self, texts, max_size: int = 2000):
        cnt = Counter(ch for t in texts for ch in t)
        self.itos = ["<pad>", "<unk>"] + [c for c, _ in cnt.most_common(max_size)]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, texts, max_len: int):
        ids = torch.zeros(len(texts), max_len, dtype=torch.long)
        lens = torch.zeros(len(texts), dtype=torch.long)
        for i, t in enumerate(texts):
            t = t[:max_len] or " "
            ids[i, : len(t)] = torch.tensor([self.stoi.get(c, 1) for c in t])
            lens[i] = len(t)
        return ids, lens
