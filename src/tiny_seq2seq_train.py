"""1회차 실습 1 ③에서 실행, 실습 3 ⑦에서 해석: 소규모 Encoder-Decoder를 직접 학습한다(과제: 한글 수사 → 숫자 표기).

같은 데이터·같은 학습 시간으로 세 구조를 비교해 '고정 길이 벡터 병목 → 어텐션 → 트랜스포머' 흐름을 확인한다.
  gru        인코더 마지막 은닉 상태(고정 길이 벡터 c) 하나만 디코더에 넘김   ← Cho·Sutskever(2014) 방식
  gru_attn   디코더가 매 시점 인코더 전체 상태를 가중합으로 참조(Bahdanau 어텐션)
  transformer RNN 없이 어텐션만(인코더 self, 디코더 masked self + cross)    ← Vaswani(2017)

  python src/tiny_seq2seq_train.py --arch transformer            # 기본: 2분 학습
  python src/tiny_seq2seq_train.py --arch gru --minutes 2
  python src/tiny_seq2seq_train.py --arch all --minutes 2        # 세 구조 연속 학습·비교표
  python src/tiny_seq2seq_train.py --arch gru --hidden 32        # 병목을 더 좁혀 보기

결과
  runs/numerals_<arch>.pt                       체크포인트(viz_attention.py가 읽음)
  results/numerals_<arch>.json                  Loss 기록, 입력 길이 구간별 정확도, 초당 학습 문장 수
  results/fig/numerals_loss.png, numerals_by_length.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import utils

sys.path.insert(0, str(config.PATHS["common"] / "numerals"))
import make_numerals  # noqa: E402

PAD, BOS, EOS, UNK = 0, 1, 2, 3
BUCKETS = [(0, 8), (9, 12), (13, 16), (17, 99)]


# ================================================================ 데이터
class Vocab:
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, s, bos=False, eos=False):
        ids = [self.stoi.get(c, UNK) for c in s]
        return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i > UNK:
                out.append(self.itos[i])
        return "".join(out)


def load_rows(name):
    return [json.loads(x) for x in open(config.PATHS["common"] / "numerals" / name, encoding="utf-8")]


def batchify(rows, vocab, device):
    src = [vocab.encode(r["src"]) for r in rows]
    tgt = [vocab.encode(r["tgt"], bos=True, eos=True) for r in rows]
    S, T = max(map(len, src)), max(map(len, tgt))
    s = torch.full((len(rows), S), PAD, dtype=torch.long)
    t = torch.full((len(rows), T), PAD, dtype=torch.long)
    for i, (a, b) in enumerate(zip(src, tgt)):
        s[i, :len(a)] = torch.tensor(a)
        t[i, :len(b)] = torch.tensor(b)
    return s.to(device), t.to(device)


# ================================================================ 모델 ① GRU (고정 길이 벡터)·② GRU+어텐션
class GRUSeq2Seq(nn.Module):
    def __init__(self, V, emb=64, hidden=128, attention=False):
        super().__init__()
        self.attention = attention
        self.emb_src = nn.Embedding(V, emb, padding_idx=PAD)
        self.emb_tgt = nn.Embedding(V, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hidden, batch_first=True)
        self.decoder = nn.GRUCell(emb + hidden, hidden)
        if attention:                       # Bahdanau(가산) 어텐션
            self.W_h = nn.Linear(hidden, hidden, bias=False)
            self.W_s = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, 1, bias=False)
        self.out = nn.Linear(hidden * 2, V)
        self.last_attn = None

    def encode(self, src):
        H, h = self.encoder(self.emb_src(src))            # H: (B,S,h) 모든 시점, h: 마지막 시점
        return H, h[0]

    def step(self, y_prev, s, H, src_mask):
        if self.attention:
            e = self.v(torch.tanh(self.W_h(H) + self.W_s(s)[:, None])).squeeze(-1)   # (B,S)
            e = e.masked_fill(~src_mask, -1e9)
            a = torch.softmax(e, -1)
            c = (a[:, :, None] * H).sum(1)                 # 매 시점 새로 만드는 문맥 벡터
        else:
            a, c = None, self.c                            # 처음 만든 고정 길이 벡터 그대로
        s = self.decoder(torch.cat([self.emb_tgt(y_prev), c], -1), s)
        return self.out(torch.cat([s, c], -1)), s, a

    def forward(self, src, tgt_in):
        H, h = self.encode(src)
        self.c, s = h, h
        mask = src != PAD
        logits, attn = [], []
        for t in range(tgt_in.size(1)):                    # RNN은 한 시점씩 차례로 계산(병렬화 불가)
            o, s, a = self.step(tgt_in[:, t], s, H, mask)
            logits.append(o)
            attn.append(a)
        self.last_attn = torch.stack(attn, 1) if self.attention else None
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy(self, src, max_len=16):
        H, h = self.encode(src)
        self.c, s = h, h
        mask = src != PAD
        y = torch.full((src.size(0),), BOS, dtype=torch.long, device=src.device)
        outs, attn = [], []
        for _ in range(max_len):
            o, s, a = self.step(y, s, H, mask)
            y = o.argmax(-1)
            outs.append(y)
            attn.append(a)
        self.last_attn = torch.stack(attn, 1) if self.attention else None
        return torch.stack(outs, 1)


# ================================================================ 모델 ③ 트랜스포머
class PosEnc(nn.Module):
    def __init__(self, d, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class DecoderLayer(nn.Module):
    """masked self-attention → cross-attention → FFN. cross-attention 가중치를 저장해 시각화에 쓴다."""

    def __init__(self, d, nhead, ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross_w = None

    def forward(self, x, mem, causal, mem_pad):
        x = self.n1(x + self.self_attn(x, x, x, attn_mask=causal, need_weights=False)[0])
        y, w = self.cross_attn(x, mem, mem, key_padding_mask=mem_pad, need_weights=True, average_attn_weights=False)
        self.cross_w = w                                    # (B, heads, T, S)
        x = self.n2(x + y)
        return self.n3(x + self.ff(x))


class TransformerSeq2Seq(nn.Module):
    def __init__(self, V, d=128, nhead=4, layers=2, ff=256):
        super().__init__()
        self.emb_src, self.emb_tgt = nn.Embedding(V, d, padding_idx=PAD), nn.Embedding(V, d, padding_idx=PAD)
        self.pos = PosEnc(d)
        enc_layer = nn.TransformerEncoderLayer(d, nhead, ff, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.dec = nn.ModuleList(DecoderLayer(d, nhead, ff) for _ in range(layers))
        self.out = nn.Linear(d, V)
        self.scale = math.sqrt(d)
        self.last_attn = None

    def forward(self, src, tgt_in):
        pad = src == PAD
        mem = self.encoder(self.pos(self.emb_src(src) * self.scale), src_key_padding_mask=pad)
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=src.device), 1)
        x = self.pos(self.emb_tgt(tgt_in) * self.scale)
        for layer in self.dec:                              # 모든 출력 위치를 한 번에 계산(병렬)
            x = layer(x, mem, causal, pad)
        self.last_attn = self.dec[-1].cross_w.mean(1)       # 마지막 층, 헤드 평균 (B,T,S)
        return self.out(x)

    @torch.no_grad()
    def greedy(self, src, max_len=16):
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            nxt = self.forward(src, ys)[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], 1)
        return ys[:, 1:]


def build(arch, V, hidden):
    if arch == "gru":
        return GRUSeq2Seq(V, hidden=hidden, attention=False)
    if arch == "gru_attn":
        return GRUSeq2Seq(V, hidden=hidden, attention=True)
    return TransformerSeq2Seq(V)


# ================================================================ 학습·평가
def evaluate(model, rows, vocab, device, bs=256):
    model.eval()
    hits, by_len = 0, {f"{a}~{b if b < 99 else ''}": [0, 0] for a, b in BUCKETS}
    samples = []
    for i in range(0, len(rows), bs):
        part = rows[i:i + bs]
        src, _ = batchify(part, vocab, device)
        pred = model.greedy(src)
        for r, p in zip(part, pred.tolist()):
            out = vocab.decode(p)
            ok = out == r["tgt"]
            hits += ok
            n = len(r["src"])
            key = next(f"{a}~{b if b < 99 else ''}" for a, b in BUCKETS if a <= n <= b)
            by_len[key][0] += ok
            by_len[key][1] += 1
            if len(samples) < 5 and not ok:
                samples.append({"src": r["src"], "gold": r["tgt"], "pred": out})
    model.train()
    return {"acc": round(hits / len(rows), 4),
            "by_len": {k: (round(v[0] / v[1], 4) if v[1] else None) for k, v in by_len.items()},
            "errors": samples}


def train(arch, minutes, hidden, device, bs=128, seed=config.SEED) -> dict:
    utils.set_seed(seed)
    make_numerals.ensure()
    tr, va, lo = load_rows("train.jsonl"), load_rows("valid.jsonl"), load_rows("long.jsonl")
    vocab = Vocab([r["src"] + r["tgt"] for r in tr])
    model = build(arch, len(vocab.itos), hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    warm = 200 if arch == "transformer" else 1          # 트랜스포머는 초반 학습률을 천천히 올린다(warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    print(f"\n[{arch}] 파라미터 {n_params:,} · 장치 {device} · 학습 {minutes}분")
    log, step, seen, t0 = [], 0, 0, time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    model.train()
    while time.perf_counter() - t0 < minutes * 60:
        for idx in torch.randperm(len(tr), generator=g).split(bs):
            src, tgt = batchify([tr[i] for i in idx], vocab, device)
            logits = model(src, tgt[:, :-1])                        # 교사 강제: 정답 앞 토큰을 입력
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += len(idx)
            if step % 50 == 0:
                el = time.perf_counter() - t0
                log.append({"step": step, "sec": round(el, 1), "loss": round(loss.item(), 4)})
                if step % 200 == 0:
                    print(f"  step {step:5d}  {el:6.1f}s  loss {loss.item():.4f}")
            if time.perf_counter() - t0 >= minutes * 60:
                break
    sec = time.perf_counter() - t0
    ev, evl = evaluate(model, va, vocab, device), evaluate(model, lo, vocab, device)
    res = {"arch": arch, "params": n_params, "hidden": hidden, "minutes": minutes, "steps": step,
           "sent_per_sec": round(seen / sec, 1), "final_loss": log[-1]["loss"] if log else None,
           "valid_acc": ev["acc"], "valid_by_len": ev["by_len"], "long_acc": evl["acc"],
           "errors": ev["errors"] + evl["errors"][:3], "loss_log": log, "device": device}
    print(f"  → 검증 정확도 {ev['acc']:.3f} · 긴 입력(16자+) {evl['acc']:.3f} · 길이별 {ev['by_len']} · 초당 {res['sent_per_sec']}문장")
    torch.save({"arch": arch, "hidden": hidden, "state": model.state_dict(), "itos": vocab.itos},
               config.PATHS["runs"] / f"numerals_{arch}.pt")
    utils.write_json(utils.result_path(f"numerals_{arch}.json"), res)
    return res


def load_trained(arch, device="cpu"):
    ck = torch.load(config.PATHS["runs"] / f"numerals_{arch}.pt", map_location=device)
    vocab = Vocab([])
    vocab.itos = ck["itos"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    model = build(arch, len(vocab.itos), ck["hidden"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, vocab


def plot(results: list[dict]) -> None:
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for r in results:
        ax.plot([x["sec"] for x in r["loss_log"]], [x["loss"] for x in r["loss_log"]], label=r["arch"])
    ax.set_xlabel("학습 시간(초)")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("같은 시간 동안의 Loss 하락")
    fig.tight_layout()
    fig.savefig(utils.fig_path("numerals_loss.png"), dpi=150)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    keys = list(results[0]["valid_by_len"])
    w = 0.8 / len(results)
    for j, r in enumerate(results):
        ax.bar([i + j * w for i in range(len(keys))], [r["valid_by_len"][k] or 0 for k in keys], w, label=r["arch"])
    ax.set_xticks([i + w * (len(results) - 1) / 2 for i in range(len(keys))], [f"{k}자" for k in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("정확도(완전 일치)")
    ax.set_title("입력 길이 구간별 정확도")
    ax.legend()
    fig.tight_layout()
    fig.savefig(utils.fig_path("numerals_by_length.png"), dpi=150)
    print(f"그림: {utils.fig_path('numerals_loss.png')}, {utils.fig_path('numerals_by_length.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["gru", "gru_attn", "transformer", "all"], default="transformer")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--hidden", type=int, default=128, help="GRU 은닉 크기(= 고정 길이 벡터 c의 차원)")
    a = ap.parse_args()
    config.PATHS["runs"].mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    archs = ["gru", "gru_attn", "transformer"] if a.arch == "all" else [a.arch]
    results = [train(x, a.minutes, a.hidden, device) for x in archs]
    if len(results) > 1:
        print()
        utils.print_table([{"구조": r["arch"], "파라미터": r["params"], "검증 정확도": r["valid_acc"],
                            "긴 입력 정확도": r["long_acc"], "초당 문장": r["sent_per_sec"], "최종 Loss": r["final_loss"]}
                           for r in results])
    else:  # 이전에 학습한 다른 구조 결과가 있으면 함께 그린다
        for x in ("gru", "gru_attn", "transformer"):
            p = config.PATHS["results"] / f"numerals_{x}.json"
            if x != a.arch and p.exists():
                results.append(utils.read_json(p))
    order = ["gru", "gru_attn", "transformer"]
    plot(sorted(results, key=lambda r: order.index(r["arch"])))


if __name__ == "__main__":
    main()
