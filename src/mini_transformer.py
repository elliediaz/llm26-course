"""2회차: 미니 Decoder-only Transformer: LLM과 같은 부품을 100줄 남짓으로 직접 조립해 본다.

  python src/mini_transformer.py --info            # 층 구성·파라미터 수(실제 LLM 설정과 나란히 비교)
  python src/mini_transformer.py --train 1.5       # 도메인 문서로 1.5분 글자 단위 학습 → 생성 예시
  python src/mini_transformer.py --kv              # KV Cache 유무로 같은 문장 생성: 결과는 같고 계산량·시간이 다르다

부품: 토큰 임베딩 + 위치 임베딩 → [LayerNorm → Masked Self-Attention → 잔차 → LayerNorm → MLP → 잔차] × N → 출력층
Self-Attention은 Q·K·V를 직접 계산하고 causal mask(미래 토큰 가리기)를 눈으로 확인할 수 있게 썼다.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import utils


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head, self.d_head = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.last_weights = None

    def forward(self, x, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))  # (B,H,T,dh)
        if past is not None:                          # KV Cache: 앞 토큰의 K·V를 다시 계산하지 않고 이어 붙임
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        S = k.size(2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)          # (B,H,T,S): 모든 쌍의 유사도
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)   # 자기와 앞쪽만 허용
        scores = scores.masked_fill(~mask, float("-inf"))
        w = scores.softmax(-1)
        self.last_weights = w.detach()
        y = (w @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y), (k, v)


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, past=None):
        a, kv = self.attn(self.ln1(x), past)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv


class MiniGPT(nn.Module):
    def __init__(self, vocab, d=128, n_layer=4, n_head=4, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight             # 입력·출력 임베딩 공유(많은 LLM이 쓰는 방식)
        self.max_len = max_len

    def forward(self, idx, past=None):
        start = 0 if past is None else past[0][0].size(2)
        x = self.tok(idx) + self.pos(torch.arange(start, start + idx.size(1), device=idx.device))
        new_past = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, None if past is None else past[i])
            new_past.append(kv)
        return self.head(self.ln_f(x)), new_past

    @torch.no_grad()
    def generate(self, idx, n, use_cache=True):
        past, out, computed = None, idx, 0
        for _ in range(n):
            inp = out if (past is None or not use_cache) else out[:, -1:]
            logits, new_past = self.forward(inp[:, -self.max_len:], past if use_cache else None)
            computed += inp.size(1)
            past = new_past if use_cache else None
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
        return out, computed


def corpus_text() -> str:
    d = config.data_dir()
    parts = [p.read_text(encoding="utf-8") for p in sorted((d / "corpus").glob("*.md"))]
    return "\n".join(parts)


def info():
    m = MiniGPT(vocab=2000)
    rows = [{"부품": n, "파라미터": sum(p.numel() for p in mod.parameters())}
            for n, mod in [("토큰 임베딩", m.tok), ("위치 임베딩", m.pos)] + [(f"블록 {i}", b) for i, b in enumerate(m.blocks)]]
    rows.append({"부품": "합계(출력층은 임베딩과 공유)", "파라미터": sum(p.numel() for p in m.parameters())})
    utils.print_table(rows)
    print("\n같은 구조, 다른 규모(공개 설정값 기준)")
    utils.print_table([
        {"모델": "MiniGPT(오늘)", "층": 4, "d_model": 128, "헤드": 4, "어휘": "글자 수천", "파라미터": "약 100만"},
        {"모델": "Qwen3-0.6B", "층": 28, "d_model": 1024, "헤드": "Q16/KV8", "어휘": "약 15만", "파라미터": "약 6억"},
        {"모델": "Qwen3-8B", "층": 36, "d_model": 4096, "헤드": "Q32/KV8", "어휘": "약 15만", "파라미터": "약 80억"},
    ])
    print("\n※ 실제 값은 python -c \"from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3-0.6B'))\"로 확인")


def train(minutes):
    text = corpus_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_seed()
    m = MiniGPT(len(chars)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    T, B, t0, step = 128, 64, time.perf_counter(), 0
    print(f"글자 {len(text):,}개 · 어휘 {len(chars)} · 파라미터 {sum(p.numel() for p in m.parameters()):,} · {dev}")
    while time.perf_counter() - t0 < minutes * 60:
        ix = torch.randint(len(data) - T - 1, (B,))
        x = torch.stack([data[i:i + T] for i in ix]).to(dev)
        y = torch.stack([data[i + 1:i + T + 1] for i in ix]).to(dev)
        logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))   # 다음 글자 예측
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"  step {step:4d}  loss {loss.item():.3f}")
    torch.save({"state": m.state_dict(), "chars": chars}, config.PATHS["runs"] / "mini_gpt.pt")
    for prompt in ("서리칼날", "운영정책 POL-0"):
        idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=dev)
        out, _ = m.generate(idx, 60)
        print(f"\n[생성] {prompt} → {''.join(chars[i] for i in out[0].tolist())}")
    return m


def kv_demo():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = config.PATHS["runs"] / "mini_gpt.pt"
    if p.exists():
        ck = torch.load(p, map_location=dev)
        m = MiniGPT(len(ck["chars"])).to(dev)
        m.load_state_dict(ck["state"])
    else:
        m = MiniGPT(2000).to(dev)
    m.eval()
    idx = torch.randint(0, 100, (1, 32), device=dev)
    rows = []
    for use in (False, True):
        utils.vram_reset()
        with utils.Timer() as t:
            out, computed = m.generate(idx, 200, use_cache=use)
        rows.append({"KV Cache": "사용" if use else "미사용", "생성 토큰": 200, "모델에 넣은 토큰 수(누적)": computed,
                     "시간(s)": round(t.sec, 3), "결과 끝 5토큰": out[0, -5:].tolist()})
    utils.print_table(rows)
    print("→ 출력은 같고, 캐시를 쓰면 매 단계 새 토큰 1개만 계산한다(대신 K·V를 메모리에 들고 있음)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--train", type=float, metavar="MIN")
    ap.add_argument("--kv", action="store_true")
    a = ap.parse_args()
    config.PATHS["runs"].mkdir(parents=True, exist_ok=True)
    if a.info or not (a.train or a.kv):
        info()
    if a.train:
        train(a.train)
    if a.kv:
        kv_demo()


if __name__ == "__main__":
    main()
