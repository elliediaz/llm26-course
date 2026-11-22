"""2회차: 미니 Decoder-only Transformer: LLM과 같은 부품을 100줄 남짓으로 직접 조립해 본다.

  python src/mini_transformer.py --info            # 층 구성·파라미터 수·Self-Attention 안의 텐서 모양(실제 LLM 설정과 비교)
  python src/mini_transformer.py --train 1.5       # 도메인 문서로 1.5분 글자 단위 학습 → 검증 손실·생성 예시
  python src/mini_transformer.py --kv              # KV Cache 유무로 같은 문장 생성: 결과는 같고 계산량·시간이 다르다
  python src/mini_transformer.py --train 1.5 --layers 2              # 해보기 ①: 하나만 바꿔 같은 시간 학습 후 비교
  python src/mini_transformer.py --train 1.5 --dropout 0.2           # 외우기를 늦추는 장치(검증 손실이 어떻게 바뀌나)
  python src/mini_transformer.py --train 1.5 --corpus nsmc           # 학습 글 바꾸기: 영화 리뷰 3만 건으로 학습하면 리뷰처럼 씀
학습할 때마다 results/mini_gpt_runs.md 에 한 줄씩 쌓이고, 손실 곡선은 results/fig/mini_gpt_loss.png 에 그려진다

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
    def __init__(self, d, n_head, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)               # 학습 때만 일부 값을 0으로: 외우기를 늦춤(평가·생성 때는 꺼짐)

    def forward(self, x, past=None):
        a, kv = self.attn(self.ln1(x), past)
        x = x + self.drop(a)                          # 잔차: 입력을 그대로 더함
        return x + self.drop(self.mlp(self.ln2(x))), kv


class MiniGPT(nn.Module):
    def __init__(self, vocab, d=128, n_layer=4, n_head=4, max_len=256, dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)              # [B,T] → [B,T,d]
        self.pos = nn.Embedding(max_len, d)            # 위치마다 다른 벡터를 더함(어순 정보)
        self.blocks = nn.ModuleList(Block(d, n_head, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)    # [B,T,d] → [B,T,vocab]: 다음 글자 분류
        self.head.weight = self.tok.weight             # 입력·출력 임베딩 공유(많은 LLM이 쓰는 방식)
        self.max_len = max_len
        self.apply(self._init)

    @staticmethod
    def _init(mod):                                    # 처음 가중치를 작게(GPT-2 방식): 첫 손실이 무작위 찍기 수준에서 시작
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            nn.init.normal_(mod.weight, std=0.02)
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)

    def forward(self, idx, past=None):
        start = 0 if past is None else past[0][0].size(2)
        x = self.tok(idx) + self.pos(torch.arange(start, start + idx.size(1), device=idx.device))
        new_past = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, None if past is None else past[i])
            new_past.append(kv)
        return self.head(self.ln_f(x)), new_past

    @torch.no_grad()
    def generate(self, idx, n, use_cache=True, temperature=0.0, gen=None):
        """다음 글자 분류를 n번 반복. temperature 0이면 greedy(항상 1등), 0보다 크면 확률대로 뽑기."""
        past, out, computed = None, idx, 0
        for _ in range(n):
            inp = out if (past is None or not use_cache) else out[:, -1:]
            logits, new_past = self.forward(inp[:, -self.max_len:], past if use_cache else None)
            computed += inp.size(1)
            past = new_past if use_cache else None
            last = logits[:, -1]                                   # [B,vocab]: 마지막 위치의 다음 글자 점수
            if temperature > 0:
                nxt = torch.multinomial((last / temperature).softmax(-1), 1, generator=gen)
            else:
                nxt = last.argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
        return out, computed


PROMPTS = {"domain": ("서리칼날", "운영정책 POL-0"), "nsmc": ("이 영화", "배우들 연기")}


def corpus_docs(name: str = "domain") -> list[str]:
    """학습 글 목록. domain: 도메인 문서(data/<도메인>/corpus/*.md) · nsmc: 영화 리뷰(data/common/nsmc/train.tsv)."""
    if name == "nsmc":
        lines = (config.PATHS["common"] / "nsmc" / "train.tsv").read_text(encoding="utf-8").splitlines()[1:]
        reviews = [ln.rsplit("\t", 1)[0] for ln in lines]
        return ["\n".join(reviews[i:i + 1000]) for i in range(0, len(reviews), 1000)]   # 리뷰 1,000개를 문서 하나로
    d = config.data_dir()
    return [p.read_text(encoding="utf-8") for p in sorted((d / "corpus").glob("*.md"))]


def corpus_text(name: str = "domain") -> str:
    return "\n".join(corpus_docs(name))


def info(a):
    vocab = len(set(corpus_text(a.corpus)))             # 학습 때와 같은 글자 사전 크기
    m = MiniGPT(vocab=vocab, d=a.d_model, n_layer=a.layers, n_head=a.heads)
    rows = [{"부품": n, "파라미터": sum(p.numel() for p in mod.parameters())}
            for n, mod in [("토큰 임베딩", m.tok), ("위치 임베딩", m.pos)] + [(f"블록 {i}", b) for i, b in enumerate(m.blocks)]]
    rows.append({"부품": "합계(출력층은 임베딩과 공유)", "파라미터": sum(p.numel() for p in m.parameters())})
    utils.print_table(rows)
    T, H, D = 16, a.heads, a.d_model
    print(f"\nSelf-Attention 한 층 안의 텐서 모양(묶음 1, 토큰 {T}개, d_model {D}, 헤드 {H}개)")
    x = torch.randn(1, T, D)
    att = m.blocks[0].attn
    with torch.no_grad():
        qkv = att.qkv(x)
        att(x)
    for name, shape in [("입력 x", x.shape), ("qkv = Linear(d → 3d)", qkv.shape),
                        ("q·k·v 헤드별로 나눔", (1, H, T, D // H)), ("점수 q @ kᵀ", (1, H, T, T)),
                        ("가중치 softmax(행 합 1)", tuple(att.last_weights.shape)), ("가중합 a @ v → 헤드 합침", (1, T, D)),
                        ("출력 proj", (1, T, D))]:
        print(f"  {str(tuple(shape)).replace(' ', ''):<16}{name}")
    print(f"  → 점수표가 (토큰 수 × 토큰 수)라 길이가 2배면 계산·메모리는 4배")
    print("\n같은 구조, 다른 규모(공개 설정값 기준)")
    utils.print_table([
        {"모델": "MiniGPT(지금 설정)", "층": a.layers, "d_model": a.d_model, "헤드": a.heads, "어휘": f"글자 {vocab}",
         "파라미터": f"{sum(p.numel() for p in m.parameters()):,}"},
        {"모델": "Qwen3-0.6B", "층": 28, "d_model": 1024, "헤드": "Q16/KV8", "어휘": "약 15만", "파라미터": "약 6억"},
        {"모델": "Qwen3-8B", "층": 36, "d_model": 4096, "헤드": "Q32/KV8", "어휘": "약 15만", "파라미터": "약 80억"},
    ])
    print("\n※ 실제 값은 python -c \"from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3-0.6B'))\"로 확인")


def train(minutes, a):
    docs = corpus_docs(a.corpus)
    text = "\n".join(docs)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    enc = lambda s: [stoi[c] for c in s]                     # noqa: E731
    tr, va = [], []
    for doc in docs:                                         # 문서마다 끝 10%는 학습에 쓰지 않고 검증 손실을 잼
        cut = int(len(doc) * 0.9)
        tr += enc(doc[:cut] + "\n")
        va += enc(doc[cut:] + "\n")
    tr_data, va_data = torch.tensor(tr), torch.tensor(va)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    utils.set_seed()
    m = MiniGPT(len(chars), d=a.d_model, n_layer=a.layers, n_head=a.heads, max_len=max(256, a.context),
                dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    T, B, t0, step = a.context, 64, time.perf_counter(), 0
    n_param = sum(p.numel() for p in m.parameters())
    print(f"학습 글 {a.corpus} · 글자 {len(text):,}개(학습 {len(tr_data):,} · 검증 {len(va_data):,}) · 어휘 {len(chars)} · 층 {a.layers} · "
          f"d_model {a.d_model} · 헤드 {a.heads} · 문맥 {T}글자 · 드롭아웃 {a.dropout} · 파라미터 {n_param:,} · {dev}")

    def batch(src):
        ix = torch.randint(len(src) - T - 1, (B,))
        return (torch.stack([src[i:i + T] for i in ix]).to(dev), torch.stack([src[i + 1:i + T + 1] for i in ix]).to(dev))

    g = torch.Generator().manual_seed(0)                      # 검증 묶음은 매번 같은 자리(설정끼리 비교하려고)
    vix = torch.randint(len(va_data) - T - 1, (4, B), generator=g)

    @torch.no_grad()
    def val_loss():
        m.eval()
        tot = 0.0
        for ix in vix:
            x = torch.stack([va_data[i:i + T] for i in ix]).to(dev)
            y = torch.stack([va_data[i + 1:i + T + 1] for i in ix]).to(dev)
            lg, _ = m(x)
            tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()
        m.train()
        return tot / len(vix)

    curve, recent = [], []                                    # (스텝, 학습 손실 최근 평균, 검증 손실)
    print(f"  {'스텝':>5}  {'학습 손실':>8}  {'검증 손실':>8}   (검증 = 학습에 안 쓴 글에서 다음 글자 맞히기)")
    while time.perf_counter() - t0 < minutes * 60:
        x, y = batch(tr_data)
        logits, _ = m(x)                                      # [B,T] → [B,T,vocab]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))   # 다음 글자 분류
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        recent.append(loss.item())
        if step % 50 == 0:
            curve.append((step, sum(recent) / len(recent), val_loss()))
            recent = []
            if step % 100 == 0:
                print(f"  {step:5d}  {curve[-1][1]:8.3f}  {curve[-1][2]:8.3f}")
    if recent:
        curve.append((step, sum(recent) / len(recent), val_loss()))
    tl, vl = curve[-1][1], curve[-1][2]
    best = min(curve, key=lambda c: c[2])
    print(f"  학습 {step} 스텝 · 마지막 학습 손실 {tl:.3f} · 검증 손실 {vl:.3f} · 가장 낮았던 검증 손실 {best[2]:.3f}(스텝 {best[0]})")
    if vl - best[2] > 0.1:
        print("  → 검증 손실이 바닥을 찍고 다시 올라감: 학습 글을 외우기 시작했다는 신호(과적합)")
    plot_curve(curve, a, minutes)
    torch.save({"state": m.state_dict(), "chars": chars,
                "cfg": {"d": a.d_model, "n_layer": a.layers, "n_head": a.heads, "max_len": max(256, a.context)}},
               config.PATHS["runs"] / "mini_gpt.pt")
    samples = {}
    m.eval()
    for prompt in PROMPTS[a.corpus]:
        idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=dev)
        for how, temp in (("greedy", 0.0), ("T 0.8", 0.8)):          # 같은 모델, 고르는 법만 다름
            gen = torch.Generator(device=dev).manual_seed(0)
            out, _ = m.generate(idx, 60, temperature=temp, gen=gen)
            samples[f"{prompt} ({how})"] = "".join(chars[i] for i in out[0].tolist())
            print(f"\n[생성·{how}] " + samples[f"{prompt} ({how})"].replace("\n", " ⏎ "))
    rec = {"학습 글": a.corpus, "층": a.layers, "d_model": a.d_model, "헤드": a.heads, "문맥": T, "학습률": a.lr, "드롭아웃": a.dropout,
           "분": minutes, "파라미터": n_param, "스텝": step, "학습 손실": f"{tl:.3f}", "검증 손실": f"{vl:.3f}",
           "최저 검증(스텝)": f"{best[2]:.3f}({best[0]})"}
    utils.write_json(utils.result_path("mini_gpt.json"), {**rec, "curve": curve, "samples": samples, "measured_at": utils.now()})
    log = utils.result_path("mini_gpt_runs.md")
    head = ("| 학습 글 | 층 | d_model | 헤드 | 문맥 | 학습률 | 드롭아웃 | 분 | 파라미터 | 스텝 | 학습 손실 | 검증 손실 | 최저 검증(스텝) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    line = "| " + " | ".join(str(v) for v in rec.values()) + " |\n"
    old = log.read_text(encoding="utf-8") if log.exists() else ""
    if "학습 글" not in old:                                 # 예전 형식 기록은 새 머리글로 다시 시작
        old = "# 미니 GPT 학습 기록(해보기 ①)\n\n" + head
    log.write_text(old + line, encoding="utf-8")
    print(f"\n기록: {log} (설정을 바꿔 다시 돌리면 한 줄씩 쌓임)")
    return m


def plot_curve(curve, a, minutes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    utils.korean_font()
    st = [c[0] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(st, [c[1] for c in curve], label="학습 손실", color="#3B3D8C")
    ax.plot(st, [c[2] for c in curve], label="검증 손실", color="#C8281E")
    b = min(curve, key=lambda c: c[2])
    ax.scatter([b[0]], [b[2]], color="#C8281E", zorder=5)
    ax.annotate(f"최저 {b[2]:.2f}", (b[0], b[2]), textcoords="offset points", xytext=(6, -14), color="#C8281E")
    ax.set_xlabel("스텝")
    ax.set_ylabel("손실(낮을수록 잘 맞힘)")
    ax.set_title(f"{a.corpus} · 층 {a.layers} · d {a.d_model} · 헤드 {a.heads} · 문맥 {a.context} · 학습률 {a.lr} · 드롭아웃 {a.dropout} · {minutes}분",
                 fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    out = utils.fig_path("mini_gpt_loss.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  손실 곡선: {out}")


def kv_demo():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = config.PATHS["runs"] / "mini_gpt.pt"
    if p.exists():
        ck = torch.load(p, map_location=dev)
        cfg = ck.get("cfg", {})
        m = MiniGPT(len(ck["chars"]), **cfg).to(dev)          # 학습 때 설정 그대로(드롭아웃은 생성 때 꺼져 있음)
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
    ap.add_argument("--layers", type=int, default=4, help="블록(층) 수")
    ap.add_argument("--d-model", type=int, default=128, help="토큰 하나를 나타내는 숫자 개수")
    ap.add_argument("--heads", type=int, default=4, help="어텐션 헤드 수(d_model을 나눠 떨어져야 함)")
    ap.add_argument("--context", type=int, default=128, help="한 번에 보는 글자 수")
    ap.add_argument("--lr", type=float, default=3e-3, help="학습률(한 번에 가중치를 얼마나 움직일지)")
    ap.add_argument("--dropout", type=float, default=0.0, help="학습 때 값을 0으로 만드는 비율(0~0.5)")
    ap.add_argument("--corpus", choices=sorted(PROMPTS), default="domain", help="학습 글: domain(도메인 문서) · nsmc(영화 리뷰)")
    a = ap.parse_args()
    if a.d_model % a.heads:
        ap.error(f"d_model({a.d_model})이 헤드 수({a.heads})로 나눠 떨어지지 않음")
    config.PATHS["runs"].mkdir(parents=True, exist_ok=True)
    if a.info or not (a.train or a.kv):
        info(a)
    if a.train:
        train(a.train, a)
    if a.kv:
        kv_demo()


if __name__ == "__main__":
    main()
