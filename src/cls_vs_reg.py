"""1회차 해보기 F: 같은 문제를 회귀로도, 분류로도 풀어 본다(한글 금액 수사 → 숫자).

  python src/cls_vs_reg.py                       # 두 방식을 각각 학습(기본 60초씩) → 비교표·그림
  python src/cls_vs_reg.py --seconds 30          # 시간이 없을 때
  python src/cls_vs_reg.py --show "삼억 칠천이백만 오천 원"   # 학습 후 한 문장 예측

같은 입력, 같은 인코더 구조에서 '출력층과 손실'만 다르다.
  공통   금액 수사 문장 → 글자 임베딩 → 양방향 GRU → 문장 벡터 h(256차원)
  회귀   h → 숫자 1개(log10 금액).          손실 MSE(평균제곱오차)
  분류   h → 9자리 × 10클래스(각 자리 0~9).  손실 교차 엔트로피
확인할 것
  - 회귀는 '대략의 크기'는 잘 맞히지만 정확한 금액은 거의 못 맞힌다(연속값의 근사)
  - 분류는 자리마다 정답 글자를 고르므로 정확히 맞히는 비율이 높다
  - LLM이 숫자를 쓰는 방식은 분류 쪽이다(숫자 '글자'를 고름). 그래서 계산은 실수할 수 있고,
    정확한 계산은 규칙·Tool에 맡긴다(8회차)
결과: results/cls_vs_reg.json, results/fig/cls_vs_reg.png
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
import utils

sys.path.insert(0, str(config.PATHS["common"] / "numerals"))
from make_numerals import sino  # noqa: E402  (1회차 수사 변환 데이터 생성기와 같은 규칙)

DIGITS = 9                                  # 1 ~ 999,999,999원
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 데이터
def make_rows(n: int, seed: int) -> list[dict]:
    """자릿수(1~9자리)를 고르게 뽑아 금액을 만든다 → 작은 금액과 큰 금액이 비슷한 비율로 섞인다."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        k = rng.randint(1, DIGITS)
        v = rng.randint(10 ** (k - 1), 10 ** k - 1)
        txt = sino(v, rng) + (" 원" if rng.random() < 0.5 else "원")
        rows.append({"src": txt, "value": v})
    return rows


class CharVocab:
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = ["<pad>", "<unk>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, texts: list[str]) -> torch.Tensor:
        L = max(len(t) for t in texts)
        x = torch.zeros(len(texts), L, dtype=torch.long)
        for i, t in enumerate(texts):
            x[i, :len(t)] = torch.tensor([self.stoi.get(c, 1) for c in t])
        return x


def digits_of(values: list[int]) -> torch.Tensor:
    """372005000 → [0,0,0,...]가 아니라 9자리로 앞을 0으로 채운 각 자리 숫자 (B, 9)."""
    return torch.tensor([[int(c) for c in f"{v:0{DIGITS}d}"] for v in values])


# ---------------------------------------------------------------- 모델: 인코더는 같고 출력층만 다르다
class Encoder(nn.Module):
    def __init__(self, V: int, emb: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, 256), nn.ReLU())

    def forward(self, x):
        lengths = (x != 0).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)                     # h: (2, B, hidden) 정방향·역방향 마지막 상태
        return self.proj(torch.cat([h[0], h[1]], dim=-1))


class Regressor(nn.Module):
    """출력 1개 = 연속값(log10 금액)."""
    def __init__(self, V):
        super().__init__()
        self.enc = Encoder(V)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(-1)


class Classifier(nn.Module):
    """출력 9 × 10 = 자리마다 '0~9 중 어느 것인가'를 고르는 분류 9개."""
    def __init__(self, V):
        super().__init__()
        self.enc = Encoder(V)
        self.head = nn.Linear(256, DIGITS * 10)

    def forward(self, x):
        return self.head(self.enc(x)).view(-1, DIGITS, 10)


def loss_fn(kind, out, values, dig):
    if kind == "reg":
        y = torch.log10(values.float())
        return F.mse_loss(out, y)                   # 회귀: 평균제곱오차
    return F.cross_entropy(out.reshape(-1, 10), dig.reshape(-1))   # 분류: 교차 엔트로피


@torch.no_grad()
def predict(kind, model, x) -> list[int]:
    model.eval()
    out = model(x.to(DEV))
    if kind == "reg":
        return [int(round(10 ** float(v))) for v in out.clamp(0, DIGITS).cpu()]
    return [int("".join(str(d) for d in row)) for row in out.argmax(-1).cpu().tolist()]


def train(kind, vocab, rows, seconds, bs=256, seed=config.SEED):
    utils.set_seed(seed)
    model = (Regressor if kind == "reg" else Classifier)(len(vocab.itos)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    t0, step, hist = time.perf_counter(), 0, []
    rng = random.Random(seed)
    while time.perf_counter() - t0 < seconds:
        batch = rng.sample(rows, bs)
        x = vocab.encode([r["src"] for r in batch]).to(DEV)
        values = torch.tensor([r["value"] for r in batch], device=DEV)
        dig = digits_of([r["value"] for r in batch]).to(DEV)
        model.train()
        loss = loss_fn(kind, model(x), values, dig)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            hist.append((round(time.perf_counter() - t0, 1), round(loss.item(), 4)))
    return model, {"steps": step, "seconds": round(time.perf_counter() - t0, 1), "loss_hist": hist}


def evaluate(kind, model, vocab, rows) -> dict:
    preds = []
    for i in range(0, len(rows), 512):
        chunk = rows[i:i + 512]
        preds += predict(kind, model, vocab.encode([r["src"] for r in chunk]))
    gold = [r["value"] for r in rows]
    exact = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    mag = sum(len(str(max(p, 1))) == len(str(g)) for p, g in zip(preds, gold)) / len(gold)
    rel = sorted(abs(p - g) / g for p, g in zip(preds, gold))
    out = {"exact": round(exact, 4), "magnitude": round(mag, 4), "median_rel_err": round(rel[len(rel) // 2], 4)}
    if kind == "cls":
        pd, gd = digits_of(preds), digits_of(gold)
        out["digit_acc"] = round((pd == gd).float().mean().item(), 4)
    return out, preds


def plot(rows, preds_reg, preds_cls, res):
    utils.korean_font()
    import matplotlib.pyplot as plt
    g = [r["value"] for r in rows]
    err = lambda ps: [max(-20, min(20, (p - v) / v * 100)) for p, v in zip(ps, g)]  # noqa: E731  ±20%에서 자름
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    bins = [x * 0.5 for x in range(-40, 41)]
    ax[0].hist(err(preds_reg), bins=bins, alpha=0.6, label="회귀")
    ax[0].hist(err(preds_cls), bins=bins, alpha=0.6, label="분류(자리별)")
    ax[0].set_xlabel("예측 오차(%) = (예측 - 정답) / 정답 × 100")
    ax[0].set_ylabel("문장 수")
    ax[0].set_title("회귀는 0 근처에 퍼지고, 분류는 0 한 곳에 모인다")
    ax[0].legend()
    names = ["정확히 맞힘", "자릿수 맞힘"]
    x = range(len(names))
    ax[1].bar([i - 0.2 for i in x], [res["reg"]["exact"], res["reg"]["magnitude"]], width=0.4, label="회귀")
    ax[1].bar([i + 0.2 for i in x], [res["cls"]["exact"], res["cls"]["magnitude"]], width=0.4, label="분류(자리별)")
    ax[1].set_xticks(list(x), names)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("같은 데이터·같은 인코더, 출력층만 다름")
    ax[1].legend()
    fig.tight_layout()
    path = utils.fig_path("cls_vs_reg.png")
    fig.savefig(path, dpi=150)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60, help="방식별 학습 시간(초)")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--show", help="학습 후 이 문장을 두 방식으로 예측")
    a = ap.parse_args()

    train_rows, test_rows = make_rows(a.n_train, 1), make_rows(2000, 2)
    vocab = CharVocab([r["src"] for r in train_rows + test_rows])
    print(f"[데이터] 학습 {len(train_rows)} · 시험 {len(test_rows)} · 예: {train_rows[0]['src']} → {train_rows[0]['value']}")
    print(f"[장치] {DEV} · 방식별 학습 {a.seconds:.0f}초")

    res, preds, models = {}, {}, {}
    for kind, name in (("reg", "회귀"), ("cls", "분류")):
        with utils.section(f"{name} 학습"):
            model, info = train(kind, vocab, train_rows, a.seconds)
            res[kind], preds[kind] = evaluate(kind, model, vocab, test_rows)
            res[kind].update(info)
            models[kind] = model
            n_out = model.head.out_features
            print(f"출력층 크기 {n_out} · 학습 {info['steps']}스텝 · 마지막 손실 {info['loss_hist'][-1][1] if info['loss_hist'] else '-'}")

    print("\n## 시험 2000건 결과")
    table = utils.print_table([
        {"방식": "회귀(숫자 1개)", "정확히 맞힘": res["reg"]["exact"], "자릿수 맞힘": res["reg"]["magnitude"],
         "상대오차 중앙값": res["reg"]["median_rel_err"]},
        {"방식": "분류(9자리×10)", "정확히 맞힘": res["cls"]["exact"], "자릿수 맞힘": res["cls"]["magnitude"],
         "상대오차 중앙값": res["cls"]["median_rel_err"]},
    ])
    print("\n## 예시")
    ex = [{"입력": r["src"], "정답": r["value"], "회귀": preds["reg"][i], "분류": preds["cls"][i]}
          for i, r in enumerate(test_rows[:6])]
    utils.print_table(ex)
    if a.show:
        x = vocab.encode([a.show])
        print(f"\n[{a.show}] 회귀 → {predict('reg', models['reg'], x)[0]:,} / 분류 → {predict('cls', models['cls'], x)[0]:,}")

    fig = plot(test_rows, preds["reg"], preds["cls"], res)
    out = utils.write_json(utils.result_path("cls_vs_reg.json"),
                           {"device": DEV, "seconds_each": a.seconds, "result": res, "examples": ex, "table": table})
    print(f"\n저장: {out}, {fig}")


if __name__ == "__main__":
    main()
