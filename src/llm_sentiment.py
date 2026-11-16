"""2회차 실습 F: 1회차 LSTM과 같은 영화 리뷰를 LLM으로 분류한다(학습 없이)

  python src/llm_sentiment.py                     # Qwen3-0.6B, 예시 0개, 리뷰 60건(코랩 CPU 약 1분)
  python src/llm_sentiment.py --shots 4           # 앞에 예시 4개를 붙임(문맥 내 학습)
  python src/llm_sentiment.py --model Qwen/Qwen3-0.6B-Base
  python src/llm_sentiment.py --text "배우는 좋은데 각본이 아쉽다"

방법: 프롬프트 뒤에 ' 긍정'과 ' 부정'을 각각 붙여 두 문장의 확률을 비교함
      = 후보 두 개 중 하나를 고르는 분류. 학습한 가중치는 하나도 바꾸지 않음
비교: 1회차 LSTM(2에폭 학습) 정확도 0.80, 길이별 값은 ref/s01/nn_rnn.json
결과: results/llm_sentiment.json, results/fig/llm_sentiment.png
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
import utils

LABELS = [" 부정", " 긍정"]                     # 0 부정, 1 긍정(NSMC 라벨과 같은 순서)
BUCKETS = [(1, 20, "1~20자"), (21, 40, "21~40자"), (41, 80, "41~80자"), (81, 10**6, "81~자")]
HEAD = "영화 리뷰의 감정을 긍정 또는 부정으로 답함.\n\n"


def read_tsv(name):
    with open(config.PATHS["common"] / "nsmc" / name, encoding="utf-8") as f:
        return [(r["document"], int(r["label"])) for r in csv.DictReader(f, delimiter="\t") if r["document"]]


def prefix(shots):
    return HEAD + "".join(f"리뷰: {t}\n감정:{LABELS[y]}\n\n" for t, y in shots)


@torch.no_grad()
def score(model, tok, pre_ids, cache, text, dev):
    """앞부분(지시문·예시)은 한 번만 계산해 두고, 리뷰 뒤에 라벨을 붙인 두 경우의 로그 확률 합을 비교함"""
    body = tok(f"리뷰: {text}\n감정:", add_special_tokens=False).input_ids
    scores = []
    for lab in LABELS:
        ids = tok(lab, add_special_tokens=False).input_ids
        x = torch.tensor([body + ids], device=dev)
        past = copy.deepcopy(cache) if cache is not None else None
        lp = model(x, past_key_values=past, use_cache=past is not None).logits[0].float().log_softmax(-1)
        n = len(body)
        scores.append(sum(lp[n - 1 + i, t].item() for i, t in enumerate(ids)))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.SMALL_MODEL_ID)
    ap.add_argument("--n", type=int, default=60, help="시험할 리뷰 수(긍정·부정 반반)")
    ap.add_argument("--shots", type=int, default=0, help="앞에 붙일 예시 수")
    ap.add_argument("--text", nargs="*", help="내 문장만 분류")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()

    rng = random.Random(a.seed)
    train = read_tsv("train.tsv")
    pos = [r for r in train if r[1] == 1 and len(r[0]) <= 40]
    neg = [r for r in train if r[1] == 0 and len(r[0]) <= 40]
    shots = []
    for i in range(a.shots):
        shots.append(rng.choice(pos if i % 2 else neg))
    rng.shuffle(shots)

    pre_ids = tok(prefix(shots), add_special_tokens=False).input_ids
    with torch.no_grad():
        cache = model(torch.tensor([pre_ids], device=dev), use_cache=True).past_key_values

    if a.text:
        for t in a.text:
            s = score(model, tok, pre_ids, cache, t, dev)
            p = torch.tensor(s).softmax(0)
            print(f"긍정 {p[1]:.2f} · 부정 {p[0]:.2f}  {t}")
        return

    test = read_tsv("test.tsv")
    half = a.n // 2
    sample = rng.sample([r for r in test if r[1] == 1], half) + rng.sample([r for r in test if r[1] == 0], a.n - half)
    rng.shuffle(sample)

    t0, rows = time.time(), []
    for text, y in sample:
        s = score(model, tok, pre_ids, cache, text, dev)
        pred = int(s[1] > s[0])
        rows.append({"text": text, "label": y, "pred": pred, "len": len(text)})
    sec = round(time.time() - t0, 1)
    acc = sum(r["pred"] == r["label"] for r in rows) / len(rows)
    by_len = {}
    for lo, hi, name in BUCKETS:
        sub = [r for r in rows if lo <= r["len"] <= hi]
        if sub:
            by_len[name] = round(sum(r["pred"] == r["label"] for r in sub) / len(sub), 3)
    pred_pos = sum(r["pred"] for r in rows) / len(rows)

    print(f"모델 {a.model} · 예시 {a.shots}개 · 리뷰 {len(rows)}건 · {sec}초")
    print(f"정확도 {acc:.3f}  (1회차 LSTM 0.800) · '긍정'으로 답한 비율 {pred_pos:.2f}")
    print("길이별:", "  ".join(f"{k} {v:.2f}" for k, v in by_len.items()))
    wrong = [r for r in rows if r["pred"] != r["label"]][:5]
    for r in wrong:
        print(f"  틀림  정답 {LABELS[r['label']].strip()}  {r['text'][:60]}")

    name = f"llm_sentiment_{a.model.split('/')[-1]}_{a.shots}shot"
    utils.write_json(utils.result_path(f"{name}.json"),
                     {"model": a.model, "shots": a.shots, "n": len(rows), "sec": sec, "acc": round(acc, 3),
                      "pred_pos_rate": round(pred_pos, 3), "by_length": by_len, "wrong": wrong})

    try:
        lstm = utils.read_json(config.ROOT / "ref" / "s01" / "nn_rnn.json")["result"]["lstm"]["by_length"]
    except Exception:
        lstm = {}
    import matplotlib.pyplot as plt
    utils.korean_font()
    names = [b[2] for b in BUCKETS if b[2] in by_len]
    xs = range(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if lstm:
        ax.bar([x - 0.2 for x in xs], [lstm.get(n, 0) for n in names], 0.4, label="1회차 LSTM(학습 2에폭)", color="#8aa1c1")
    ax.bar([x + 0.2 for x in xs], [by_len[n] for n in names], 0.4, label=f"LLM 0.6B(학습 없음, 예시 {a.shots}개)", color="#c0622b")
    ax.set_xticks(list(xs), names); ax.set_ylim(0.4, 1.0); ax.set_ylabel("정확도"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("리뷰 길이별 감성 분류 정확도")
    fig.tight_layout()
    out = utils.fig_path("llm_sentiment.png")
    fig.savefig(out, dpi=130)
    print("그림:", out)


if __name__ == "__main__":
    main()
