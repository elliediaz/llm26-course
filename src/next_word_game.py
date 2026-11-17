"""2회차 미션 M4: 다음 단어 맞히기. LLM이 하는 일은 '다음 토큰 고르기' 하나뿐임을 확인

  python src/next_word_game.py                                   # 속담 50개(한국어 40·영어 10) 이어 말하기 채점
  python src/next_word_game.py --model both                      # 대화용(Instruct)과 원본(Base) 비교
  python src/next_word_game.py --prefix "가는 말이 고와야"        # 다음 토큰 후보 10개와 확률, 이어 쓰기
  python src/next_word_game.py --prefix "오늘 점심 메뉴는" --temp 1.0 --samples 5   # 온도를 올려 여러 번 뽑기

채점
  완전: 모델이 이어 쓴 글이 정답 뒷부분과 띄어쓰기 빼고 같게 시작함
  첫 낱말: 정답의 첫 낱말까지만 맞음
  후보 순위: 정답 첫 토큰이 다음 토큰 후보 중 몇 번째였나(1이면 가장 유력, 어휘 약 15만 개 중)
결과: results/next_word_game_<모델>.json, results/fig/next_word_game.png
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import config
import utils

DATA = config.ROOT / "data" / "s02" / "proverbs.json"
MODELS = {"small": config.SMALL_MODEL_ID, "base": config.BASE_MODEL_ID}


def load(model_id):
    hf_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(model_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32                       # T4는 bfloat16을 하드웨어로 지원하지 않으므로 float32로 통일
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
    return tok, model


def squeeze(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


@torch.no_grad()
def topk(tok, model, prefix: str, k: int = 10):
    ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    p = model(ids).logits[0, -1].float().softmax(-1)
    v, i = p.topk(k)
    return [(tok.decode([t]), round(x, 3)) for t, x in zip(i.tolist(), v.tolist())], p


@torch.no_grad()
def cont(tok, model, prefix: str, n: int, temp: float = 0.0) -> str:
    ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    kw = {"do_sample": True, "temperature": temp, "top_p": 1.0, "top_k": 0} if temp > 0 else {"do_sample": False}
    out = model.generate(ids, max_new_tokens=n, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def quiz(tok, model, items):
    rows = []
    for it in items:
        prefix, ans = it["prefix"], it["answer"]
        first_ids = tok(" " + ans, add_special_tokens=False).input_ids
        gen = cont(tok, model, prefix, len(first_ids) + 2)
        _, p = topk(tok, model, prefix, 1)
        rank = int((p > p[first_ids[0]]).sum().item()) + 1
        g, s = squeeze(gen), squeeze(ans)
        first = squeeze(ans.split()[0])
        rows.append({"앞부분": prefix, "정답": ans, "모델": gen.replace("\ufffd", "").strip().split("\n")[0][:24],
                     "완전": int(g[: len(s)] == s), "첫 낱말": int(g[: len(first)] == first), "후보 순위": rank})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="small · base · both · 허깅페이스 모델 이름")
    ap.add_argument("--prefix", nargs="*", help="내가 고른 앞부분")
    ap.add_argument("--temp", type=float, default=0.0, help="0이면 가장 유력한 것만(탐욕), 높일수록 다양")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--len", type=int, default=16, help="이어 쓸 토큰 수")
    a = ap.parse_args()
    keys = ["small", "base"] if a.model == "both" else [a.model]

    if a.prefix:
        for key in keys:
            mid = MODELS.get(key, key)
            tok, model = load(mid)
            print(f"\n===== {mid.split('/')[-1]} =====")
            for pre in a.prefix:
                top, _ = topk(tok, model, pre)
                print(f"\n'{pre}' 다음 토큰 후보")
                for t, p in top:
                    print(f"  {p:6.3f}  {'█' * int(p * 40):40s} {t!r}")
                torch.manual_seed(0)
                for s in range(a.samples):
                    print(f"  이어 쓰기{'' if a.samples == 1 else f' {s + 1}'}: {pre}{cont(tok, model, pre, a.len, a.temp)}")
        return

    d = utils.read_json(DATA)
    summary = []
    for key in keys:
        mid = MODELS.get(key, key)
        tok, model = load(mid)
        t0 = time.time()
        res = {lang: quiz(tok, model, d[lang]) for lang in ("ko", "en")}
        name = mid.split("/")[-1]
        print(f"\n===== {name} · {time.time() - t0:.0f}초 =====")
        utils.print_table(res["ko"][:12] + res["en"][:4])
        for lang in ("ko", "en"):
            r = res[lang]
            row = {"모델": name, "언어": "한국어" if lang == "ko" else "영어", "문항": len(r),
                   "완전": round(sum(x["완전"] for x in r) / len(r), 2), "첫 낱말": round(sum(x["첫 낱말"] for x in r) / len(r), 2),
                   "1순위 비율": round(sum(x["후보 순위"] == 1 for x in r) / len(r), 2),
                   "5위 안 비율": round(sum(x["후보 순위"] <= 5 for x in r) / len(r), 2)}
            summary.append(row)
        utils.write_json(utils.result_path(f"next_word_game_{name}.json"), res)
    print("\n요약")
    utils.print_table(summary)

    import matplotlib.pyplot as plt
    utils.korean_font()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = [f"{r['모델']}\n{r['언어']}" for r in summary]
    ax.bar(range(len(summary)), [r["첫 낱말"] for r in summary], 0.5, color="#8aa1c1", label="첫 낱말")
    ax.bar(range(len(summary)), [r["완전"] for r in summary], 0.3, color="#c0622b", label="완전")
    ax.set_xticks(range(len(summary)), labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("정답 비율"); ax.legend(fontsize=8)
    ax.set_title("속담 이어 말하기", fontsize=10)
    fig.tight_layout()
    out = utils.fig_path("next_word_game.png")
    fig.savefig(out, dpi=130)
    print("그림:", out)


if __name__ == "__main__":
    main()
