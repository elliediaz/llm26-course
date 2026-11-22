"""2회차 미션 M5: 학습 없이 분류기 만들기(프롬프트 경연)

  python src/llm_classify.py --list                          # 쓸 수 있는 과제와 라벨 보기
  python src/llm_classify.py --task delivery                 # 배달 리뷰, 예시 0개, 개발셋(dev)으로 점검
  python src/llm_classify.py --task delivery --shots 5       # 라벨마다 예시 1개씩 앞에 붙임
  python src/llm_classify.py --task delivery --calibrate     # 라벨 쏠림 보정
  python src/llm_classify.py --task delivery --prompt prompts/my_prompt.txt --tag v2
  python src/llm_classify.py --task delivery --shots 5 --split test --tag 최종   # 시험셋은 마지막에 한 번만
  python src/llm_classify.py --task smishing --text "택배 주소 확인: http://a.example"
  python src/llm_classify.py --data data/s02/my_task.json    # 직접 만든 데이터

과제(data/s02/)
  delivery  배달앱 리뷰 유형 5종      smishing  스미싱 문자 판별 2종
  news      뉴스 제목 분야 5종        game      게임 고객 문의 유형 5종(도메인 G 데이터)
  nsmc      영화 리뷰 감정 2종(1회차 LSTM과 같은 데이터)

방법
  지시문 + 예시 + "입력: ... / 답: " 뒤에 라벨 단어를 하나씩 붙여 보고, 모델이 가장 그럴듯하다고 본
  라벨을 고름. 가중치는 바꾸지 않음. 고칠 수 있는 것은 지시문(--prompt), 예시 수(--shots),
  라벨 단어(--words), 보정(--calibrate), 모델(--model base) 다섯 가지
결과
  results/llm_classify_<과제>_<태그>.json, results/fig/llm_classify_<과제>.png
  results/leaderboard.csv 에 한 줄씩 누적

메모(11/17, 2코어 CPU)
  - 개발셋 한 번 30초 안팎, 시험셋은 과제에 따라 25~90초
  - 보정을 켜면 대개 오르지만 늘 그렇지는 않음(Base 모델 배달 리뷰는 0.68 → 0.62). 켜고 끈 값을 둘 다 적어 둘 것
  - 지시문 예시는 prompts/예시_delivery.txt
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import config
import utils

DATA = config.ROOT / "data" / "s02"
TASKS = {"delivery": "delivery_reviews.json", "smishing": "smishing.json", "news": "news_ynat.json",
         "game": "game_inquiry.json", "nsmc": None}
MODELS = {"small": config.SMALL_MODEL_ID, "base": config.BASE_MODEL_ID}
CONTENT_FREE = ["", "N/A", "…"]          # 보정용 '내용 없는 입력'


def load_task(name: str, seed: int) -> dict:
    if TASKS[name]:
        return utils.read_json(DATA / TASKS[name])
    # nsmc: 1회차 데이터에서 개발셋 20건(학습셋), 시험셋 60건(평가셋)을 그때그때 뽑음
    def read(fn):
        with open(config.PATHS["common"] / "nsmc" / fn, encoding="utf-8") as f:
            return [(r["document"], int(r["label"])) for r in csv.DictReader(f, delimiter="\t") if r["document"]]
    rng, names = random.Random(seed), ["부정", "긍정"]
    def pick(rows, n):
        out = rng.sample([r for r in rows if r[1] == 1 and len(r[0]) <= 60], n // 2) + \
              rng.sample([r for r in rows if r[1] == 0 and len(r[0]) <= 60], n - n // 2)
        rng.shuffle(out)
        return [{"text": t, "label": names[y]} for t, y in out]
    return {"name": "영화 리뷰 감정", "task": "nsmc", "instruction": "영화 리뷰의 감정을 긍정 또는 부정으로 답함.",
            "input_name": "리뷰", "answer_name": "감정", "labels": names,
            "source": "NSMC(e9t/nsmc)", "license": "CC0 1.0",
            "dev": pick(read("train.tsv"), 20), "test": pick(read("test.tsv"), 60)}


def pick_shots(dev: list[dict], labels: list[str], k: int, seed: int) -> list[dict]:
    """라벨을 돌아가며 하나씩 골라 k개를 채움(라벨 쏠림 방지)"""
    rng = random.Random(seed)
    pools = {lab: rng.sample([r for r in dev if r["label"] == lab], sum(r["label"] == lab for r in dev)) for lab in labels}
    shots, i = [], 0
    while len(shots) < k and any(pools.values()):
        lab = labels[i % len(labels)]
        if pools[lab]:
            shots.append(pools[lab].pop())
        i += 1
    rng.shuffle(shots)
    return shots


class Scorer:
    """앞부분(지시문·예시)은 한 번만 계산해 KV 캐시로 저장하고, 문장마다 뒷부분만 이어서 계산함"""

    def __init__(self, model, tok, head: str, inp: str, ans: str, words: list[str], norm: str):
        self.model, self.tok, self.inp, self.ans, self.norm = model, tok, inp, ans, norm
        self.dev = next(model.parameters()).device
        self.word_ids = [tok(" " + w, add_special_tokens=False).input_ids for w in words]
        ids = tok(head, add_special_tokens=False).input_ids
        self.n_prefix = len(ids)
        with torch.no_grad():
            self.cache = model(torch.tensor([ids], device=self.dev), use_cache=True).past_key_values

    @torch.no_grad()
    def __call__(self, text: str) -> list[float]:
        body = self.tok(f"{self.inp}: {text}\n{self.ans}:", add_special_tokens=False).input_ids
        out = self.model(torch.tensor([body], device=self.dev), past_key_values=copy.deepcopy(self.cache), use_cache=True)
        first = out.logits[0, -1].float().log_softmax(-1)
        scores = []
        for ids in self.word_ids:
            s = first[ids[0]].item()
            if len(ids) > 1:            # 라벨이 여러 토큰이면 나머지 토큰 확률도 이어서 더함
                lp = self.model(torch.tensor([ids[:-1]], device=self.dev), past_key_values=copy.deepcopy(out.past_key_values),
                                use_cache=True).logits[0].float().log_softmax(-1)
                s += sum(lp[i, t].item() for i, t in enumerate(ids[1:]))
            scores.append(s / len(ids) if self.norm == "mean" else s)
        return scores


def normalize(s: list[float]) -> list[float]:
    m = max(s)
    z = m + math.log(sum(math.exp(v - m) for v in s))
    return [v - z for v in s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="delivery", choices=list(TASKS))
    ap.add_argument("--data", help="직접 만든 데이터 파일(data/s02/README.md 구조). 주면 --task 대신 씀")
    ap.add_argument("--split", default="dev", choices=["dev", "test"], help="dev: 고치면서 보는 셋, test: 마지막에 한 번")
    ap.add_argument("--shots", type=int, default=0, help="앞에 붙일 예시 수(개발셋에서 뽑음)")
    ap.add_argument("--prompt", help="지시문을 바꿀 텍스트 파일")
    ap.add_argument("--instruction", help="지시문을 바꿀 문자열(--prompt 대신)")
    ap.add_argument("--words", help="라벨 대신 쓸 단어, 라벨 순서대로 쉼표 구분(예: 맛없음,늦음,적음,새고더러움,만족)")
    ap.add_argument("--calibrate", action="store_true", help="내용 없는 입력의 점수를 빼서 라벨 쏠림을 보정")
    ap.add_argument("--norm", default="sum", choices=["sum", "mean"], help="여러 토큰 라벨 점수: 합(sum)·평균(mean)")
    ap.add_argument("--model", default="small", help="small(Qwen3-0.6B) · base(Qwen3-0.6B-Base) · 허깅페이스 모델 이름")
    ap.add_argument("--limit", type=int, help="앞에서부터 이 건수만 평가(빠른 점검용)")
    ap.add_argument("--tag", default="", help="리더보드에 남길 이름(시도 번호 등)")
    ap.add_argument("--text", nargs="*", help="내 문장만 분류")
    ap.add_argument("--show-prompt", action="store_true", help="모델에 들어가는 앞부분을 출력")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.list:
        for name in TASKS:
            d = load_task(name, a.seed)
            print(f"{name:9s} {d['name']:12s} 라벨 {', '.join(d['labels'])}  · 개발 {len(d['dev'])}건 · 시험 {len(d['test'])}건")
        return

    d = utils.read_json(a.data) if a.data else load_task(a.task, a.seed)
    if a.data:
        a.task = d.get("task") or Path(a.data).stem
    labels = d["labels"]
    words = [w.strip() for w in a.words.split(",")] if a.words else labels
    if len(words) != len(labels):
        raise SystemExit(f"--words 는 라벨 수({len(labels)})만큼 적어야 함: {labels}")
    instruction = d["instruction"]
    if a.prompt:
        instruction = Path(a.prompt).read_text(encoding="utf-8").strip()
    elif a.instruction:
        instruction = a.instruction
    shots = pick_shots(d["dev"], labels, a.shots, a.seed)
    shot_texts = {s["text"] for s in shots}
    w_of = dict(zip(labels, words))
    head = instruction + "\n\n" + "".join(f"{d['input_name']}: {s['text']}\n{d['answer_name']}: {w_of[s['label']]}\n\n" for s in shots)
    if a.show_prompt:
        print("----- 모델에 들어가는 앞부분 -----")
        print(head + f"{d['input_name']}: (분류할 문장)\n{d['answer_name']}: (라벨 후보: {' / '.join(words)})")
        print("--------------------------------")

    model_id = MODELS.get(a.model, a.model)
    hf_logging.disable_progress_bar()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32                       # 0.6B는 T4에서도 float32로 충분하고 수치가 안정적임
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(dev).eval()
    scorer = Scorer(model, tok, head, d["input_name"], d["answer_name"], words, a.norm)

    cf = None
    if a.calibrate:
        cfs = [normalize(scorer(t)) for t in CONTENT_FREE]
        cf = [sum(c[i] for c in cfs) / len(cfs) for i in range(len(labels))]

    def predict(text):
        s = normalize(scorer(text))
        if cf:
            s = [v - c for v, c in zip(s, cf)]
        return s

    if a.text:
        for t in a.text:
            p = torch.tensor(predict(t)).softmax(0).tolist()
            best = max(range(len(labels)), key=lambda i: p[i])
            print(f"{labels[best]:6s} ({p[best]:.2f})  " + " · ".join(f"{l} {v:.2f}" for l, v in zip(labels, p)) + f"  | {t}")
        return

    rows = [r for r in d[a.split] if r["text"] not in shot_texts]
    if a.limit:
        rows = rows[: a.limit]
    t0, out = time.time(), []
    for r in rows:
        s = predict(r["text"])
        pred = labels[max(range(len(labels)), key=lambda i: s[i])]
        out.append({"text": r["text"], "label": r["label"], "pred": pred})
    sec = round(time.time() - t0, 1)
    acc = sum(o["pred"] == o["label"] for o in out) / len(out)

    cm = {t: {p: 0 for p in labels} for t in labels}
    for o in out:
        cm[o["label"]][o["pred"]] += 1
    per = {lab: round(cm[lab][lab] / max(1, sum(cm[lab].values())), 3) for lab in labels}
    share = {lab: round(sum(o["pred"] == lab for o in out) / len(out), 3) for lab in labels}

    print(f"\n[{d['name']}] {a.split} {len(out)}건 · 모델 {model_id.split('/')[-1]} · 예시 {len(shots)}개"
          f"{' · 보정' if cf else ''} · {sec}초")
    print(f"정확도 {acc:.3f}   (찍기 수준 {1 / len(labels):.2f})")
    utils.print_table([{"정답 \\ 예측": t, **{p: cm[t][p] for p in labels}, "맞힌 비율": per[t]} for t in labels])
    print("예측 비율:", "  ".join(f"{k} {v:.2f}" for k, v in share.items()))
    wrong = [o for o in out if o["pred"] != o["label"]]
    for o in wrong[:6]:
        print(f"  틀림  정답 {o['label']} → 예측 {o['pred']}  {o['text'][:50]}")

    tag = a.tag or "기본"
    res = {"task": a.task, "split": a.split, "model": model_id, "shots": len(shots), "calibrate": bool(cf),
           "words": words, "instruction": instruction, "norm": a.norm, "tag": tag, "n": len(out), "sec": sec,
           "acc": round(acc, 3), "per_label": per, "pred_share": share, "confusion": cm, "wrong": wrong}
    utils.write_json(utils.result_path(f"llm_classify_{a.task}_{tag}.json"), res)

    board = utils.result_path("leaderboard.csv")
    new = not board.exists()
    with open(board, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["시각", "과제", "평가셋", "모델", "예시", "보정", "라벨단어", "태그", "정확도", "건수", "초"])
        w.writerow([utils.now(), a.task, a.split, model_id.split("/")[-1], len(shots), int(bool(cf)),
                    "기본" if not a.words else a.words, tag, round(acc, 3), len(out), sec])
    print(f"\n제출 줄 → {a.task} | {a.split} | {tag} | 예시 {len(shots)} | 보정 {int(bool(cf))} | 정확도 {acc:.3f}")

    import matplotlib.pyplot as plt
    utils.korean_font()
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(labels), 1.0 + 0.8 * len(labels)))
    m = [[cm[t][p] for p in labels] for t in labels]
    ax.imshow(m, cmap="Oranges")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, m[i][j], ha="center", va="center", fontsize=10,
                    color="white" if m[i][j] > max(map(max, m)) / 2 else "#222")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("예측"); ax.set_ylabel("정답")
    ax.set_title(f"{d['name']} · 정확도 {acc:.2f}", fontsize=10)
    fig.tight_layout()
    fp = utils.fig_path(f"llm_classify_{a.task}.png")
    fig.savefig(fp, dpi=130)
    print("그림:", fp)


if __name__ == "__main__":
    main()
