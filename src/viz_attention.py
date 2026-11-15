"""1·2회차: Attention 가중치 시각화.

  python src/viz_attention.py tiny --arch gru_attn --text "삼만 오천이백 원"       # 1회차: 직접 학습한 모델의 Cross-Attention
  python src/viz_attention.py tiny --arch transformer --text "이천이십육년 십일월 십육일"
  python src/viz_attention.py opus --text "결제했는데 아이템이 안 들어왔어요"      # 1회차: 사전학습 번역 모델(opus-mt)의 Cross-Attention
  python src/viz_attention.py self --text "서리칼날 공격력이 왜 줄었나요"          # 2회차: Decoder-only의 Self-Attention
  python src/viz_attention.py compare --text "결제했는데 아이템이 안 들어왔어요"   # 2회차 실습 1 ④: Cross vs Self 나란히

Attention 가중치를 받으려면 로딩할 때 attn_implementation="eager"가 필요하다(기본 SDPA는 가중치를 돌려주지 않음).
그림: results/fig/attn_<모드>.png
"""
from __future__ import annotations

import argparse

import torch

import config
import utils


def heatmap(ax, w, xlabels, ylabels, title):
    ax.imshow(w, aspect="auto", cmap="viridis")
    short = all(len(x) <= 2 for x in xlabels)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=0 if short else 90, fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    ax.set_xlabel("참조되는 쪽(Key)")
    ax.set_ylabel("만드는 쪽(Query)")
    ax.set_title(title, fontsize=10)


def tiny(arch: str, text: str):
    from tiny_seq2seq_train import load_trained, batchify, EOS
    model, vocab = load_trained(arch)
    src, _ = batchify([{"src": text, "tgt": ""}], vocab, "cpu")
    pred = model.greedy(src)[0].tolist()
    out = vocab.decode(pred)
    n_out = pred.index(EOS) + 1 if EOS in pred else len(pred)
    w = model.last_attn
    if w is None:
        raise SystemExit("gru(어텐션 없음) 구조는 볼 Attention이 없다: 고정 길이 벡터 하나만 넘기기 때문")
    w = w[0, :n_out, :len(text)].numpy()
    ylabels = list(out) + ["<eos>"]
    return w, list(text), ylabels[:n_out], f"{arch} Cross-Attention: {text} -> {out}"


def opus(text: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.ENC_DEC_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.ENC_DEC_MODEL, attn_implementation="eager").eval()
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=64, num_beams=1)
        out = model(**enc, decoder_input_ids=gen[:, :-1], output_attentions=True)   # 생성 결과로 한 번 더 통과
    w = out.cross_attentions[-1][0].mean(0).numpy()            # 마지막 층, 헤드 평균 (출력, 입력)
    xs = tok.convert_ids_to_tokens(enc["input_ids"][0])
    ys = tok.convert_ids_to_tokens(gen[0, 1:])
    return w, xs, ys, f"opus-mt Cross-Attention: {tok.decode(gen[0], skip_special_tokens=True)}"


def self_attn(text: str, layer: int = -1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.SMALL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(config.SMALL_MODEL_ID, attn_implementation="eager", dtype=torch.float32).eval()
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    w = out.attentions[layer][0].mean(0).numpy()                # (토큰, 토큰): 대각선 위쪽이 0(미래를 못 봄)
    toks = [tok.decode([i]) for i in enc["input_ids"][0]]
    return w, toks, toks, f"{config.SMALL_MODEL_ID} Self-Attention(층 {layer}, 헤드 평균)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tiny", "opus", "self", "compare"])
    ap.add_argument("--text", default="결제했는데 아이템이 안 들어왔어요")
    ap.add_argument("--arch", default="transformer", choices=["gru_attn", "transformer"])
    ap.add_argument("--layer", type=int, default=-1)
    a = ap.parse_args()
    utils.korean_font()
    import matplotlib.pyplot as plt
    if a.mode == "compare":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (w, xs, ys, title) in zip(axes, (opus(a.text), self_attn(a.text, a.layer))):
            heatmap(ax, w, xs, ys, title)
    else:
        w, xs, ys, title = {"tiny": lambda: tiny(a.arch, a.text), "opus": lambda: opus(a.text),
                            "self": lambda: self_attn(a.text, a.layer)}[a.mode]()
        fig, ax = plt.subplots(figsize=(max(5, 0.35 * len(xs)), max(3.5, 0.3 * len(ys))))
        heatmap(ax, w, xs, ys, title)
    fig.tight_layout()
    out = utils.fig_path(f"attn_{a.mode}{'_' + a.arch if a.mode == 'tiny' else ''}.png")
    fig.savefig(out, dpi=150)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
