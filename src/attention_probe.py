"""2회차 미션 M2: 어텐션 탐정. '그것'이 무엇을 가리키는지 모델 안을 들여다봄

  python src/attention_probe.py                       # 지시어 문장 12개: 층별 어텐션 + 모델에게 직접 묻기
  python src/attention_probe.py --polysemy            # 같은 '배'가 문맥에 따라 다른 벡터가 되는지
  python src/attention_probe.py --text "고양이가 쥐를 쫓았다. 그것은 배가 고팠다." --target 그것 --cands 고양이,쥐

보는 것 세 가지
  ① 어텐션: 지시어 자리와 문장 끝 단어 자리에서, 후보 두 낱말로 가는 어텐션 비율(층마다, 헤드 평균)
  ② 묻기: 문장 뒤에 "'그것'이 가리키는 것은?"을 붙이고 후보 낱말의 확률을 비교
  ③ 다의어: '배'(과일·선박·신체) 자리의 은닉 벡터끼리 코사인 유사도. 0층(임베딩)과 깊은 층을 비교
주의: 이 모델(Decoder-only)은 앞 토큰만 봄. "too big"의 big은 it보다 뒤에 있으므로 it 자리에서는 아직 모름
결과: results/attention_probe.json, results/fig/attention_probe.png, results/fig/polysemy.png
"""
from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import config
import utils

DATA = config.ROOT / "data" / "s02" / "coref_sentences.json"
PUNCT = set(".,!?。 \n")
SINK = "\n"        # 첫 토큰은 어텐션이 몰리는 자리(어텐션 싱크). 후보 낱말이 첫 토큰이 되지 않도록 줄바꿈 하나를 앞에 붙임


def is_en(text: str) -> bool:
    return bool(re.match(r"[A-Za-z]", text))


def char_span(text: str, word: str, after: int = 0) -> tuple[int, int]:
    pat = rf"\b{re.escape(word)}\b" if is_en(word) else re.escape(word)
    m = re.compile(pat).search(text, after)
    if not m:
        raise ValueError(f"'{word}'를 문장에서 찾지 못함: {text}")
    return m.start(), m.end()


def tok_idx(offsets, span) -> list[int]:
    return [i for i, (s, e) in enumerate(offsets) if s < span[1] and e > span[0]]


def load(model_id: str):
    hf_logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(model_id)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    return tok, model


@torch.no_grad()
def probe(tok, model, text: str, target: str, cands: list[str]) -> dict:
    text = SINK + text
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    c_spans = [char_span(text, c) for c in cands]
    t_span = char_span(text, target, after=max(s[1] for s in c_spans))
    c_tok = [tok_idx(offs, s) for s in c_spans]
    t_tok = tok_idx(offs, t_span)
    last = max(i for i, (s, e) in enumerate(offs) if text[s:e].strip() and not set(text[s:e]) <= PUNCT)
    att = model(torch.tensor([ids], device=model.device), output_attentions=True).attentions     # 층마다 [1, 헤드, T, T]
    A = torch.stack([a[0].float().cpu() for a in att])                                     # [층, 헤드, T, T]

    def share(q_idx):
        to = torch.stack([A[:, :, q_idx][:, :, :, c].sum(-1).mean(-1) for c in c_tok], -1)   # [층, 헤드, 후보]
        return to / to.sum(-1, keepdim=True).clamp_min(1e-9)
    s_target, s_last = share(t_tok), share([last])
    return {"tokens": [tok.decode([i]) for i in ids], "target_tok": t_tok, "last_tok": last, "cand_tok": c_tok,
            "from_target": s_target.mean(1), "from_last": s_last.mean(1), "heads_last": s_last, "attn": A}


@torch.no_grad()
def ask(tok, model, text: str, target: str, cands: list[str]) -> list[float]:
    if is_en(text):
        q = f"{text}\nQuestion: In the sentence above, what does '{target}' refer to?\nAnswer: The"
    else:
        q = f"{text}\n질문: 위 문장에서 '{target}'이 가리키는 것은?\n답:"
    base = tok(q, add_special_tokens=False).input_ids
    out = []
    for c in cands:
        cid = tok(" " + c, add_special_tokens=False).input_ids
        lp = model(torch.tensor([base + cid], device=model.device)).logits[0].float().log_softmax(-1)
        out.append(sum(lp[len(base) - 1 + i, t].item() for i, t in enumerate(cid)))
    return out


def run_coref(tok, model):
    items = utils.read_json(DATA)["coref"]
    rows, curves, head_hits = [], [], None
    for it in items:
        r = probe(tok, model, it["text"], it["target"], it["candidates"])
        k = it["candidates"].index(it["answer"])
        ft, fl = r["from_target"][:, k], r["from_last"][:, k]
        hits = (r["heads_last"].argmax(-1) == k).int()                    # [층, 헤드] 정답 쪽을 더 본 헤드
        head_hits = hits if head_hits is None else head_hits + hits
        a = ask(tok, model, it["text"], it["target"], it["candidates"])
        c0 = it["candidates"][0]
        rows.append({"문장": it["text"][:40], "정답": it["answer"],
                     "지시어 자리": round(r["from_target"][:, 0].mean().item(), 2),
                     "문장 끝": round(r["from_last"][:, 0].mean().item(), 2),
                     "묻기 답": it["candidates"][int(a[1] > a[0])]})
        curves.append((it, r["from_target"][:, 0].tolist(), r["from_last"][:, 0].tolist()))
    utils.print_table(rows)
    ask_acc = sum(r["묻기 답"] == r["정답"] for r in rows) / len(rows)
    L, H = head_hits.shape
    best = sorted(((head_hits[l, h].item(), l, h) for l in range(L) for h in range(H)), reverse=True)[:5]
    print(f"\n묻기 정답률 {ask_acc:.2f} ({len(rows)}문장, 찍기 0.50)")
    print("문장 끝 단어에서 정답 후보를 더 많이 본 헤드(12문장 중 맞힌 수):",
          ", ".join(f"{l}층 {h}번 {n}개" for n, l, h in best))
    print("숫자 읽는 법: 앞 후보(trophy·트로피·cat·고양이 …) 쪽으로 간 어텐션 비율, 층·헤드 평균. 0.5보다 크면 앞 후보를 더 봄")
    print("            big·small 쌍에서 '지시어 자리' 값이 똑같으면, 그 자리에서는 뒤 단어를 아직 못 본 것(앞만 보는 마스크)")
    return rows, curves, ask_acc, best


def run_polysemy(tok, model):
    items = utils.read_json(DATA)["polysemy"]
    vecs = {}
    with torch.no_grad():
        for it in items:
            enc = tok(it["text"], return_offsets_mapping=True, add_special_tokens=False)
            idx = tok_idx(enc["offset_mapping"], char_span(it["text"], it["word"]))
            hs = model(torch.tensor([enc["input_ids"]], device=model.device), output_hidden_states=True).hidden_states
            for layer in (0, len(hs) // 2, len(hs) - 1):
                vecs.setdefault(layer, []).append(hs[layer][0, idx].mean(0).float().cpu())
    res = {}
    for layer, vs in vecs.items():
        V = torch.nn.functional.normalize(torch.stack(vs), dim=-1)
        res[layer] = (V @ V.T).tolist()
    for word in dict.fromkeys(it["word"] for it in items):
        ix = [i for i, it in enumerate(items) if it["word"] == word]
        print(f"\n'{word}' 자리 벡터의 코사인 유사도(같은 뜻끼리 평균 / 다른 뜻끼리 평균)")
        for layer, M in res.items():
            same = [M[i][j] for i in ix for j in ix if i < j and items[i]["sense"] == items[j]["sense"]]
            diff = [M[i][j] for i in ix for j in ix if i < j and items[i]["sense"] != items[j]["sense"]]
            print(f"  {layer:2d}층  같은 뜻 {sum(same) / len(same):.3f} / 다른 뜻 {sum(diff) / len(diff):.3f}")
    return items, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.SMALL_MODEL_ID)
    ap.add_argument("--polysemy", action="store_true", help="다의어 벡터 비교만")
    ap.add_argument("--text")
    ap.add_argument("--target", help="지시어(예: 그것, it)")
    ap.add_argument("--cands", help="후보 낱말 두 개 이상, 쉼표 구분")
    ap.add_argument("--layer", type=int, default=-1, help="--text 모드에서 그림으로 볼 층(기본: 마지막 층)")
    a = ap.parse_args()
    tok, model = load(a.model)
    import matplotlib.pyplot as plt
    utils.korean_font()

    if a.text:
        cands = [c.strip() for c in a.cands.split(",")]
        r = probe(tok, model, a.text, a.target, cands)
        print("층별 어텐션 비율(헤드 평균): 지시어 자리 / 문장 끝 단어 자리")
        for l in range(r["from_last"].shape[0]):
            print(f"  {l:2d}층  " + "  ".join(f"{c} {r['from_target'][l, i]:.2f}/{r['from_last'][l, i]:.2f}" for i, c in enumerate(cands)))
        a_ = ask(tok, model, a.text, a.target, cands)
        print("묻기:", " · ".join(f"{c} {p:.2f}" for c, p in zip(cands, torch.tensor(a_).softmax(0).tolist())))
        A = r["attn"][a.layer].mean(0)                                     # 헤드 평균 [T, T]
        toks = [t.replace("\n", "↵") for t in r["tokens"]]
        fig, ax = plt.subplots(figsize=(0.45 * len(toks) + 2, 0.45 * len(toks) + 1.5))
        ax.imshow(A[:, :].numpy(), cmap="viridis")
        ax.set_xticks(range(len(toks)), toks, rotation=90, fontsize=8)
        ax.set_yticks(range(len(toks)), toks, fontsize=8)
        ax.set_xlabel("참조되는 쪽(Key)"); ax.set_ylabel("만드는 쪽(Query)")
        ax.set_title(f"{a.layer if a.layer >= 0 else r['attn'].shape[0] + a.layer}층 어텐션(헤드 평균)", fontsize=10)
        fig.tight_layout()
        out = utils.fig_path("attention_probe_text.png")
        fig.savefig(out, dpi=130)
        print("그림:", out)
        return

    if a.polysemy:
        items, res = run_polysemy(tok, model)
        layers = list(res)
        fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.2))
        labels = [f"{it['word']}·{it['sense']}" for it in items]
        for ax, layer in zip(axes, layers):
            ax.imshow(res[layer], vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_title(f"{layer}층" + (" (임베딩)" if layer == 0 else ""), fontsize=10)
        fig.suptitle("같은 낱말, 다른 문맥: 은닉 벡터 코사인 유사도", fontsize=11)
        fig.tight_layout()
        out = utils.fig_path("polysemy.png")
        fig.savefig(out, dpi=130)
        utils.write_json(utils.result_path("polysemy.json"), {"items": items, "cosine": {str(k): v for k, v in res.items()}})
        print("그림:", out)
        return

    rows, curves, ask_acc, best = run_coref(tok, model)
    utils.write_json(utils.result_path("attention_probe.json"),
                     {"model": a.model, "rows": rows, "ask_acc": round(ask_acc, 3),
                      "best_heads": [{"layer": l, "head": h, "hits": n} for n, l, h in best],
                     "curves": [{"text": it["text"], "first_cand": it["candidates"][0], "answer": it["answer"],
                                 "from_target": [round(v, 4) for v in ft], "from_last": [round(v, 4) for v in fl]}
                                for it, ft, fl in curves]})
    pick = [c for c in curves if "trophy" in c[0]["text"] or "트로피" in c[0]["text"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, lang in zip(axes, ["en", "ko"]):
        sub = [c for c in pick if is_en(c[0]["text"]) == (lang == "en")]
        for (it, ft, fl), color in zip(sub, ["#2b5c9e", "#c0622b"]):
            key = "too big" if "big" in it["text"] else "too small" if "small" in it["text"] else \
                  "컸기" if "컸기" in it["text"] else "작았기"
            ax.plot(fl, color=color, lw=2, label=f"'{key}' 문장 · 문장 끝 자리")
            ax.plot(ft, "--", color=color, lw=1.2, label=f"'{key}' 문장 · 지시어 자리")
        ax.axhline(0.5, color="#999", lw=0.8)
        ax.set_xlabel("층"); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="lower left")
        ax.set_title("it → trophy 쪽 비율" if lang == "en" else "그것 → 트로피 쪽 비율", fontsize=10)
    axes[0].set_ylabel("앞 후보 쪽 어텐션 비율")
    fig.suptitle("점선 두 개가 겹치면 그 자리에서는 아직 뒤 단어를 못 본 것, 실선이 갈라지면 두 문장을 구별한 것", fontsize=9)
    fig.tight_layout()
    out = utils.fig_path("attention_probe.png")
    fig.savefig(out, dpi=130)
    print("그림:", out)


if __name__ == "__main__":
    main()
