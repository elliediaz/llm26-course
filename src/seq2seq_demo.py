"""1회차 도전·복습: 세 구조를 같은 입력으로 돌려 보고 구조별 모델 비교표를 만든다.

  python src/seq2seq_demo.py                 # Encoder · Encoder-Decoder · Decoder-only 비교표
  python src/seq2seq_demo.py --only encdec   # 번역(opus-mt-ko-en)만
  python src/seq2seq_demo.py --text "점검이 2시간 연장되었습니다"

확인할 것
  Encoder         입력 → 벡터 하나(출력 길이 개념이 없음). 검색·분류에 쓴다
  Encoder-Decoder 입력 길이와 출력 길이가 따로 논다(번역). 디코더가 인코더 출력을 참조(Cross-Attention)
  Decoder-only    입력과 출력이 한 줄로 이어진다. 지시문으로 과업을 바꾼다(번역도 생성 문제가 됨)
결과: results/structure_compare.json, results/structure_compare.md
"""
from __future__ import annotations

import argparse

import torch

import config
import utils

SENTENCES = [
    "결제했는데 아이템이 안 들어왔어요.",
    "정기 점검은 매주 수요일 오전 6시부터 8시까지 진행됩니다.",
    "지난 패치 이후 서리칼날의 공격력이 120에서 105로 낮아졌습니다.",
]


def gpu_mem() -> float | None:
    return round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None


def run_encoder(texts):
    import model_utils
    m = model_utils.load_embedder()
    with utils.Timer() as t:
        v = m.encode(texts, normalize_embeddings=True)
    return {"구조": "Encoder-only", "모델": config.EMBED_MODEL,
            "파라미터": sum(p.numel() for p in m.parameters()),
            "입력→출력": f"문장 {len(texts)}개 → 벡터 {v.shape}", "출력 예": f"[{', '.join(f'{x:.3f}' for x in v[0][:4])}, …]",
            "시간(s)": round(t.sec, 3), "용도": "검색·분류·임베딩(1·6회차)"}


def run_encdec(texts):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.ENC_DEC_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForSeq2SeqLM.from_pretrained(config.ENC_DEC_MODEL).to(dev).eval()
    enc = tok(texts, return_tensors="pt", padding=True).to(dev)
    with utils.Timer() as t, torch.no_grad():
        gen = m.generate(**enc, max_new_tokens=64, num_beams=4)
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    lens = [(int(a.sum()), int((g != tok.pad_token_id).sum()) - 1) for a, g in zip(enc["attention_mask"], gen)]
    for s, o, (i, j) in zip(texts, outs, lens):
        print(f"  [{i:2d}→{j:2d} 토큰] {s}\n           → {o}")
    return {"구조": "Encoder-Decoder", "모델": config.ENC_DEC_MODEL, "파라미터": sum(p.numel() for p in m.parameters()),
            "입력→출력": "토큰 수 " + ", ".join(f"{i}→{j}" for i, j in lens), "출력 예": outs[0],
            "시간(s)": round(t.sec, 3), "용도": "번역·요약·음성 인식"}


def run_decoder(texts):
    import model_utils
    m, tok = model_utils.load_llm(config.SMALL_MODEL_ID, quant="bf16")
    outs, tsum = [], 0.0
    for s in texts:
        msgs = [{"role": "user", "content": f"다음 문장을 영어로 번역해. 번역문만 출력해.\n{s}"}]
        g = model_utils.generate(m, tok, msgs, {"do_sample": False, "max_new_tokens": 64})
        outs.append(g)
        tsum += g["latency"]
        print(f"  [{g['n_in']:2d}→{g['n_out']:2d} 토큰] {s}\n           → {g['text']}")
    return {"구조": "Decoder-only", "모델": config.SMALL_MODEL_ID, "파라미터": sum(p.numel() for p in m.parameters()),
            "입력→출력": "토큰 수 " + ", ".join(f"{g['n_in']}→{g['n_out']}" for g in outs) + " (입력에 지시문·템플릿 포함)",
            "출력 예": outs[0]["text"], "시간(s)": round(tsum, 3), "용도": "생성·범용 LLM(2~10회차)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["encoder", "encdec", "decoder"])
    ap.add_argument("--text", action="append", help="입력 문장(여러 번 지정 가능)")
    a = ap.parse_args()
    texts = a.text or SENTENCES
    rows = []
    for key, fn in (("encoder", run_encoder), ("encdec", run_encdec), ("decoder", run_decoder)):
        if a.only and a.only != key:
            continue
        with utils.section(key):
            utils.vram_reset()
            r = fn(texts)
            r["VRAM 최대(GB)"] = utils.vram_peak_gb()
            rows.append(r)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print()
    md = utils.print_table(rows, ["구조", "모델", "파라미터", "입력→출력", "시간(s)", "VRAM 최대(GB)", "용도"])
    utils.write_json(utils.result_path("structure_compare.json"), {"measured_at": utils.now(), "texts": texts, "rows": rows})
    utils.result_path("structure_compare.md").write_text("# 구조별 모델 비교표\n\n" + md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
