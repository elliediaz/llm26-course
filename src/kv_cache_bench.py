"""2회차 실습 2 ⑧: KV Cache가 생성 속도와 메모리에 미치는 영향 측정 + 도전과제 ②(입력 길이별 VRAM·지연).

  python src/kv_cache_bench.py                         # 소형 모델(Qwen3-0.6B), 입력 길이 4단계 × 캐시 유/무
  python src/kv_cache_bench.py --lengths 512 2048 8192 --new 64 --model skt/A.X-4.0-Light   # 큰 모델·긴 입력

확인할 것
  - 캐시 없음: 새 토큰마다 앞 토큰 전체를 다시 계산 → 입력이 길수록 느려짐
  - 캐시 있음: 빠르지만 K·V를 층마다 들고 있음 → 크기 = 2 × 층 × KV헤드 × 헤드차원 × 길이 × 2바이트
  - GQA(Q 헤드보다 KV 헤드가 적음)는 이 캐시 크기를 줄이는 장치
  - 이 그래프가 6회차에서 검색 결과를 프롬프트에 얼마나 넣을지(CONTEXT_MAX_TOKENS) 정하는 근거가 된다
결과: results/kv_cache.json, results/fig/kv_cache.png
"""
from __future__ import annotations

import argparse

import torch

import config
import model_utils
import utils


def cache_bytes(cache) -> int:
    total = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:                         # transformers 5.x DynamicCache
        for L in layers:
            for t in (getattr(L, "keys", None), getattr(L, "values", None)):
                if t is not None:
                    total += t.numel() * t.element_size()
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.SMALL_MODEL_ID)
    ap.add_argument("--lengths", type=int, nargs="*", default=[128, 512, 1024, 2048])
    ap.add_argument("--new", type=int, default=64, help="생성 토큰 수")
    a = ap.parse_args()
    model, tok = model_utils.load_llm(a.model, quant="bf16")
    cfg = model.config
    n_layer, n_kv = cfg.num_hidden_layers, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    elem = torch.finfo(next(model.parameters()).dtype).bits // 8
    print(f"층 {n_layer} · Q 헤드 {cfg.num_attention_heads} · KV 헤드 {n_kv} · 헤드 차원 {head_dim}")
    base = "결제했는데 아이템이 안 들어왔어요. 정기 점검은 매주 수요일입니다. "
    rows = []
    for L in a.lengths:
        ids = tok(base * (L // 10 + 1), return_tensors="pt")["input_ids"][:, :L].to(model.device)
        for use_cache in (True, False):
            utils.vram_reset()
            with utils.Timer() as t, torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.new,
                                     min_new_tokens=a.new, do_sample=False, use_cache=use_cache,
                                     return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
            measured = cache_bytes(out.past_key_values) if use_cache and out.past_key_values is not None else 0
            theory = 2 * n_layer * n_kv * head_dim * (L + a.new) * elem
            rows.append({"입력 길이": L, "KV Cache": "사용" if use_cache else "미사용", "시간(s)": round(t.sec, 2),
                         "토큰/초": round(a.new / t.sec, 1), "VRAM 최대(GB)": utils.vram_peak_gb(),
                         "캐시 실측(MB)": round(measured / 1024**2, 1) if use_cache else None,
                         "캐시 이론(MB)": round(theory / 1024**2, 1)})
            print(f"  L={L:5d} cache={'O' if use_cache else 'X'} {t.sec:6.2f}s")
    print()
    utils.print_table(rows)
    utils.write_json(utils.result_path("kv_cache.json"), {"model": a.model, "new_tokens": a.new, "rows": rows,
                                                          "layers": n_layer, "kv_heads": n_kv, "head_dim": head_dim})
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for use in ("사용", "미사용"):
        r = [x for x in rows if x["KV Cache"] == use]
        axes[0].plot([x["입력 길이"] for x in r], [x["시간(s)"] for x in r], marker="o", label=f"캐시 {use}")
        if r[0]["VRAM 최대(GB)"] is not None:
            axes[1].plot([x["입력 길이"] for x in r], [x["VRAM 최대(GB)"] for x in r], marker="o", label=f"캐시 {use}")
    axes[0].set(xlabel="입력 길이(토큰)", ylabel=f"{a.new}토큰 생성 시간(s)", title="생성 시간")
    axes[1].set(xlabel="입력 길이(토큰)", ylabel="VRAM 최대(GB)", title="메모리")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(utils.fig_path("kv_cache.png"), dpi=150)
    print(f"그림: {utils.fig_path('kv_cache.png')}")


if __name__ == "__main__":
    main()
