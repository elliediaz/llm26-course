"""2회차 실습 1 ①: 토크나이저 해부. 같은 뜻의 한국어·영어 문장이 몇 토큰이 되는지, 어떻게 쪼개지는지 본다.

  python src/inspect_tokenizer.py                           # 기본 문장 세트 × 모델 3종 토큰 수 비교
  python src/inspect_tokenizer.py --text "너프 먹은 서리칼날"  # 한 문장 분할 단위와 Token ID 역변환

왜 중요한가
  - 토큰 수 = 비용·속도·컨텍스트 한도. 같은 뜻이라도 한국어가 토큰을 더 쓰면 그만큼 비싸고 길게 못 넣는다
  - 어휘집 밖 단어(신조어·은어)는 더 잘게 쪼개진다(서브워드 토큰화가 OOV를 없앤 방식)
결과: results/tokenizer_compare.json
"""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

import config
import utils

MODELS = [config.MODEL_ID, config.SMALL_MODEL_ID, config.EMBED_MODEL]


def pieces(tok, text: str) -> list[str]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.decode([i]) for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text")
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    toks = {}
    for m in a.models:
        try:
            toks[m] = AutoTokenizer.from_pretrained(m)
        except Exception as e:
            print(f"  (건너뜀) {m}: {e.__class__.__name__}")
    if a.text:
        for m, tok in toks.items():
            ids = tok(a.text, add_special_tokens=False)["input_ids"]
            print(f"\n[{m}] 어휘 크기 {len(tok):,} · {len(ids)}토큰")
            print("  ID     :", ids)
            print("  청크   :", " | ".join(repr(p) for p in pieces(tok, a.text)))
            print("  역변환 :", tok.decode(ids))
        return
    sents = utils.read_json(config.PATHS["common"] / "sentences.json")
    rows = []
    for s in sents:
        for lang in ("ko", "en", "ko_para"):
            r = {"문장": s["id"], "구분": {"ko": "한국어", "en": "영어", "ko_para": "한국어(바꿔 말하기)"}[lang],
                 "글자 수": len(s[lang])}
            for m, tok in toks.items():
                r[m.split("/")[-1]] = len(tok(s[lang], add_special_tokens=False)["input_ids"])
            rows.append(r)
    utils.print_table(rows)
    print("\n글자당 토큰(한국어 평균):")
    for m in toks:
        k = m.split("/")[-1]
        ko = [r for r in rows if r["구분"] == "한국어"]
        print(f"  {k:25s} {sum(r[k] for r in ko) / sum(r['글자 수'] for r in ko):.2f}")
    utils.write_json(utils.result_path("tokenizer_compare.json"), {"models": list(toks), "rows": rows})


if __name__ == "__main__":
    main()
