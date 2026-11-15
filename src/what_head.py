"""1회차 핵심개념 시연·복습: 수업에 쓰는 모델의 '출력층'만 보고 분류인지 회귀(점수)인지 판정한다.

  python src/what_head.py            # config.py의 모델들(설정 파일만 읽음, 가중치는 불러오지 않아 몇 초면 끝남)

모델 구조 이름과 출력 크기로 무엇을 하는 모델인지 판정할 수 있다.
  …ForCausalLM / …MTModel   출력 = 어휘 크기 → 매 토큰 '어휘 중 하나'를 고르는 분류(생성 = 분류 반복)
  …ForSequenceClassification 출력 = num_labels 개 → 1개면 점수(회귀형), 2개 이상이면 클래스 분류
  임베딩(Encoder)            출력 = 벡터 → 그 자체는 표현. 두 벡터의 코사인이 연속 점수, 가장 높은 것을 고르면 분류
결과: results/what_head.md
"""
from __future__ import annotations

from transformers import AutoConfig

import config
import utils

MODELS = [
    ("메인 LLM(5~10회차)", config.MODEL_ID),
    ("소형 Decoder(2회차)", config.SMALL_MODEL_ID),
    ("번역 Enc-Dec(1회차)", config.ENC_DEC_MODEL),
    ("빈칸 맞히기 MLM(2회차)", config.MLM_MODEL_ID),
    ("임베딩(1·6회차)", config.EMBED_MODEL),
    ("재순위화(6회차)", config.RERANK_MODEL),
]


def judge(cfg) -> tuple[str, str]:
    arch = (cfg.architectures or ["?"])[0]
    V = getattr(cfg, "vocab_size", None)
    if arch.endswith("ForCausalLM") or arch.endswith("MTModel") or arch.endswith("ConditionalGeneration"):
        return arch, f"출력 {V:,}개(어휘) → 다음 토큰 분류, 생성 = 분류 반복"
    if arch.endswith("ForMaskedLM"):
        return arch, f"출력 {V:,}개(어휘) → 빈칸 자리의 토큰 분류"
    if arch.endswith("ForSequenceClassification"):
        n = getattr(cfg, "num_labels", None) or len(getattr(cfg, "id2label", {}) or {})
        return arch, ("출력 1개 → 관련도 점수(회귀형), 기준값을 넘으면 '관련'으로 판정(이진 분류)" if n == 1
                      else f"출력 {n}개 → {n}개 클래스 분류")
    hid = getattr(cfg, "hidden_size", None)
    return arch, f"출력 = {hid}차원 벡터(표현) → 코사인 점수(연속값) + 가장 높은 것 고르기(분류)"


def main() -> None:
    rows = []
    for role, mid in MODELS:
        try:
            cfg = AutoConfig.from_pretrained(mid)
            arch, verdict = judge(cfg)
        except Exception as e:  # noqa: BLE001  (시딩되지 않은 모델은 건너뜀)
            arch, verdict = "-", f"설정을 읽지 못함({type(e).__name__})"
        rows.append({"역할": role, "모델": mid, "구조": arch, "판정": verdict})
    table = utils.print_table(rows)
    path = utils.result_path("what_head.md")
    path.write_text("# 출력층으로 본 모델의 정체\n\n" + table + "\n", encoding="utf-8")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
