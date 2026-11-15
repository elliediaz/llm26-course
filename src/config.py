"""과정 공통 설정.

모델 이름, 생성 설정, 경로는 이 파일 한 곳에서만 바꾼다. 다른 스크립트는 값을 직접 쓰지 않고
`import config`로 가져다 쓴다.

회차별로 학생이 고치는 곳
  1회차  EMBED_MODEL 교체 실험(선택)
  2회차  GEN_SERVICE 확정, CONTEXT_MAX_TOKENS 근거 측정(도전과제)
  4회차  MODEL_ID, QUANT  ← 차승희 박사 회차에서 팀 모델 선정 결과로 교체
  5회차  DOMAIN 확정
  6회차  VECTOR_DB, RAG_TOP_K, CHUNK 설정
  7회차  LORA 설정
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 저장소 위치
ROOT = Path(__file__).resolve().parents[1]          # Pod: /workspace/course

# ---------------------------------------------------------------- 팀 도메인
# G: 게임 운영 AI · L: 알바·자취 생활법률 · S: 스포츠·e스포츠 2026 시즌
DOMAIN = os.environ.get("COURSE_DOMAIN", "G")

# ---------------------------------------------------------------- 모델 (2026-10 기준)
MODEL_ID = os.environ.get("COURSE_MODEL_ID", "skt/A.X-4.0-Light")   # 메인 LLM. 4회차 선정 결과로 교체
SMALL_MODEL_ID = "Qwen/Qwen3-0.6B"                  # 2회차 Attention·KV Cache 관찰용 소형 Decoder(Instruct)
BASE_MODEL_ID = "Qwen/Qwen3-0.6B-Base"              # 2회차 정렬 전(Base) 모델 비교·문맥 내 학습 확인용
MLM_MODEL_ID = "FacebookAI/xlm-roberta-base"        # 2회차 빈칸 맞히기(MLM) 확인용 인코더
ENC_DEC_MODEL = "Helsinki-NLP/opus-mt-ko-en"        # 1회차 Encoder-Decoder 관찰용(번역)
EMBED_MODEL = "nlpai-lab/KURE-v1"                   # 1·6회차 임베딩(백업: BAAI/bge-m3)
RERANK_MODEL = "dragonkue/bge-reranker-v2-m3-ko"    # 6회차 도전과제(백업: BAAI/bge-reranker-v2-m3)

QUANT = os.environ.get("COURSE_QUANT", "bf16")      # "bf16" | "nf4"(4-bit, QLoRA·저메모리 추론)

# Chat Template 추가 인자. Qwen3 계열은 생각 모드를 끈다(다른 모델 템플릿에서는 무시됨)
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# ---------------------------------------------------------------- 생성 설정
# 평가용: 5~7회차의 모든 측정은 이 값으로 고정한다. 바꾸면 이전 측정과 비교할 수 없다.
GEN_EVAL = {"do_sample": False, "max_new_tokens": 256}

# 서비스용: 2회차 생성 설정 비교표로 팀이 확정한다. 8~10회차 Agent·Service에서 쓴다.
GEN_SERVICE = {"do_sample": True, "temperature": 0.3, "top_p": 0.9, "max_new_tokens": 256}
GEN_SERVICE_REASON = "(2회차에 한 줄로 기록)"

SEED = 42

# ---------------------------------------------------------------- 6회차 RAG
VECTOR_DB = "chroma"                # "chroma" | "faiss"
CHUNK = {
    "fixed": {"size": 500, "overlap": 100},        # 글자 수 기준
    "sentence": {"max_chars": 500},                # 문장·문단 경계로 자르고 max_chars까지 묶음
}
CHUNK_STRATEGY = "sentence"         # 6회차 검색 품질 비교 후 선택한 값으로 바꾼다
RAG_TOP_K = 4
CONTEXT_MAX_TOKENS = 1500           # 검색 결과를 프롬프트에 넣을 때의 상한(2회차 도전과제 ② 측정값 근거)

# ---------------------------------------------------------------- 7회차 LoRA
LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 2e-4,
    "epochs": 2,
    "batch_size": 8,
    "grad_accum": 1,
    "max_length": 768,              # 7회차 학습이 40분 안에 끝나도록 정한 값
    "max_minutes": 35,              # 시간 초과 시 그 시점까지 학습한 Adapter 저장 후 종료
}

# ---------------------------------------------------------------- 경로
PATHS = {
    "data": ROOT / "data",
    "common": ROOT / "data" / "common",
    "results": ROOT / "results",
    "figs": ROOT / "results" / "fig",
    "index": ROOT / "index",
    "adapters": ROOT / "adapters",
    "runs": ROOT / "runs",          # 1회차 소규모 학습 체크포인트 등(공유 대상 아님)
}


def data_dir(domain: str | None = None) -> Path:
    """팀 도메인 데이터 폴더(data/G 등)."""
    return PATHS["data"] / (domain or DOMAIN)


def summary() -> dict:
    """결과 파일에 함께 남길 설정 요약. 측정값을 나중에 해석할 때 근거가 된다."""
    return {
        "domain": DOMAIN,
        "model_id": MODEL_ID,
        "quant": QUANT,
        "gen_eval": GEN_EVAL,
        "embed_model": EMBED_MODEL,
        "chunk_strategy": CHUNK_STRATEGY,
        "rag_top_k": RAG_TOP_K,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
    for k, v in PATHS.items():
        print(f"{k:8s} {v}")
