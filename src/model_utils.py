"""모델 로딩과 생성 도우미 (transformers 5.x 기준).

transformers 4.x와 달라진 점
  - 자료형은 dtype=, 4-bit는 quantization_config=BitsAndBytesConfig(...)
  - Attention 가중치를 보려면 로딩할 때 attn_implementation="eager"
  - Chat Template은 return_dict=True로 받아 model.generate(**inputs)
"""
from __future__ import annotations

import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()       # 가중치 로딩 진행 막대가 로그를 덮지 않게 함

import config
import utils


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(model_id: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_id or config.MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_llm(model_id: str | None = None, quant: str | None = None, attn: str | None = None,
             adapter: str | None = None):
    """Decoder-only LLM과 토크나이저를 불러온다.

    quant  "bf16"(기본) | "nf4"(4-bit, GPU 전용)
    attn   None(기본 SDPA) | "eager"(Attention 가중치 관찰용)
    adapter  7회차 LoRA Adapter 폴더. 주면 본체 위에 얹는다
    """
    model_id = model_id or config.MODEL_ID
    quant = quant or config.QUANT
    kwargs: dict = {"dtype": default_dtype()}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if attn:
        kwargs["attn_implementation"] = attn
    if quant == "nf4":
        if not torch.cuda.is_available():
            raise RuntimeError("nf4(4-bit) 로딩은 GPU 에서만 됩니다. config.QUANT='bf16'으로 바꾸세요.")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    t0 = time.perf_counter()
    tok = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    dt = str(next(model.parameters()).dtype).replace("torch.", "")    # CPU 서버에서는 bf16 대신 float32로 올라감
    print(f"[모델] {model_id} ({quant if quant != 'bf16' else dt}{', +adapter' if adapter else ''}) 로딩 {time.perf_counter() - t0:.1f}s"
          + (f", VRAM {torch.cuda.memory_allocated() / 1024**3:.1f}GB" if torch.cuda.is_available() else ""))
    return model, tok


def chat_inputs(tok, messages: list[dict], use_template: bool = True):
    """메시지 목록 → 모델 입력.

    use_template=False는 5회차 비교 실험용(템플릿 없이 내용만 이어 붙임). 실제 사용에서는 쓰지 않는다.
    """
    if use_template:
        return tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", **config.CHAT_TEMPLATE_KWARGS)
    text = "\n\n".join(m["content"] for m in messages) + "\n\n"
    return tok(text, return_tensors="pt")


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> str:
    """Qwen3 계열이 남기는 빈 생각 블록을 지운다."""
    return _THINK.sub("", text).strip()


@torch.inference_mode()
def generate(model, tok, messages: list[dict], gen: dict | None = None, use_template: bool = True,
             logits_processor=None) -> dict:
    """한 번 생성하고 출력·토큰 수·지연시간을 돌려준다. gen 기본값은 평가용 GEN_EVAL."""
    gen = dict(gen or config.GEN_EVAL)
    inputs = chat_inputs(tok, messages, use_template).to(model.device)
    n_in = inputs["input_ids"].shape[1]
    if logits_processor is not None:
        gen["logits_processor"] = logits_processor
    with utils.Timer() as t:
        out = model.generate(**inputs, pad_token_id=tok.pad_token_id, **gen)
    new = out[0, n_in:]
    text = strip_think(tok.decode(new, skip_special_tokens=True))
    return {"text": text, "n_in": int(n_in), "n_out": int(new.shape[0]), "latency": round(t.sec, 3)}


def load_embedder(name: str | None = None):
    """문장 임베딩 모델(Encoder). 1·6회차."""
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    m = SentenceTransformer(name or config.EMBED_MODEL, device=device())
    dim = m.get_embedding_dimension() if hasattr(m, "get_embedding_dimension") else m.get_sentence_embedding_dimension()
    print(f"[임베딩] {name or config.EMBED_MODEL} 로딩 {time.perf_counter() - t0:.1f}s, 차원 {dim}")
    return m


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "ratio": round(trainable / total, 6) if total else 0}
