"""5~7회차: 프롬프트 구성.

평가(benchmark.py)와 학습(build_train_data.py)이 같은 함수로 메시지를 만든다.
두 쪽 프롬프트가 어긋나면 LoRA 효과가 제대로 나오지 않는다(학습 데이터 원칙: 템플릿 일치).

출력 형식 강제 3방식(5회차 실습 ⑤)
  instruction  지시문: 시스템 프롬프트에 형식 규칙과 필드 설명을 적는다
  fewshot      예시 제공: instruction + 정답 예시 2건을 앞선 대화로 넣는다
  constrained  디코딩 제약: instruction + 생성 단계에서 스키마에 맞는 토큰만 허용(xgrammar 필요)
  plain        비교 기준: 형식 설명 없이 "JSON으로 답하라"만 적는다
"""
from __future__ import annotations

import json

import config
import utils

STRATEGIES = ("plain", "instruction", "fewshot", "constrained")


def _field_lines(schema: dict) -> str:
    lines = []
    for name, spec in schema["properties"].items():
        if "enum" in spec:
            vals = " | ".join("null" if v is None else str(v) for v in spec["enum"])
            kind = f"다음 중 하나: {vals}"
        else:
            t = spec.get("type")
            kind = " 또는 ".join(t) if isinstance(t, list) else str(t)
        lines.append(f"- {name}: {spec.get('description', '')} ({kind})")
    return "\n".join(lines)


def system_prompt(schema: dict, strategy: str = "instruction") -> str:
    if strategy == "plain":
        return "고객 문의를 읽고 JSON으로 답하세요."
    return (
        f"{schema.get('description', '')}\n\n"
        "규칙\n"
        "1. 출력은 JSON 객체 하나뿐이다. 설명 문장, 코드 블록 표시(```)를 붙이지 않는다.\n"
        "2. 모든 필드를 빠짐없이 쓰고, 해당 값이 없으면 null로 쓴다.\n"
        "3. [참고 문서]가 주어지면 그 내용을 근거로 판단한다. 문서에 없는 내용은 지어내지 않는다.\n\n"
        f"필드\n{_field_lines(schema)}"
    )


def user_prompt(text: str, context: str | None = None) -> str:
    if context:
        return f"[참고 문서]\n{context}\n\n[문의]\n{text}"
    return f"[문의]\n{text}"


def fewshot_examples(domain: str | None = None, k: int = 2) -> list[dict]:
    """학습 시드에서 유형이 다른 예시 k건(평가셋과 겹치지 않음)."""
    path = config.data_dir(domain) / "train_seed.jsonl"
    rows = utils.read_jsonl(path)
    picked, seen = [], set()
    for r in rows:
        key = (r.get("type"), r["gold"].get("category"))
        if key not in seen:
            seen.add(key)
            picked.append(r)
        if len(picked) == k:
            break
    return picked


def build_messages(text: str, schema: dict, strategy: str = "instruction", context: str | None = None,
                   domain: str | None = None) -> list[dict]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy는 {STRATEGIES} 중 하나")
    msgs = [{"role": "system", "content": system_prompt(schema, strategy)}]
    if strategy == "fewshot":
        for ex in fewshot_examples(domain):
            msgs.append({"role": "user", "content": user_prompt(ex["input"])})
            msgs.append({"role": "assistant", "content": json.dumps(ex["gold"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_prompt(text, context)})
    return msgs


# ---------------------------------------------------------------- 디코딩 제약(도전·비교용)
_COMPILED: dict = {}


def json_constraint(tok, model, schema: dict):
    """스키마에 맞는 토큰만 허용하는 logits processor 목록. 생성 1회마다 새로 만들어야 한다."""
    try:
        import xgrammar as xgr
    except ImportError as e:
        raise RuntimeError("constrained 방식은 xgrammar가 필요합니다: pip install xgrammar") from e
    key = (id(tok), json.dumps(schema, sort_keys=True, ensure_ascii=False))
    if key not in _COMPILED:
        info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=model.config.vocab_size)
        _COMPILED[key] = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema, ensure_ascii=False))
    return [xgr.contrib.hf.LogitsProcessor(_COMPILED[key])]


def to_train_example(text: str, gold: dict, schema: dict, domain: str | None = None,
                     context: str | None = None) -> dict:
    """7회차 학습 예시(TRL prompt-completion 대화 형식). 평가와 같은 instruction 프롬프트를 쓴다."""
    prompt = build_messages(text, schema, "instruction", context, domain)
    return {"prompt": prompt,
            "completion": [{"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)}]}
