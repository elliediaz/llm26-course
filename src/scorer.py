"""5회차: 채점기: 모델 출력(문자열)을 정답 JSON과 비교해 점수를 낸다.

구성
  parse()      출력에서 JSON을 꺼낸다. loose(기본)는 앞뒤 설명문·코드 펜스를 걷어 내고 첫 '{'~마지막 '}'를 읽는다.
               strict는 출력 전체가 JSON 이어야 한다(도전과제 ②: 느슨한 판과 엄격한 판 비교).
  validate()   schema.json 검증(JSON 성공률의 기준)
  score_item() 필드별 점수(scoring.yaml) → 가중 평균(Task Accuracy의 문항 점수), 핵심 필드 전부 정답 여부(Exact Match)
  aggregate()  문항 결과 → 지표(JSON 성공률, Task Accuracy, Exact Match, 유형별 정확도)

채점 규칙(scoring.yaml)을 바꾸면 이전 측정과 비교할 수 없다. 바꿀 때는 Baseline부터 다시 잰다.

단독 실행(이미 저장된 출력 다시 채점):
  python src/scorer.py results/baseline.json --mode strict
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import yaml

import config
import utils

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------- 파싱·검증
def parse(text: str, mode: str = "loose") -> tuple[dict | None, str]:
    """(객체 또는 None, 실패 사유)를 돌려준다."""
    t = text.strip()
    if mode == "strict":
        try:
            obj = json.loads(t)
        except json.JSONDecodeError as e:
            return None, f"JSON 아님({e.msg})"
        return (obj, "") if isinstance(obj, dict) else (None, "객체가 아님")
    m = _FENCE.search(t)
    if m:
        t = m.group(1)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None, "중괄호 없음"
    try:
        obj = json.loads(t[i:j + 1])
    except json.JSONDecodeError as e:
        return None, f"JSON 파싱 실패({e.msg})"
    return (obj, "") if isinstance(obj, dict) else (None, "객체가 아님")


def load_schema(domain: str | None = None) -> dict:
    return utils.read_json(config.data_dir(domain) / "schema.json")


def load_rules(domain: str | None = None, path: str | None = None) -> dict:
    p = Path(path) if path else config.data_dir(domain) / "scoring.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)["fields"]


def validate(obj: dict, schema: dict) -> list[str]:
    """스키마 위반 목록(비어 있으면 통과)."""
    from jsonschema import Draft202012Validator
    v = Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path)) or '(최상위)'}: {e.message}" for e in v.iter_errors(obj)]


# ---------------------------------------------------------------- 필드 비교
def norm(v) -> str:
    s = unicodedata.normalize("NFKC", str(v)).lower()
    return re.sub(r"[\s,·'\"‘’“”]", "", s)


def _empty(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", "null", "None")) or v == []


def numbers(v) -> set[str]:
    return {str(float(x)).rstrip("0").rstrip(".") for x in _NUM.findall(str(v).replace(",", ""))}


def field_score(pred, gold, rule: dict) -> float:
    if _empty(gold):
        return 1.0 if _empty(pred) else 0.0
    if _empty(pred):
        return 0.0
    m = rule.get("match", "exact")
    if m in ("exact", "enum"):
        return float(norm(pred) == norm(gold))
    if m == "bool":
        if isinstance(pred, str):
            pred = {"true": True, "false": False}.get(pred.strip().lower(), pred)
        return float(pred is gold)
    if m == "numeric":
        try:
            return float(abs(float(pred) - float(gold)) <= float(rule.get("tol", 0)))
        except (TypeError, ValueError):
            return 0.0
    if m == "numbers":          # 정답에 있는 숫자가 출력에 모두 있으면 정답
        g = numbers(gold)
        return float(bool(g) and g <= numbers(pred)) if g else float(norm(pred) == norm(gold))
    if m == "contains":
        return float(norm(gold) in norm(pred))
    if m == "set_f1":
        p = {norm(x) for x in (pred if isinstance(pred, list) else [pred])}
        g = {norm(x) for x in (gold if isinstance(gold, list) else [gold])}
        tp = len(p & g)
        if not tp:
            return 0.0
        prec, rec = tp / len(p), tp / len(g)
        return 2 * prec * rec / (prec + rec)
    raise ValueError(f"알 수 없는 match 방식: {m}")


def score_item(text: str, gold: dict, rules: dict, schema: dict, mode: str = "loose") -> dict:
    obj, why = parse(text, mode)
    errs = validate(obj, schema) if obj is not None else [why]
    json_ok = obj is not None and not errs
    fields, total, wsum = {}, 0.0, 0.0
    for name, rule in rules.items():
        w = float(rule.get("weight", 1))
        s = field_score(obj.get(name) if obj else None, gold.get(name), rule) if obj is not None else 0.0
        fields[name] = s
        total += w * s
        wsum += w
    core = [n for n, r in rules.items() if r.get("core")]
    return {
        "parsed": obj,
        "json_ok": json_ok,
        "errors": errs[:3],
        "fields": fields,
        "score": round(total / wsum, 4) if obj is not None else 0.0,   # 파싱 실패는 0점
        "exact": bool(obj is not None and core and all(fields[n] == 1.0 for n in core)),
    }


def aggregate(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {}
    by_type: dict[str, list[float]] = {}
    for it in items:
        by_type.setdefault(it.get("type", "?"), []).append(it["score"])
    field_names = list(items[0]["fields"].keys())
    return {
        "n": n,
        "json_success": round(sum(it["json_ok"] for it in items) / n, 4),
        "task_accuracy": round(sum(it["score"] for it in items) / n, 4),
        "exact_match": round(sum(it["exact"] for it in items) / n, 4),
        "by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
        "by_field": {f: round(sum(it["fields"][f] for it in items) / n, 4) for f in field_names},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="저장된 benchmark 결과를 다른 규칙으로 다시 채점")
    ap.add_argument("result", help="benchmark.py가 저장한 results/*.json")
    ap.add_argument("--mode", choices=["loose", "strict"], default="loose")
    ap.add_argument("--rules", help="다른 scoring.yaml 경로(느슨·엄격 규칙 비교용)")
    a = ap.parse_args()
    res = utils.read_json(a.result)
    domain = res["config"]["domain"]
    rules, schema = load_rules(domain, a.rules), load_schema(domain)
    items = []
    for it in res["items"]:
        s = score_item(it["output"], it["gold"], rules, schema, a.mode)
        items.append({**s, "type": it.get("type")})
    agg = aggregate(items)
    print(json.dumps({"원래": res["metrics"], f"재채점({a.mode})": agg}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
