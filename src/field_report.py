"""5회차 결과 검증: JSON 필드별 점수를 '분류·추출·숫자'로 나눠 보고, 1회차 고전 분류기와 같은 문항에서 비교한다.

  python src/field_report.py                     # results/base.json(Baseline) 기준
  python src/field_report.py --name fmt_constrained
  python src/field_report.py --name base --clf results/ticket_clf.json

필드 성격(schema.json에 후보 목록 enum이 있으면 분류, 없으면 scoring.yaml의 match로 판단)
  enum·bool          → 분류: 정해진 후보 중 하나를 고름. 지표 = 정확도
  numeric            → 회귀: 숫자의 오차를 허용 범위로 판정
  exact·numbers 등    → 추출: 입력·문서의 글자를 그대로 옮김(LLM에게는 '토큰 분류'의 연속)
확인할 것
  - LLM의 category·action·priority 정확도 vs 1회차 로지스틱 회귀 분류기(같은 시드 문항)
  - 전용 분류기는 그 필드 하나만 잘하고, LLM은 7개 필드를 한 번에 JSON으로 낸다(형식 실패가 점수를 깎음)
  - 어느 필드가 약한지 → 6회차(RAG: 지식이 필요한 entity·patch·fact)·7회차(LoRA: 형식·판단 필드)로 이어짐
결과: results/field_report_<name>.md
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import config
import scorer
import utils

KIND = {"enum": "분류", "bool": "분류", "numeric": "회귀(숫자)", "exact": "추출", "numbers": "추출(숫자 포함)",
        "set_f1": "추출(목록)", "contains": "추출"}


def kind_of(field: str, rule: dict, schema_props: dict) -> str:
    if "enum" in (schema_props.get(field) or {}):          # 후보 목록이 정해져 있으면 채점 방식과 무관하게 분류
        return "분류"
    return KIND.get(rule.get("match"), rule.get("match"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="base", help="results/<name>.json (benchmark.py 결과)")
    ap.add_argument("--clf", default=str(config.PATHS["results"] / "ticket_clf.json"), help="1회차 ticket_classifier.py 결과")
    a = ap.parse_args()

    res = utils.read_json(utils.result_path(f"{a.name}.json"))
    domain = res.get("config", {}).get("domain")
    rules = scorer.load_rules(domain)
    try:
        schema = utils.read_json(config.data_dir(domain) / "schema.json")
        props = schema.get("properties", schema)
    except FileNotFoundError:
        props = {}
    items = res["items"]
    per = defaultdict(list)
    for it in items:
        for f, s in (it.get("fields") or {}).items():
            per[f].append(s)

    clf_by_id = {}
    try:
        clf = utils.read_json(a.clf)
        clf_by_id = {p["id"]: p for p in clf.get("test_predictions", [])}
    except FileNotFoundError:
        print(f"[알림] {a.clf} 파일이 없어 1회차 분류기 비교는 건너뜀(1회차 해보기 E 결과)")

    rows = []
    for f, rule in rules.items():
        scores = per.get(f, [])
        row = {"필드": f, "성격": kind_of(f, rule, props), "가중치": rule.get("weight", 1),
               "LLM 정확도": round(sum(scores) / len(scores), 3) if scores else None, "문항 수": len(scores)}
        common = [it for it in items if it["id"] in clf_by_id and f in clf_by_id[it["id"]]["pred"]]
        if common:
            llm = sum((it.get("fields") or {}).get(f, 0) for it in common) / len(common)
            cl = sum(str(clf_by_id[it["id"]]["pred"][f]) == str(it["gold"][f]) for it in common) / len(common)
            row["같은 문항: LLM"] = round(llm, 3)
            row["같은 문항: 1회차 분류기"] = round(cl, 3)
            row["비교 문항"] = len(common)
        rows.append(row)

    print(f"\n## {a.name}: 필드별 점수 ({len(items)}문항, 모델 {res.get('config', {}).get('model_id')})")
    cols = ["필드", "성격", "가중치", "LLM 정확도", "문항 수", "같은 문항: LLM", "같은 문항: 1회차 분류기", "비교 문항"]
    table = utils.print_table(rows, cols)
    m = res.get("metrics", {})
    print(f"\nJSON 성공률 {m.get('json_success')} · Task Accuracy {m.get('task_accuracy')}")
    print("해석: 분류 필드가 약하면 7회차 LoRA(판단·형식), 추출 필드가 약하면 6회차 RAG(근거 문서), 회귀(숫자) 필드가 약하면 8회차 계산 Tool 쪽 효과를 기대함")
    path = utils.result_path(f"field_report_{a.name}.md")
    path.write_text(f"# 필드별 점수: {a.name}\n\n{table}\n", encoding="utf-8")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
