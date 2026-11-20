"""5회차 도전과제 ①: 규칙으로 채점하기 어려운 필드만 LLM에게 채점을 맡기고 규칙 채점과 비교한다.

  python src/judge.py results/base.json --field fact

주의
  - 채점 모델이 같은 모델이면 자기 출력에 후한 경향이 있을 수 있다(가능하면 다른 모델로 --judge-model 지정).
  - LLM 채점도 달라진다. 같은 문항을 두 번 채점해 결과가 같은지 확인한다.
  - 주지표(Task Accuracy)는 규칙 채점으로 유지하고, LLM 채점은 참고 지표로만 기록한다.
"""
from __future__ import annotations

import argparse

import config
import model_utils
import scorer
import utils

PROMPT = """두 문장이 같은 사실을 말하는지 판정한다.
정답: {gold}
응답: {pred}
같은 사실이면 "예", 다르거나 응답이 비어 있으면 "아니오" 한 단어로만 답한다."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--field", default="fact")
    ap.add_argument("--judge-model", default=None, help="기본: config.MODEL_ID")
    a = ap.parse_args()
    res = utils.read_json(a.result)
    model, tok = model_utils.load_llm(a.judge_model)
    rows, agree = [], 0
    for it in res["items"]:
        gold = it["gold"].get(a.field)
        pred = (it.get("parsed") or {}).get(a.field)
        if gold is None:
            continue
        msgs = [{"role": "user", "content": PROMPT.format(gold=gold, pred=pred)}]
        out = model_utils.generate(model, tok, msgs, {"do_sample": False, "max_new_tokens": 5})["text"]
        llm = 1.0 if out.strip().startswith("예") else 0.0
        rule = it["fields"].get(a.field, 0.0)
        agree += int(llm == rule)
        rows.append({"id": it["id"], "정답": gold, "응답": pred, "규칙": rule, "LLM": llm})
    utils.print_table(rows)
    if rows:
        print(f"\n규칙 채점과 LLM 채점 일치율: {agree / len(rows):.2f} ({len(rows)}문항, 필드 {a.field})")
        utils.write_json(utils.result_path(f"judge_{res['name']}_{a.field}.json"),
                         {"field": a.field, "agreement": agree / len(rows), "rows": rows, "judge": a.judge_model or config.MODEL_ID})


if __name__ == "__main__":
    main()
