"""5회차 사건 6: 채점 규칙 한 줄이 점수를 얼마나 바꾸는지 봄(같은 출력을 규칙만 바꿔 다시 채점)

  python experiments/s05/scoring_rules.py                                  # results/base.json, 없으면 예시 출력 8건
  python experiments/s05/scoring_rules.py --result results/sweep_strategy.json --value fewshot
  python experiments/s05/scoring_rules.py --weight category=1 action=1     # 가중치 바꾸기
  python experiments/s05/scoring_rules.py --match fact=exact               # 비교 방식 바꾸기(G)
  python experiments/s05/scoring_rules.py --tol unpaid_krw=1000            # 허용 오차 바꾸기(L)
  python experiments/s05/scoring_rules.py --match metric=exact             # 지표 이름을 정확히 요구(S)
  python experiments/s05/scoring_rules.py --mode strict                    # 앞뒤에 설명문이 붙으면 0점

출력: 원래 규칙 대 바꾼 규칙의 Task Accuracy·Exact Match·필드별 점수, 점수가 바뀐 문항
판단: 규칙은 Baseline 전에 팀이 합의해 고정함. 바꾸면 그 뒤 측정은 모두 다시 잼
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import scorer  # noqa: E402
import utils  # noqa: E402


def sample_outputs(items: list[dict]) -> list[dict]:
    """결과 파일이 없을 때 쓰는 예시 출력. 실제 모델이 자주 내는 모양 네 가지를 돌려 가며 만듦"""
    out = []
    for k, it in enumerate(items):
        g = dict(it["gold"])
        kind = k % 4
        if kind == 0:
            text = json.dumps(g, ensure_ascii=False)
        elif kind == 1:
            text = "```json\n" + json.dumps(g, ensure_ascii=False) + "\n```"
        elif kind == 2:
            first = next(iter(g))
            g2 = dict(g)
            g2[first] = None if g[first] is not None else "기타"
            text = "확인해 보니 다음과 같습니다.\n" + json.dumps(g2, ensure_ascii=False)
        else:
            g2 = {key: (f"{v} 정도" if isinstance(v, str) and any(c.isdigit() for c in v) else v) for key, v in g.items()}
            text = json.dumps(g2, ensure_ascii=False)
        out.append({"id": it["id"], "type": it.get("type"), "gold": it["gold"], "output": text})
    return out


def kv(pairs: list[str] | None) -> dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def rescore(items, rules, schema, mode):
    rows = [{**scorer.score_item(it["output"], it["gold"], rules, schema, mode), "id": it["id"], "type": it.get("type")} for it in items]
    return rows, scorer.aggregate(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", help="benchmark.py·sweep.py 결과 파일(기본 results/base.json)")
    ap.add_argument("--value", help="sweep 결과일 때 볼 값(예: fewshot)")
    ap.add_argument("--weight", nargs="*", help="필드=가중치")
    ap.add_argument("--match", nargs="*", help="필드=비교방식(exact·enum·bool·numeric·numbers·numseq·contains)")
    ap.add_argument("--tol", nargs="*", help="필드=허용오차(numeric)")
    ap.add_argument("--mode", choices=["loose", "strict"], default="loose")
    a = ap.parse_args()

    schema, rules = scorer.load_schema(), scorer.load_rules()
    path = Path(a.result) if a.result else config.PATHS["results"] / "base.json"
    if path.exists():
        res = utils.read_json(path)
        items = res["detail"][a.value or res["values"][0]]["items"] if "detail" in res else res["items"]
        src = f"{path.name}" + (f" [{a.value or res['values'][0]}]" if "detail" in res else "")
    else:
        items = sample_outputs(utils.read_json(config.data_dir() / "test_seed.json")[:8])
        src = "예시 출력 8건(results/base.json이 아직 없음)"

    new = copy.deepcopy(rules)
    for f, w in kv(a.weight).items():
        new[f]["weight"] = float(w)
    for f, m in kv(a.match).items():
        new[f]["match"] = m
    for f, t in kv(a.tol).items():
        new[f]["tol"] = float(t)

    r0, m0 = rescore(items, rules, schema, "loose")
    r1, m1 = rescore(items, new, schema, a.mode)
    print(f"[채점 규칙 실험] 도메인 {config.DOMAIN} · {src} · {len(items)}문항")
    changed = {f: {k: v for k, v in new[f].items() if rules[f].get(k) != v} for f in new}
    changed = {f: c for f, c in changed.items() if c}
    print("바꾼 규칙 :", json.dumps(changed, ensure_ascii=False) if changed else "(없음)", f"· 파싱 {a.mode}")
    rows = [{"지표": "Task Accuracy", "원래": m0["task_accuracy"], "바꾼 뒤": m1["task_accuracy"]},
            {"지표": "Exact Match", "원래": m0["exact_match"], "바꾼 뒤": m1["exact_match"]},
            {"지표": "JSON 성공률", "원래": m0["json_success"], "바꾼 뒤": m1["json_success"]}]
    rows += [{"지표": f"필드 {f}", "원래": m0["by_field"][f], "바꾼 뒤": m1["by_field"][f]} for f in rules]
    utils.print_table(rows)
    moved = [(x["id"], x["score"], y["score"]) for x, y in zip(r0, r1) if abs(x["score"] - y["score"]) > 1e-9]
    print(f"\n점수가 바뀐 문항 {len(moved)}건")
    for i, s0, s1 in moved[:12]:
        print(f"  {i}: {s0:.2f} → {s1:.2f}")


if __name__ == "__main__":
    main()
