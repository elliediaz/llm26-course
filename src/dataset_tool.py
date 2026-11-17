"""5회차 실습 ③: 팀 문항 작성 → 점검 → 시드와 합쳐 Test Dataset 30건 완성.

  cp data/G/team_item_template.json data/G/team_items.json     # 팀 작성 파일(복사본에 작성)
  python src/dataset_tool.py check                             # 형식·유형 할당·정답 스키마·중복 점검
  python src/dataset_tool.py merge                             # test_seed + team_items → test_dataset.json(30건)
  python src/dataset_tool.py stats                             # 현재 test_dataset.json 유형 분포
  python src/dataset_tool.py leak                              # 7회차: 학습 파일(train*.jsonl)과 시험 문항이 겹치는지 점검

강사 전용(현장 보정): python src/dataset_tool.py swap G-T-07 G-P-K03 --pool solutions/data/G/test_pool.json
"""
from __future__ import annotations

import argparse
import collections
import sys

import config
import scorer
import utils


def _ngrams(s: str, n: int = 3) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def similarity(a: str, b: str) -> float:
    x, y = _ngrams(a), _ngrams(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def check(team_path, seed, schema, alloc) -> list[str]:
    team = utils.read_json(team_path)
    items = team["items"] if isinstance(team, dict) else team
    errs = []
    cnt = collections.Counter(it["type"] for it in items if it.get("input"))
    for t, k in alloc.items():
        if cnt.get(t, 0) != k:
            errs.append(f"유형 {t}: {cnt.get(t, 0)}건 작성 / 할당 {k}건")
    doc_ids = {m["doc_id"] for m in utils.read_json(config.data_dir() / "corpus_manifest.json")}
    seen = [s["input"] for s in seed]
    for it in items:
        where = it.get("id", "?")
        if not it.get("input"):
            errs.append(f"{where}: input 비어 있음")
            continue
        bad = scorer.validate(it["gold"], schema)
        if bad:
            errs.append(f"{where}: gold 스키마 위반: {bad[0]}")
        if it["type"] == "K" and not it.get("gold_doc_ids"):
            errs.append(f"{where}: K 유형은 gold_doc_ids 필수")
        for d in it.get("gold_doc_ids") or []:
            if d not in doc_ids:
                errs.append(f"{where}: 없는 문서 id {d}")
        for s in seen:
            if similarity(it["input"], s) > 0.6:
                errs.append(f"{where}: 다른 문항과 너무 비슷함 → '{s[:30]}…'")
                break
        seen.append(it["input"])
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "merge", "stats", "swap", "leak"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--team", default=None, help="기본: data/<DOMAIN>/team_items.json")
    ap.add_argument("--pool", help="swap에 쓸 교체 풀(강사 전용)")
    ap.add_argument("--force", action="store_true", help="check 경고가 있어도 merge")
    a = ap.parse_args()
    d = config.data_dir()
    team_path = a.team or d / "team_items.json"
    seed = utils.read_json(d / "test_seed.json")
    schema = scorer.load_schema()
    alloc = utils.read_json(d / "team_item_template.json")["allocation"]

    if a.cmd == "check":
        errs = check(team_path, seed, schema, alloc)
        print("\n".join(f"  [오류] {e}" for e in errs) if errs else "  [통과] 점검 통과")
        sys.exit(1 if errs else 0)
    if a.cmd == "merge":
        errs = check(team_path, seed, schema, alloc)
        if errs and not a.force:
            print("\n".join(f"  [오류] {e}" for e in errs) + "\n점검 실패 → 고친 뒤 다시 실행(--force로 강제 가능)")
            sys.exit(1)
        team = utils.read_json(team_path)
        items = [it for it in (team["items"] if isinstance(team, dict) else team) if it.get("input")]
        merged = seed + [{k: it[k] for k in ("id", "type", "input", "gold", "gold_doc_ids")} for it in items]
        utils.write_json(d / "test_dataset.json", merged)
        print(f"  test_dataset.json {len(merged)}건 저장 (시드 {len(seed)} + 팀 {len(items)})")
    if a.cmd == "stats":
        ds = utils.read_json(d / "test_dataset.json")
        print(f"  {len(ds)}건 {dict(collections.Counter(x['type'] for x in ds))}")
    if a.cmd == "leak":
        ds = utils.read_json(d / "test_dataset.json")
        for f in sorted(d.glob("train*.jsonl")):
            if f.name == "train_seed.jsonl":
                continue
            rows = utils.read_jsonl(f)
            texts = [next(m["content"] for m in r["prompt"] if m["role"] == "user").split("[문의]\n")[-1] for r in rows if "prompt" in r]
            same = near = 0
            for t in ds:
                best = max((similarity(t["input"], x) for x in texts), default=0)
                same += best >= 0.99
                near += 0.8 <= best < 0.99
            flag = "  ← 시험 문항이 학습에 들어 있음" if same else ""
            print(f"  {f.name:24s} 학습 {len(rows):4d}건 · 시험 문항과 같음 {same}건 · 매우 비슷(0.8 이상) {near}건{flag}")
    if a.cmd == "swap":
        out_id, in_id = a.args
        pool = {p["id"]: p for p in utils.read_json(a.pool)}
        ds = utils.read_json(d / "test_dataset.json")
        i = next(i for i, x in enumerate(ds) if x["id"] == out_id)
        new = pool[in_id]
        if new["type"] != ds[i]["type"]:
            sys.exit(f"유형이 다름: {ds[i]['type']} → {new['type']} (같은 유형끼리만 교체)")
        ds[i] = {k: new[k] for k in ("id", "type", "input", "gold", "gold_doc_ids")}
        utils.write_json(d / "test_dataset.json", ds)
        print(f"  {out_id} → {in_id} 교체. 성능 기록표에도 남길 것: python src/scoreboard.py --swap {out_id} {in_id} --reason ...")


if __name__ == "__main__":
    main()
