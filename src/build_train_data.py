"""7회차 실습 1 ①②: 학습 데이터 준비: 시드 + 내가 쓴 추가분 + 규칙 증강 → 중복·평가셋 중첩 검사 → 학습 형식 변환.

  python src/build_train_data.py                       # 시드만(약 200건)
  python src/build_train_data.py --augment 300         # 규칙 증강으로 300건 추가(권장 총 500건)
  python src/build_train_data.py --my data/G/my_train_items.jsonl --augment 300 --out-tag mine   # 내가 쓴 학습 문항을 더함 → train_mine.jsonl
  python src/build_train_data.py --augment 300 --out-tag n100 --limit 100     # 데이터 양 실험(train_n100.jsonl)
  사건 실험(일부러 문제를 만들어 결과를 비교, 사건 카드 7회차)
  python src/build_train_data.py --augment 300 --noise 0.3 --out-tag noise30  # 학습 정답의 30%를 틀리게
  python src/build_train_data.py --augment 300 --leak --out-tag leak          # 평가 문항을 학습에 섞음(누수)

규칙
  - 메시지는 평가와 같은 prompt_utils.build_messages(instruction)로 만든다 → 템플릿 일치
  - 평가셋(test_dataset.json)과 같은 입력·너무 비슷한 입력(문자 3-gram 자카드 0.8 이상)은 뺀다
  - 지식(패치 수치 등)을 외우게 하려는 게 아니라 형식·판단 규칙을 맞추는 것이 목적이다
결과: data/<도메인>/train[_태그].jsonl, valid.jsonl(10%), results/train_data_report[_태그].json
"""
from __future__ import annotations

import argparse
import random
import re

import config
import dataset_tool
import prompt_utils
import scorer
import utils

# 표현 증강 규칙(도메인 공통 + G 은어). 정답은 바꾸지 않는 변형만 쓴다
SWAPS = [("안 들어왔어요", "안 들어와요"), ("안 들어왔어요", "아직 안 왔어요"), ("확인해 주세요", "확인 부탁드립니다"),
         ("확인 부탁드려요", "봐 주세요"), ("같아요", "같은데요"), ("했는데", "했거든요"), ("버그", "오류"),
         ("루멘석", "루멘"), ("게임이 꺼져요", "게임이 튕겨요"), ("의도된 건가요", "일부러 그런 건가요"),
         ("환불", "환불 신청"), ("계정", "제 계정")]
TAILS = ["", " ㅠㅠ", " 빨리 답변 주세요", " 급합니다", " 부탁드려요", "!!"]


def augment(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        r = rng.choice(rows)
        text = r["input"]
        for a, b in rng.sample(SWAPS, k=len(SWAPS)):
            if a in text and rng.random() < 0.6:
                text = text.replace(a, b, 1)
        text = text + rng.choice(TAILS)
        if rng.random() < 0.3:                        # 띄어쓰기 흔들기
            text = re.sub(r" ", "", text, count=1)
        if text != r["input"]:
            out.append({**r, "id": f"{r['id']}-a{len(out)}", "input": text, "aug": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--augment", type=int, default=0)
    ap.add_argument("--my", "--team", dest="my", help="직접 쓴 학습 문항(jsonl: id, type, input, gold). 예: --my data/<도메인>/my_train_items.jsonl")
    ap.add_argument("--max-sim", type=float, default=0.8)
    ap.add_argument("--limit", type=int, help="학습 건수를 N건으로 줄임(데이터 양 실험)")
    ap.add_argument("--noise", type=float, default=0.0, help="학습 정답 중 이 비율의 첫 필드를 다른 값으로 바꿈(라벨 오류 사건)")
    ap.add_argument("--leak", action="store_true", help="평가 문항을 학습에 섞음(누수 사건: 점수가 부풀려지는지 확인)")
    ap.add_argument("--out-tag", default="", help="train_<태그>.jsonl로 따로 저장(기본 train.jsonl을 덮지 않음)")
    a = ap.parse_args()
    rng = random.Random(config.SEED)
    d = config.data_dir()
    schema = scorer.load_schema()
    rows = utils.read_jsonl(d / "train_seed.jsonl")
    if a.my:
        rows += utils.read_jsonl(a.my)
    rows += augment(rows, a.augment, rng)

    test = utils.read_json(d / "test_dataset.json")
    if a.leak:                                   # 일부러 만든 문제: 평가 문항을 그대로 학습에 넣음
        rows += [{**t, "id": f"{t['id']}-leak"} for t in test]
    kept, drop_dup, drop_test, drop_bad, seen = [], 0, [], 0, set()
    for r in rows:
        if r["input"] in seen:
            drop_dup += 1
            continue
        if scorer.validate(r["gold"], schema):
            drop_bad += 1
            continue
        near = max((dataset_tool.similarity(r["input"], t["input"]) for t in test), default=0)
        if near >= a.max_sim and not a.leak:
            drop_test.append({"id": r["id"], "input": r["input"], "sim": round(near, 2)})
            continue
        seen.add(r["input"])
        kept.append(r)
    rng.shuffle(kept)
    n_val = max(1, len(kept) // 10)
    val, tr = kept[:n_val], kept[n_val:]
    if a.limit:
        tr = tr[:a.limit]
    n_noise = 0
    if a.noise > 0:                              # 일부러 만든 문제: 첫 필드(분류·쟁점·리그)를 다른 후보로 바꿈
        first = next(iter(schema["properties"]))
        cands = [v for v in schema["properties"][first].get("enum", []) if v is not None]
        for r in rng.sample(tr, int(len(tr) * a.noise)):
            if cands and r["gold"].get(first) in cands:
                r["gold"] = {**r["gold"], first: rng.choice([c for c in cands if c != r["gold"][first]])}
                n_noise += 1
    tag = f"_{a.out_tag}" if a.out_tag else ""
    utils.write_jsonl(d / f"train{tag}.jsonl", [prompt_utils.to_train_example(r["input"], r["gold"], schema) for r in tr])
    utils.write_jsonl(d / "valid.jsonl", [prompt_utils.to_train_example(r["input"], r["gold"], schema) for r in val])

    lens = None
    try:
        import model_utils
        tok = model_utils.load_tokenizer()
        ex = prompt_utils.to_train_example(tr[0]["input"], tr[0]["gold"], schema)
        lens = []
        for r in tr[:200]:
            e = prompt_utils.to_train_example(r["input"], r["gold"], schema)
            ids = tok.apply_chat_template(e["prompt"] + e["completion"], tokenize=True, **config.CHAT_TEMPLATE_KWARGS)
            lens.append(len(ids["input_ids"] if hasattr(ids, "keys") else ids))
        print("\n[학습 예시 1건: 모델 템플릿 적용 결과]")
        print(tok.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False, **config.CHAT_TEMPLATE_KWARGS)[-600:])
    except Exception as e:
        print(f"(토큰 길이 확인 생략: {e.__class__.__name__})")

    from collections import Counter
    report = {"seed": len(utils.read_jsonl(d / "train_seed.jsonl")), "augmented": a.augment, "total_in": len(rows),
              "train": len(tr), "valid": len(val), "drop_duplicate": drop_dup, "drop_schema": drop_bad,
              "drop_near_test": len(drop_test), "near_test_examples": drop_test[:5],
              "noise_changed": n_noise, "leak": a.leak, "file": f"train{tag}.jsonl",
              "type_counts": dict(Counter(r.get("type") for r in tr)),
              "token_len": {"max": max(lens), "p95": utils.percentile(lens, 0.95), "mean": round(sum(lens) / len(lens), 1)} if lens else None}
    print("\n[학습 데이터]", {k: v for k, v in report.items() if k != "near_test_examples"})
    if lens and report["token_len"]["max"] > config.LORA["max_length"]:
        print(f"  [주의] 최대 토큰 {report['token_len']['max']} > LORA.max_length {config.LORA['max_length']} → 잘리는 예시가 생김")
    if a.leak:
        print("  [주의] 평가 문항이 학습에 들어감(누수 사건 실험). 이 데이터로 잰 점수는 기록표에 올리지 않음")
    utils.write_json(utils.result_path(f"train_data_report{tag}.json"), report)


if __name__ == "__main__":
    main()
