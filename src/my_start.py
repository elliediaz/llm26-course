"""개인 실습 시작(6회차부터, 개인 GPU 서버): 내 도메인·시험지·프롬프트 방식을 정하고 진행 상태를 봄.

  python src/my_start.py                       # 내 설정과 진행 상태(무엇이 있고 무엇이 없는지)
  python src/my_start.py --domain S            # 도메인 고르기(G · L · S) + 개인 시험지 30건 만들기
  python src/my_start.py --domain S --seat 7   # 문항 풀에서 내 서버 번호로 20건을 뽑음(사람마다 다른 시험지)
  python src/my_start.py --strategy fewshot --shots 2 --prompt prompts/s05/형식엄수.txt   # 5회차에 고른 방식·규칙 파일
  python src/my_start.py --model Qwen/Qwen3-8B # 내 모델 바꾸기(받아 둔 모델 또는 hf download 로 받은 모델)
  python src/my_start.py --rebuild             # 시험지를 다시 만듦(직접 고친 시험지를 덮어씀)

개인 시험지 = 공통 시드 10건 + 문항 풀(data/<도메인>/item_pool.json)에서 K 8 · R 7 · F 5건.
  앞 10건(시드)은 모두 같으므로 `--limit 10` 측정은 워크북의 실측 표·옆 사람 결과와 견줄 수 있고,
  뒤 20건은 --seat 번호에 따라 달라짐(L 은 풀이 20건이라 모두 같음).
설정은 저장소 맨 위 my_config.json 에 남고, 모든 측정 스크립트가 이 값을 기본으로 씀.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys

import config

MY_PATH = config.ROOT / "my_config.json"
DOMAINS = {"G": "게임 운영(상담 티켓 JSON)", "L": "알바 급여 정산(정산 JSON)", "S": "스포츠 기록실(답변 JSON)"}
ALLOC = {"K": 8, "R": 7, "F": 5}
KEYS = ("id", "type", "input", "gold", "gold_doc_ids")


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def build_dataset(domain: str, seat: int) -> list[dict]:
    d = config.PATHS["data"] / domain
    seed, pool = read_json(d / "test_seed.json"), read_json(d / "item_pool.json")
    rng = random.Random(1000 * "GLS".index(domain) + seat)
    picked = []
    for t, n in ALLOC.items():
        cand = sorted((p for p in pool if p["type"] == t), key=lambda p: p["id"])
        picked += sorted(rng.sample(cand, n), key=lambda p: p["id"]) if len(cand) > n else cand
    ds = seed + [{k: p.get(k) for k in KEYS} for p in picked]
    (d / "test_dataset.json").write_text(json.dumps(ds, ensure_ascii=False, indent=1), encoding="utf-8")
    return ds


def status(domain: str) -> None:
    d, r = config.PATHS["data"] / domain, config.PATHS["results"]
    ds = read_json(d / "test_dataset.json")
    types = dict(collections.Counter(x["type"] for x in ds))
    idx = sorted(p.parent.name for p in config.PATHS["index"].glob(f"{domain}_*/meta.json"))
    ads = sorted(str(p.parent.relative_to(config.ROOT)) for p in config.PATHS["adapters"].glob("*/*/adapter_config.json"))
    rows = [
        ("시험지", f"{len(ds)}건 {types}", "30건이 아니면 python src/my_start.py --domain " + domain + " --rebuild"),
        ("Baseline", "있음" if (r / "base.json").exists() else "없음", "python src/benchmark.py --runs 2"),
        ("인덱스(6회차)", ", ".join(idx) or "없음", "python src/index_builder.py"),
        ("RAG 결과(6회차)", "있음" if (r / "rag.json").exists() else "없음", "python src/benchmark.py --mode rag"),
        ("학습 데이터(7회차)", "있음" if (d / "train.jsonl").exists() else "없음", "python src/build_train_data.py --augment 300"),
        ("Adapter(7회차)", f"{len(ads)}개 " + " ".join(ads[:3]) if ads else "없음", "python src/train_lora.py --run r16"),
    ]
    w = max(len(a) for a, _, _ in rows) + 2
    for a, b, c in rows:
        print(f"  {a:<{w}}{b}" + (f"   → {c}" if b.startswith("없음") or (a == "시험지" and len(ds) != 30) else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=list(DOMAINS))
    ap.add_argument("--seat", type=int, help="내 서버 번호(P07이면 7). 문항 풀에서 뽑는 20건이 번호마다 달라짐")
    ap.add_argument("--name", help="Adapter 폴더 이름(기본 me)")
    ap.add_argument("--model", help="내 모델(Hugging Face 저장소 이름)")
    ap.add_argument("--quant", choices=["bf16", "nf4"])
    ap.add_argument("--strategy", choices=["plain", "instruction", "fewshot", "constrained"])
    ap.add_argument("--shots", type=int)
    ap.add_argument("--prompt", help="덧붙일 규칙 파일. base 를 넣으면 지움")
    ap.add_argument("--rebuild", action="store_true", help="시험지를 다시 만듦")
    a = ap.parse_args()

    my = dict(config.MY)
    changed = False
    for key, val in (("domain", a.domain), ("seat", a.seat), ("name", a.name), ("model_id", a.model), ("quant", a.quant),
                     ("strategy", a.strategy), ("shots", a.shots)):
        if val is not None and my.get(key) != val:
            my[key], changed = val, True
    if a.prompt is not None:
        if a.prompt != "base" and not (config.ROOT / a.prompt).exists():
            sys.exit(f"규칙 파일이 없음: {a.prompt}")
        my["prompt"], changed = (None if a.prompt == "base" else a.prompt), True
    domain = my.get("domain") or config.DOMAIN
    if not my.get("domain"):
        print("도메인을 아직 정하지 않았음. 하나를 골라 다시 실행함")
        for k, v in DOMAINS.items():
            print(f"  python src/my_start.py --domain {k}    # {v}")
        print("  data/<도메인>/README.md 와 corpus/ 문서 두세 건을 읽어 보고 고름")
        return
    if a.domain and config.MY.get("domain") and config.MY["domain"] != a.domain and any(config.PATHS["results"].glob("*.json")):
        print(f"[주의] 도메인을 {config.MY['domain']} → {a.domain} 로 바꿈. results/ 의 이전 결과는 예전 도메인 것이므로 "
              f"폴더 이름을 바꿔 두고(mv results results_{config.MY['domain']}; mkdir results) 새로 시작함")
    if changed:
        MY_PATH.write_text(json.dumps(my, ensure_ascii=False, indent=1), encoding="utf-8")
    ds = read_json(config.PATHS["data"] / domain / "test_dataset.json")
    if a.rebuild or (a.domain and len(ds) <= 10) or (a.seat is not None and changed and len(ds) <= 30 and a.domain):
        ds = build_dataset(domain, int(my.get("seat") or 0))
        print(f"[시험지] data/{domain}/test_dataset.json {len(ds)}건(시드 10 + 문항 풀 {len(ds) - 10}, 번호 {my.get('seat') or 0})")
    shots = f"({my['shots']})" if my.get("shots") is not None else ""
    print(f"[내 설정] 도메인 {domain}({DOMAINS[domain]}) · 모델 {my.get('model_id') or config.MODEL_ID} · {my.get('quant') or config.QUANT}"
          f" · 방식 {my.get('strategy') or 'instruction'}{shots} · 규칙 파일 {my.get('prompt') or '없음'} · 이름 {my.get('name') or 'me'}")
    status(domain)


if __name__ == "__main__":
    main()
