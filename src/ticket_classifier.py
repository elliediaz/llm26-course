"""1회차 실습 2 ⑥: LLM 없이 '고전적인' 분류기·회귀기로 게임 문의 티켓을 판정한다.

  python src/ticket_classifier.py                    # 임베딩(KURE) 특징 + 로지스틱 회귀/릿지
  python src/ticket_classifier.py --features tfidf   # 비교: 글자 2-gram TF-IDF 특징(키워드 겹침)
  python src/ticket_classifier.py --features both    # 두 특징을 나란히(입력 표현이 좋아지면 분류가 좋아지는가)

데이터  data/G/train_seed.jsonl(185건) → 학습 150 · 검증 35(시드 고정)
        data/G/test_seed.json(10건) → 5회차에 LLM이 풀 문항과 같음(같은 문항으로 나중에 비교)
판정할 필드(5회차 JSON의 일부)
  category  7개 중 하나        → 분류(로지스틱 회귀)
  action    6개 중 하나        → 분류
  priority  P1~P4(순서 있음)   → 분류로도, 회귀(1~4 숫자, 릿지)로도 풀어 비교
확인할 것
  - 분류의 지표: 정확도·macro-F1·혼동행렬 / 회귀의 지표: MAE(평균 절대 오차)
  - 순서가 있는 값(우선순위·별점)은 둘 다 가능하다: 무엇을 잘못으로 볼지(한 단계 차이도 오답인가)에 따라 고른다
  - 같은 분류기라도 입력 표현(TF-IDF vs 임베딩)이 좋아지면 성능이 오른다 → 발전사의 핵심
결과: results/ticket_clf.json  (5회차 src/field_report.py가 LLM 필드 정확도와 나란히 비교)
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

import config
import utils

PRI = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
INV = {v: k for k, v in PRI.items()}


def load(seed: int = config.SEED):
    rows = utils.read_jsonl(config.data_dir("G") / "train_seed.jsonl")
    rng = random.Random(seed)
    rng.shuffle(rows)
    test = utils.read_json(config.data_dir("G") / "test_seed.json")
    return rows[:150], rows[150:], test


def featurizer(kind: str, fit_texts: list[str]):
    """문장 → 고정 길이 벡터. kind: embed(KURE-v1, 1024차원) | tfidf(글자 2-gram)."""
    if kind == "embed":
        import model_utils
        emb = model_utils.load_embedder()
        return lambda texts: emb.encode(texts, normalize_embeddings=True, batch_size=64)
    from faq_search import TfidfIndex
    idx = TfidfIndex(fit_texts)
    return idx.encode


def run(kind: str, train, valid, test) -> dict:
    enc = featurizer(kind, [r["input"] for r in train])
    X = {n: np.asarray(enc([r["input"] for r in d])) for n, d in (("train", train), ("valid", valid), ("test", test))}
    gold = lambda d, f: [r["gold"][f] for r in d]  # noqa: E731
    out: dict = {"features": kind, "dim": int(X["train"].shape[1]), "fields": {}}

    # 1) 분류: category · action
    for field in ("category", "action"):
        clf = LogisticRegression(max_iter=3000, C=10).fit(X["train"], gold(train, field))
        pv, pt = clf.predict(X["valid"]), clf.predict(X["test"])
        labels = sorted(set(gold(train, field)) | set(gold(valid, field)))
        out["fields"][field] = {
            "kind": "분류", "classes": len(clf.classes_),
            "valid_acc": round(accuracy_score(gold(valid, field), pv), 3),
            "valid_macro_f1": round(f1_score(gold(valid, field), pv, average="macro", zero_division=0), 3),
            "test_acc": round(accuracy_score(gold(test, field), pt), 3),
            "labels": labels,
            "confusion": confusion_matrix(gold(valid, field), pv, labels=labels).tolist(),
            "test_pred": [str(x) for x in pt],
        }

    # 2) priority: 분류(4클래스) vs 회귀(숫자 1~4)
    yt = [PRI[p] for p in gold(train, "priority")]
    yv, yte = [PRI[p] for p in gold(valid, "priority")], [PRI[p] for p in gold(test, "priority")]
    clf = LogisticRegression(max_iter=3000, C=10).fit(X["train"], yt)
    reg = Ridge(alpha=1.0).fit(X["train"], yt)
    pc_v, pc_t = clf.predict(X["valid"]), clf.predict(X["test"])
    raw_v, raw_t = reg.predict(X["valid"]), reg.predict(X["test"])
    pr_v = np.clip(np.rint(raw_v), 1, 4).astype(int)
    pr_t = np.clip(np.rint(raw_t), 1, 4).astype(int)
    within1 = lambda p, g: float(np.mean(np.abs(np.asarray(p) - np.asarray(g)) <= 1))  # noqa: E731
    out["fields"]["priority"] = {
        "분류": {"valid_acc": round(accuracy_score(yv, pc_v), 3), "valid_mae": round(mean_absolute_error(yv, pc_v), 3),
                 "valid_within1": round(within1(pc_v, yv), 3), "test_acc": round(accuracy_score(yte, pc_t), 3),
                 "test_pred": [INV[int(x)] for x in pc_t]},
        "회귀": {"valid_acc": round(accuracy_score(yv, pr_v), 3), "valid_mae": round(mean_absolute_error(yv, raw_v), 3),
                 "valid_within1": round(within1(pr_v, yv), 3), "test_acc": round(accuracy_score(yte, pr_t), 3),
                 "test_raw": [round(float(x), 2) for x in raw_t], "test_pred": [INV[int(x)] for x in pr_t]},
    }
    return out


def report(res: dict, valid) -> None:
    print(f"\n## 특징: {res['features']} ({res['dim']}차원)")
    rows = []
    for f in ("category", "action"):
        r = res["fields"][f]
        rows.append({"필드": f, "방식": f"분류({r['classes']}클래스)", "검증 정확도": r["valid_acc"],
                     "검증 macro-F1": r["valid_macro_f1"], "시드10 정확도": r["test_acc"]})
    for k in ("분류", "회귀"):
        r = res["fields"]["priority"][k]
        rows.append({"필드": "priority", "방식": k, "검증 정확도": r["valid_acc"], "검증 MAE": r["valid_mae"],
                     "한 단계 이내": r["valid_within1"], "시드10 정확도": r["test_acc"]})
    utils.print_table(rows, ["필드", "방식", "검증 정확도", "검증 macro-F1", "검증 MAE", "한 단계 이내", "시드10 정확도"])
    c = res["fields"]["category"]
    print("\n혼동행렬(category, 검증 35건; 행: 정답, 열: 예측)")
    print("        " + " ".join(f"{x[:2]:>4s}" for x in c["labels"]))
    for lab, line in zip(c["labels"], c["confusion"]):
        print(f"{lab[:3]:>6s}  " + " ".join(f"{v:4d}" for v in line))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=["embed", "tfidf", "both"], default="embed")
    a = ap.parse_args()
    utils.set_seed()
    train, valid, test = load()
    print(f"[데이터] 학습 {len(train)} · 검증 {len(valid)} · 시드 {len(test)}")
    print("category 분포(학습):", dict(Counter(r["gold"]["category"] for r in train).most_common()))
    kinds = ["tfidf", "embed"] if a.features == "both" else [a.features]
    results = []
    for k in kinds:
        with utils.section(f"특징 {k}"):
            res = run(k, train, valid, test)
            report(res, valid)
            results.append(res)
    best = results[-1]
    preds = [{"id": t["id"], "input": t["input"], "gold": {f: t["gold"][f] for f in ("category", "action", "priority")},
              "pred": {"category": best["fields"]["category"]["test_pred"][i],
                       "action": best["fields"]["action"]["test_pred"][i],
                       "priority": best["fields"]["priority"]["분류"]["test_pred"][i]}} for i, t in enumerate(test)]
    out = utils.write_json(utils.result_path("ticket_clf.json"),
                           {"runs": results, "test_predictions": preds, "note": "5회차 field_report.py 비교용"})
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
