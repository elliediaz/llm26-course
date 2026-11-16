"""[따라 치기 5-T2] 필드별 채점 함수: 분류 필드는 정답 여부, 숫자 필드는 오차 (5회차 코드 구조)

1회차 명제 복습: JSON의 각 필드는 분류(정해진 후보 중 하나) 아니면 숫자·글자를 옮기는 것.
실행: python typing/s05/t2_field_scorer.py      (빈칸 ____을 채운 뒤)
확인: 필드별 0/1 점수 → 가중 평균이 Task Accuracy. 실제 채점은 src/scorer.py + data/<도메인>/scoring.yaml
"""
import re

RULES = {"category": ("enum", 2), "action": ("enum", 2), "priority": ("enum", 1),
         "is_intended": ("bool", 1), "fact": ("numbers", 1)}


def nums(v) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", str(v or "")))


def field_score(kind, pred, gold) -> float:
    if gold is None:                                   # 정답이 '없음'이면 출력도 비어 있어야 정답
        return float(pred in (None, ""))
    if kind in ("enum", "bool"):                       # 분류: 맞으면 1, 틀리면 0
        return float(____)
    if kind == "numbers":                              # 정답 숫자가 출력에 모두 있으면 정답
        return float(nums(gold) ____ nums(pred))
    if kind == "numeric":                              # 회귀형: 허용 오차 안이면 정답
        return float(abs(float(pred) - float(gold)) <= 0.01 * abs(float(gold)))
    return float(str(pred).strip() == str(gold).strip())


gold = {"category": "밸런스", "action": "안내", "priority": "P3", "is_intended": True, "fact": "공격력 120→105"}
pred = {"category": "버그", "action": "안내", "priority": "P3", "is_intended": False, "fact": "공격력이 120에서 105로"}
total = weight = 0
for f, (kind, w) in RULES.items():
    s = field_score(kind, pred.get(f), gold.get(f))
    total += ____
    weight += w
    print(f"{f:12s} {kind:8s} 정답 {gold[f]!s:16s} 출력 {pred[f]!s:22s} → {s:.0f}")
print("Task Accuracy(가중 평균):", round(total / weight, 3))
