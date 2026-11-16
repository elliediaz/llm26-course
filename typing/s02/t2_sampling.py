"""[따라 치기 2-T2] 같은 확률 분포에서 토큰을 고르는 방법 = 생성 설정 (2회차 실습 2 ⑤)

1회차 T3에서 LLM은 다음 토큰을 고르는 분류기였다. 분류 결과(확률 분포)는 하나인데,
그중 무엇을 고를지(greedy·Temperature·Top-k·Top-p)에 따라 출력이 달라진다.
실행: python typing/s02/t2_sampling.py      (빈칸 ____을 채운 뒤)
확인: Temperature가 높을수록·Top-p가 클수록 '서로 다른 토큰 수'가 늘어남. greedy는 항상 1
"""
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32).to(DEV)
ids = tok("이번 패치에서 가장 크게 달라진 점은", return_tensors="pt").to(DEV)
with torch.no_grad():
    logits = model(**ids).logits[0, -1].float()


def pick(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature == 0:
        return logits.argmax()                                    # greedy: 항상 1등
    probs = torch.softmax(logits ____ temperature, dim=-1)            # 온도: 분포를 날카롭게(<1) 또는 평평하게(>1)
    if top_k:
        kth = torch.topk(probs, top_k).values[-1]
        probs[probs < kth] = 0                                    # 상위 k개만 남김
    if top_p < 1.0:
        s, i = torch.sort(probs, descending=True)
        drop = s.____(0) - s >= top_p                           # 누적 확률이 p를 넘은 뒤의 후보는 버림
        probs[i[drop]] = 0
    probs = probs / probs.sum()
    return torch.____(probs, 1)[0]                         # 확률대로 하나 뽑기


torch.manual_seed(42)
for setting in [dict(temperature=0), dict(temperature=0.3), dict(temperature=1.0), dict(temperature=1.5),
                dict(temperature=1.0, top_k=5), dict(temperature=1.0, top_p=0.9)]:
    picks = [tok.decode(pick(logits.clone(), **setting)) for _ in range(30)]
    print(f"{str(setting):38s} 서로 다른 토큰 {len(set(picks)):2d}개 · 많이 나온 것 {Counter(picks).most_common(3)}")
