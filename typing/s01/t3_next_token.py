"""[따라 치기 T3] 다음 토큰 분류기로서의 LLM (1회차 3교시 실습 E)

실행: python typing/s01/t3_next_token.py      (빈칸 ____을 채운 뒤)
확인: ① 출력층 크기 = 어휘 크기(약 15만) = 클래스 수
      ② 다음 토큰 확률 5개, 확률 합 = 1
      ③ 생성 = 분류를 15번 반복한 결과
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16 if DEV == "cuda" else torch.float32).to(DEV)

# 1) 출력층의 크기 = 고를 수 있는 '클래스'의 수
print("어휘 크기(클래스 수):", model.config.____)
print("출력층:", model.lm_head)

# 2) 분류 한 번: 마지막 위치에서 다음 토큰의 확률 분포
ids = tok("결제했는데 아이템이 안", return_tensors="pt").to(DEV)
with torch.no_grad():
    logits = model(**ids).logits                      # (1, 입력 길이, 어휘 크기)
probs = torch.____(logits[0, -1].float(), dim=-1)
top = torch.____(probs, 5)
for p, i in zip(top.values, top.indices):
    print(f"{p.item():.3f}  {tok.decode(i)!r}")
print("확률 합:", round(probs.sum().item(), 4))

# 3) 생성 = 분류를 반복: 고른 토큰을 뒤에 붙이고 다시 분류
x = ids["input_ids"]
for _ in range(15):
    with torch.no_grad():
        nxt = model(x).logits[0, -1].____()        # 가장 확률이 높은 클래스(greedy)
    x = torch.____([x, nxt.view(1, 1)], dim=1)
print(tok.decode(x[0]))
