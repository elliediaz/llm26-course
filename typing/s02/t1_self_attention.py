"""[따라 치기 2-T1] Self-Attention 직접 계산 (2회차 실습 1 ②)

실행: python typing/s02/t1_self_attention.py      (빈칸 ____을 채운 뒤)
확인: ① 점수표 s는 (토큰 수 × 토큰 수): 모든 쌍의 유사도
      ② causal mask로 대각선 위쪽(미래)이 0이 됨
      ③ 행마다 합이 1 → 각 토큰이 '어느 토큰을 볼지'를 확률로 나눠 가짐(부드러운 분류)
"""
import math

import torch

torch.manual_seed(0)
T, d = 6, 8                                      # 토큰 6개, 차원 8
x = torch.randn(1, T, d)                          # (배치, 토큰, 차원)
Wq, Wk, Wv = (torch.nn.Linear(d, d, bias=False) for _ in range(3))

q, k, v = Wq(x), Wk(x), Wv(x)
s = q @ k.____(-2, -1) / math.sqrt(d)        # (1, T, T) 모든 쌍의 유사도
mask = torch.ones(T, T).____().bool()             # 아래 삼각형만 True: 자기와 과거만 봄
s = s.____(~mask, float("-inf"))           # 미래 자리는 -inf → softmax 후 0
a = s.____(dim=-1)                             # 행마다 합 1
y = ____                                         # 가중합: 새 표현

torch.set_printoptions(precision=2, sci_mode=False)
print("주의 가중치(행: 만드는 토큰, 열: 보는 토큰)\n", a[0])
print("행 합:", a[0].sum(-1))
print("출력 모양:", tuple(y.shape))
