"""[따라 치기 T1] 몸통은 같고 출력층만 다르다: Dense 신경망으로 분류와 회귀 (1회차 2교시 실습 A, 5분)

실행: python typing/s01/t1_dense_heads.py      (빈칸 ____을 채운 뒤)
확인: ① 두 모델의 층별 출력 모양이 마지막 줄만 다름: 분류 (N,2), 회귀 (N,1)
      ② 분류 손실(교차 엔트로피)과 회귀 손실(평균제곱오차)이 각각 숫자 하나로 나옴
      ③ 분류 출력에 softmax를 씌우면 두 칸의 합이 1(이탈·유지 확률)
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
import nn_tools as T  # noqa: E402

N_IN = 12                                   # 유저 활동 지표 12개(data/G/churn.csv)
x = torch.randn(8, N_IN)                    # 유저 8명 묶음(실습 A 에서는 실제 데이터)


def body():
    return nn.Sequential(nn.Linear(____, 64), nn.____(), nn.Linear(64, 32), nn.ReLU())


cls_model = nn.Sequential(body(), nn.Linear(32, ____))   # 분류: 이탈·유지 두 칸의 점수
reg_model = nn.Sequential(body(), nn.Linear(32, ____))   # 회귀: 다음 주 플레이 시간 숫자 하나

print("[분류 모델]")
T.shape_trace(cls_model, x)
print("[회귀 모델]")
T.shape_trace(reg_model, x)

y_cls = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0])                 # 이탈 여부(정답 클래스 번호)
y_reg = torch.tensor([42.0, 3.0, 55.0, 18.0, 0.0, 61.0, 35.0, 27.0])  # 다음 주 일평균 플레이 분

loss_cls = nn.____()(cls_model(x), y_cls)
loss_reg = nn.____()(reg_model(x).squeeze(1), y_reg)
print("분류 손실(교차 엔트로피):", round(loss_cls.item(), 3))
print("회귀 손실(평균제곱오차):", round(loss_reg.item(), 1))

p = torch.softmax(cls_model(x), dim=1)
print("첫 유저의 [유지, 이탈] 확률:", [round(v, 3) for v in p[0].tolist()], "합", round(p[0].sum().item(), 3))
