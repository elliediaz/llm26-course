"""[따라 치기 T2] CNN 안에서 이미지 모양이 어떻게 바뀌나 (1회차 2교시 실습 B, 5분)

실행: python typing/s01/t2_cnn_shapes.py      (빈칸 ____을 채운 뒤)
확인: ① [N,1,28,28] → [N,16,28,28] → [N,16,14,14] → [N,32,14,14] → [N,32,7,7] → [N,1568] → [N,10]
      ② 펼치기 직전 크기 32×7×7 = 1568이 Linear의 입력 크기와 같아야 오류가 나지 않음
      ③ 합성곱 층의 파라미터(160, 4,640)가 Linear 층(200,832)보다 훨씬 적음: 작은 필터를 이미지 전체에 재사용하기 때문
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")
import nn_tools as T  # noqa: E402

x = torch.rand(4, 1, 28, 28)                              # 흑백 옷 사진 4장(Fashion-MNIST 크기)

model = nn.Sequential(
    nn.Conv2d(____, 16, kernel_size=3, padding=1),           # 3×3 필터 16장 → 무늬 지도 16장
    nn.ReLU(),
    nn.____(2),                                      # 가로·세로 절반
    nn.Conv2d(____, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),                                         # [N,32,7,7] → [N,1568]
    nn.Linear(32 * ____ * ____, 128),
    nn.ReLU(),
    nn.Linear(128, ____),                                   # 10종 중 하나: 분류
)
T.shape_trace(model, x)
