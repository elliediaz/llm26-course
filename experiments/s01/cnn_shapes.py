"""1회차 실습 B 보조: CNN 안에서 이미지 모양과 파라미터 수가 어떻게 바뀌는지 설정을 바꿔 가며 봄

  python experiments/s01/cnn_shapes.py                              # 채널 16·32, 3×3 필터, 패딩 1, 풀링
  python experiments/s01/cnn_shapes.py --kernel 5 --padding 0        # 큰 필터, 패딩 없음: 가로세로가 줄어듦
  python experiments/s01/cnn_shapes.py --channels 8 16 32            # 합성곱 층 하나 더
  python experiments/s01/cnn_shapes.py --no-pool                     # 풀링을 빼면 Linear 입력이 얼마나 커지나
  python experiments/s01/cnn_shapes.py --compare-mlp                 # 같은 일을 하는 MLP와 파라미터 수 비교

볼 것
  ① [N,1,28,28]에서 시작해 층마다 [N,채널,높이,너비]가 바뀜. 펼친 크기가 Linear 입력 크기와 같아야 함
  ② 합성곱 층 파라미터 = 필터 크기² × 입력 채널 × 출력 채널 + 출력 채널. 이미지 크기와 상관없음
  ③ 풀링을 빼거나 패딩을 바꾸면 합성곱 파라미터는 그대로인데 Linear 파라미터가 크게 바뀜
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import nn_tools as T  # noqa: E402


def build(channels, kernel, padding, pool, size, classes=10):
    layers, c, h = [], 1, size
    for ch in channels:
        layers += [nn.Conv2d(c, ch, kernel_size=kernel, padding=padding), nn.ReLU()]
        h = h + 2 * padding - kernel + 1
        if pool:
            layers.append(nn.MaxPool2d(2))
            h //= 2
        c = ch
        if h < 1:
            raise SystemExit(f"이미지가 사라짐(높이 {h}). 필터를 줄이거나 패딩을 늘리거나 풀링을 빼 볼 것")
    flat = c * h * h
    layers += [nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, classes)]
    return nn.Sequential(*layers), (c, h, flat)


def params(model) -> list[tuple[str, int]]:
    return [(f"{i}:{m.__class__.__name__}", sum(p.numel() for p in m.parameters())) for i, m in enumerate(model) if any(True for _ in m.parameters())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--padding", type=int, default=1)
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--size", type=int, default=28)
    ap.add_argument("--compare-mlp", action="store_true")
    a = ap.parse_args()

    model, (c, h, flat) = build(a.channels, a.kernel, a.padding, not a.no_pool, a.size)
    T.shape_trace(model, torch.rand(4, 1, a.size, a.size))
    print(f"\n펼치기 직전 [{c}, {h}, {h}] → {c}×{h}×{h} = {flat:,} = 첫 Linear 입력 크기")
    rows = params(model)
    conv = sum(n for name, n in rows if "Conv" in name)
    lin = sum(n for name, n in rows if "Linear" in name)
    for name, n in rows:
        print(f"  {name:14s} 파라미터 {n:>10,}")
    print(f"합성곱 {conv:,} · Linear {lin:,} · 전체 {conv + lin:,}")
    if a.compare_mlp:
        mlp = nn.Sequential(nn.Flatten(), nn.Linear(a.size * a.size, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
        print(f"같은 10종 분류를 MLP(784→256→128→10)로 하면 파라미터 {sum(p.numel() for p in mlp.parameters()):,}개. 정확도 비교는 src/nn_cnn.py")


if __name__ == "__main__":
    main()
