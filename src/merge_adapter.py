"""7회차 도전과제 ②: Adapter를 본체에 병합해 단일 모델로 내보낸다(서비스 배포 때 실제로 밟는 절차).

  python src/merge_adapter.py --adapter adapters/team3/r16

주의
  - 병합은 BF16 본체 위에서 한다(4-bit 본체에는 병합하지 않음). QLoRA로 학습한 Adapter 도 BF16 본체에 병합 가능
  - 결과물은 본체 크기(7~8B 면 약 15GB)다. 볼륨 여유 공간을 먼저 확인할 것: df -h /workspace
  - 병합 전후 같은 하네스 점수가 비슷한지 확인한다(4-bit 학습 → BF16 병합 시 미세한 차이 가능)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config
import model_utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", help="기본: <adapter>-merged")
    a = ap.parse_args()
    out = Path(a.out or f"{a.adapter.rstrip('/')}-merged")
    need = 2.2 * 8  # GB, 8B 기준 대략
    free = shutil.disk_usage(config.ROOT).free / 1024**3
    if free < need:
        raise SystemExit(f"여유 공간 {free:.0f}GB < 필요 약 {need:.0f}GB: 정리 후 다시 실행")
    model, tok = model_utils.load_llm(quant="bf16", adapter=a.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"병합 모델 저장: {out}\n확인: COURSE_MODEL_ID={out} python src/benchmark.py --name merged --no-scoreboard")


if __name__ == "__main__":
    main()
