#!/usr/bin/env bash
# 1회차 참고 결과를 results/와 runs/에 복사한다. 이미 있는 파일(내가 만든 결과)은 덮어쓰지 않는다.
# 사용: bash ref/s01/use_ref.sh   (작업 폴더 /workspace/course 에서)
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
mkdir -p "$ROOT/results/fig" "$ROOT/runs"
copied=0
put() { if [ ! -e "$2/$(basename "$1")" ]; then cp "$1" "$2/" && copied=$((copied + 1)) && echo "  복사: $2/$(basename "$1")"; fi; }
for f in "$HERE"/*.json; do put "$f" "$ROOT/results"; done
for f in "$HERE"/*.png; do put "$f" "$ROOT/results/fig"; done
for f in "$HERE"/*.pt; do put "$f" "$ROOT/runs"; done
echo "참고 결과 ${copied}개 복사(이미 있던 파일은 그대로 둠)"
