#!/usr/bin/env bash
# 수업 환경 구성(볼륨마다 1번): Network Volume(/workspace)에 venv·모델 캐시·커널을 모두 둔다.
# 사용: bash setup/setup_pod.sh [모델목록파일(기본: setup/models.txt)]
#   - pods.py create --bootstrap 으로 만든 Pod 는 첫 기동 때 자동 실행된다(/workspace/.setup_done 이 없을 때만).
#   - 다시 실행해도 안전하다(멱등). 로그: /workspace/setup.log
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=/workspace/venv
export HF_HOME=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip

# 1) venv: 이미지의 torch 를 쓰기 위해 --system-site-packages, torch 는 constraints 로 고정
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install -q -U pip wheel
"$VENV/bin/pip" install -q -r "$HERE/requirements-course.txt" -c "$HERE/constraints-course.txt"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 2) Jupyter 커널: 컨테이너 디스크(~/.local)가 아니라 볼륨 안에 등록 → 정지·재생성 후에도 유지
#    Pod 생성 시 env JUPYTER_PATH=/workspace/venv/share/jupyter 로 이 위치를 Jupyter 가 찾게 한다(pods.py)
#    노트북의 !python·!pip 도 venv 를 쓰도록 커널 환경변수에 PATH·HF_HOME 을 넣는다(2026-11-15 점검에서 확인한 문제)
"$VENV/bin/python" -m ipykernel install --prefix "$VENV" --name llm-pbl --display-name "LLM-PBL (course)" \
  --env PATH "$VENV/bin:$PATH" --env HF_HOME /workspace/.cache/huggingface --env PIP_CACHE_DIR /workspace/.cache/pip

# 3) 터미널용 활성화 스크립트
#    수업 저장소 갱신(빨리 감기만)도 여기서 함. 볼륨을 다른 서버로 옮기면 소유자가 달라 safe.directory 가 필요함
cat > /workspace/activate.sh <<'EOS'
source /workspace/venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface PIP_CACHE_DIR=/workspace/.cache/pip
git config --global --add safe.directory /workspace/course 2>/dev/null
git -C /workspace/course pull -q --ff-only 2>/dev/null || true
cd /workspace/course 2>/dev/null || true
EOS

# 4) 모델 사전 시딩
#    models.txt 한 줄 형식: "repo_id [패턴 ...]". 패턴 앞에 '-' 를 붙이면 제외(--exclude), 아니면 포함(--include)
#    huggingface_hub 1.x CLI 는 패턴마다 --include/--exclude 를 따로 붙여야 한다(여러 패턴을 한 옵션 뒤에 나열하면
#    첫 패턴만 include로 쓰이고 나머지는 파일명으로 해석됨).
MODELS="${1:-$HERE/models.txt}"
REPOS=()
if [ -f "$MODELS" ]; then
  set -f                                   # 패턴(*.json 등)이 현재 폴더 파일로 펼쳐지지 않게
  while IFS= read -r line; do
    line="${line%%#*}"                     # 줄 끝 주석 제거
    read -r repo rest <<< "$line" || true
    [ -z "${repo:-}" ] && continue
    args=()
    for pat in ${rest:-}; do
      if [ "${pat:0:1}" = "-" ]; then args+=(--exclude "${pat:1}"); else args+=(--include "$pat"); fi
    done
    echo "[시딩] $repo ${args[*]:-}"
    "$VENV/bin/hf" download "$repo" "${args[@]}" > /dev/null
    REPOS+=("$repo|${rest:-}")              # 검증 단계에 같은 패턴을 넘긴다
  done < "$MODELS"
  set +f
fi

# 5) 시딩 검증: 각 저장소가 캐시에서 '오프라인으로' 열리고 필수 파일이 있는지 확인(덜 받으면 여기서 실패)
#    제외 패턴으로 받은 저장소는 같은 패턴을 줘야 IncompleteSnapshotError가 나지 않는다(hub 1.33 기준)
"$VENV/bin/python" - "${REPOS[@]}" <<'EOP'
import os, sys
from huggingface_hub import snapshot_download
bad = []
for spec in sys.argv[1:]:
    repo, _, pats = spec.partition("|")
    inc = [p for p in pats.split() if not p.startswith("-")] or None
    exc = [p[1:] for p in pats.split() if p.startswith("-")] or None
    try:
        path = snapshot_download(repo, local_files_only=True, allow_patterns=inc, ignore_patterns=exc)
    except Exception as e:
        bad.append(f"{repo}: 캐시 없음 ({e.__class__.__name__})"); continue
    files = set()
    for root, _, fs in os.walk(path):
        files.update(fs)
    has_cfg = "config.json" in files or "modules.json" in files
    has_w = any(f.endswith((".safetensors", ".bin")) for f in files)
    has_tok = any(f.startswith(("tokenizer", "vocab", "source.spm", "spiece", "sentencepiece")) for f in files)
    if not (has_cfg and has_w and has_tok):
        bad.append(f"{repo}: config={has_cfg} weights={has_w} tokenizer={has_tok}")
if bad:
    print("[실패] 시딩 불완전:\n  " + "\n  ".join(bad)); sys.exit(1)
print(f"[통과] 시딩 검증 통과 ({len(sys.argv) - 1}개 저장소)")
EOP

du -sh /workspace/venv /workspace/.cache/huggingface 2>/dev/null || true
date -Is > /workspace/.setup_done
echo "완료: 노트북 커널 'LLM-PBL (course)' 선택, 터미널은 'source /workspace/activate.sh'"
