"""1회차 1교시 실습: GPU·CUDA·PyTorch 환경 점검 → results/env_check.json

실행: python src/check_gpu.py
확인할 것
  - CUDA 사용 가능 여부와 GPU 이름(기준 A100 80GB. 재고 사정으로 대체 GPU 또는 CPU 실습 서버일 수 있음)
  - 전체·가용 VRAM, BF16 지원 여부
  - 사전 시딩된 모델이 캐시에 있는지(setup/models.txt 기준)
  - 간단한 행렬곱 속도(TFLOPS): 같은 GPU인지 대략 확인하는 용도
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time

import config
import utils


def gpu_info() -> dict:
    import torch
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if not torch.cuda.is_available():
        return info
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    info.update({
        "device_count": torch.cuda.device_count(),
        "name": p.name,
        "capability": f"{p.major}.{p.minor}",
        "vram_total_gb": round(total / 1024**3, 1),
        "vram_free_gb": round(free / 1024**3, 1),
        "bf16": torch.cuda.is_bf16_supported(),
    })
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["driver"] = drv
    except Exception:
        pass
    return info


def matmul_tflops(n: int = 8192, reps: int = 10) -> float | None:
    import torch
    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / reps
    return round(2 * n**3 / sec / 1e12, 1)


def seeded_models() -> list[dict]:
    """setup/models.txt의 저장소가 HF 캐시에 있는지 확인(내려받지 않음)."""
    path = config.ROOT / "setup" / "models.txt"
    if not path.exists():
        return []
    try:
        from huggingface_hub import scan_cache_dir
        cached = {r.repo_id: round(r.size_on_disk / 1024**3, 2) for r in scan_cache_dir().repos}
    except Exception:
        cached = {}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        repo = line.split()[0]
        rows.append({"repo": repo, "cached": repo in cached, "size_gb": cached.get(repo)})
    return rows


def main() -> None:
    utils.set_seed()
    env = {
        "checked_at": utils.now(),
        "pod_id": os.environ.get("RUNPOD_POD_ID", "(로컬)"),
        "python": platform.python_version(),
        "hf_home": os.environ.get("HF_HOME", "(기본값)"),
        **gpu_info(),
    }
    for pkg in ("transformers", "peft", "trl", "bitsandbytes", "sentence_transformers", "chromadb"):
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception:
            env[pkg] = None
    du = shutil.disk_usage("/workspace" if os.path.isdir("/workspace") else config.ROOT)
    env["disk_free_gb"] = round(du.free / 1024**3, 1)
    env["matmul_bf16_tflops"] = matmul_tflops()
    env["models"] = seeded_models()

    print("\n[환경점검표]")
    for k in ("pod_id", "name", "vram_total_gb", "vram_free_gb", "bf16", "torch", "cuda_runtime", "driver",
              "transformers", "peft", "trl", "bitsandbytes", "disk_free_gb", "matmul_bf16_tflops"):
        print(f"  {k:20s} {env.get(k)}")
    if env["models"]:
        miss = [m["repo"] for m in env["models"] if not m["cached"]]
        print(f"  {'사전 시딩 모델':20s} {len(env['models']) - len(miss)}/{len(env['models'])}"
              + (f"  (없음: {', '.join(miss)})" if miss else ""))
    models_ok = bool(env["models"]) and all(m["cached"] for m in env["models"])
    if env.get("cuda_available") and (env.get("vram_total_gb") or 0) >= 40 and models_ok:
        verdict = "통과"
    elif env.get("cuda_available") and models_ok:
        verdict = "통과(대체 GPU: 1·2·5·6회차 실습 가능, 7회차 학습 전 A100으로 바꿈)"
    elif models_ok:
        verdict = "통과(CPU 실습 서버: 1·2회차 실습은 그대로 가능, GPU 배정은 TA가 안내)"
    else:
        verdict = "확인 필요: 강사·TA에게 알릴 것"
    env["verdict"] = verdict
    print("\n판정:", verdict)
    out = utils.write_json(utils.result_path("env_check.json"), env)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
