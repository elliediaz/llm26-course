"""공통 도구: 시드 고정, JSON 입출력, 시간·VRAM 측정, 그림 한글 글꼴, 표 출력."""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import config


# ---------------------------------------------------------------- 재현성
def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def free_gpu() -> None:
    """모델을 바꿔 올리기 전에 GPU 메모리를 비움"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 파일
def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def result_path(name: str) -> Path:
    """results/ 아래 경로. 폴더가 없으면 만든다."""
    p = config.PATHS["results"] / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fig_path(name: str) -> Path:
    p = config.PATHS["figs"] / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- 측정
class Timer:
    """with Timer() as t: ...  →  t.sec"""

    def __enter__(self):
        self._sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.sec = time.perf_counter() - self.t0
        return False

    @staticmethod
    def _sync():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass


def vram_reset() -> None:
    """측정 구간 시작 전에 호출. 최대 점유량 기록을 초기화한다."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def vram_peak_gb() -> float | None:
    """측정 구간의 최대 점유량(GB). GPU가 없으면 None."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except ImportError:
        pass
    return None


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def latency_stats(xs: list[float], skip_first: bool = True) -> dict:
    """첫 1건은 워밍업(커널 준비·캐시)이 섞이므로 기본으로 뺀다."""
    data = xs[1:] if skip_first and len(xs) > 1 else xs
    return {
        "p50": percentile(data, 0.5),
        "p95": percentile(data, 0.95),
        "mean": round(statistics.mean(data), 3) if data else None,
        "n": len(data),
    }


@contextlib.contextmanager
def section(title: str):
    print(f"\n=== {title} ===")
    t0 = time.perf_counter()
    yield
    print(f"--- {title}: {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------- 그림
_FONT_READY = False


def korean_font() -> None:
    """matplotlib 한글 글꼴 설정. koreanize-matplotlib → 설치된 Nanum/Noto 순으로 찾는다."""
    global _FONT_READY
    if _FONT_READY:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import koreanize_matplotlib  # noqa: F401  (나눔고딕 포함 패키지)
    except ImportError:
        from matplotlib import font_manager
        for name in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
            if any(name in f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = name
                break
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


# ---------------------------------------------------------------- 출력
def print_table(rows: list[dict], cols: list[str] | None = None) -> str:
    """dict 목록을 마크다운 표로 출력하고 문자열로 돌려준다(결과 기록에 그대로 붙여 쓸 수 있게)."""
    if not rows:
        return ""
    cols = cols or list(rows[0].keys())
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    text = "\n".join(lines)
    print(text)
    return text
