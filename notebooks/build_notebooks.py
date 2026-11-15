"""notebooks/src/*.py(퍼센트 형식) → notebooks/*.ipynb 변환.

노트북 내용은 .py 원본에서 고친다(git에서 변경 내용을 읽기 쉽게). 변환: python notebooks/build_notebooks.py
형식: '# %%'는 코드 셀, '# %% [markdown]'는 마크다운 셀(각 줄 앞의 '# '는 제거). jupytext의 percent 형식과 같다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = {"display_name": "LLM-PBL (course)", "language": "python", "name": "llm-pbl"}


def parse(text: str) -> list[dict]:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        src = [ln + "\n" for ln in lines]
        if src:
            src[-1] = src[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def build(src: Path) -> Path:
    nb = {"cells": parse(src.read_text(encoding="utf-8")),
          "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = hashlib.md5(f"{src.stem}-{i}".encode()).hexdigest()[:8]
    out = HERE / (src.stem + ".ipynb")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    for p in sorted((HERE / "src").glob("*.py")):
        print("생성:", build(p).name)
