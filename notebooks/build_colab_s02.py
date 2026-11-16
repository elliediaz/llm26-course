"""2회차 실습 노트북(코랩·개인 PC·실습 서버 공용) 생성: notebooks/s02_colab.ipynb

따라 치기 셀은 typing/s01/의 실제 파일 내용을 %%writefile 셀로 넣는다(파일과 노트북이 어긋나지 않게).
고칠 때는 이 파일과 typing/ 원본을 고친 뒤 다시 생성한다: python notebooks/build_colab_s02.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "s02_colab.ipynb"
COLAB_URL = "https://colab.research.google.com/github/elliediaz/llm26-course/blob/main/notebooks/s02_colab.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


def typing_cell(rel: str) -> None:
    body = (ROOT / rel).read_text(encoding="utf-8").rstrip("\n")
    code(f"%%writefile {rel}\n{body}")


md("""
# 2회차 실습 노트북

실습가이드의 명령을 순서대로 셀로 옮긴 노트북임. 코랩, 개인 PC, 실습 서버에서 똑같이 쓸 수 있음

- 셀은 위에서부터 차례로 Shift + Enter로 실행함
- `%%writefile` 셀은 빈칸 `____`을 채운 뒤 실행하고, 바로 아래 셀로 결과를 봄
- 연결이 끊기면 0번 셀부터 다시 실행함
- 작업을 남기려면 파일 → Drive에 사본 저장. 결과 파일은 맨 아래 셀로 내려받음
""")
md("""
## 0. 준비
- 런타임 유형은 T4 GPU 권장(런타임 → 런타임 유형 변경). CPU로도 됨
- 아래 두 셀을 실행해 판정이 '통과'인지 봄. '확인 필요'가 나오면 TA를 부름
""")
code(r'''
import os, sys, shutil, subprocess, threading
from IPython.display import Image, Markdown, display

REPO = "https://github.com/elliediaz/llm26-course.git"
if os.path.isfile("../src/config.py"):          # 저장소 안에서 열었을 때
    HOME = os.path.abspath("..")
elif os.path.isdir("/workspace/course"):        # 실습 서버
    HOME = "/workspace/course"
else:                                           # 코랩
    HOME = "/content/course" if os.path.isdir("/content") else os.path.abspath("llm26-course")
    if not os.path.isdir(HOME):
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, HOME], check=True)
    else:
        subprocess.run(["git", "-C", HOME, "pull", "-q"])
os.chdir(HOME)
print("작업 폴더:", os.getcwd())
os.makedirs("results/fig", exist_ok=True)


def show(*paths, width=760):
    for p in paths:
        display(Image(p, width=width)) if os.path.exists(p) else print("아직 없음:", p)


def tail(path, n=8):
    if not os.path.exists(path):
        print("아직 없음:", path)
        return
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    print("\n".join(lines[-n:]))


def use_ref():
    # 학습이 늦을 때 참고 결과를 복사함(내 결과는 덮어쓰지 않음)
    n = 0
    for f in sorted(os.listdir("ref/s01")):
        dst = {".json": "results", ".png": "results/fig", ".pt": "runs"}.get(os.path.splitext(f)[1])
        if dst and not os.path.exists(os.path.join(dst, f)):
            os.makedirs(dst, exist_ok=True)
            shutil.copy(os.path.join("ref/s01", f), dst)
            n += 1
    print(f"참고 결과 {n}개 복사(이미 있던 파일은 그대로 둠)")
''')
code(r'''
%pip install -q koreanize-matplotlib sentence-transformers
!python src/check_gpu.py
''')

md("""
## 실습 1. 토큰 수와 Self-Attention
- ① 같은 문장이 모델마다 몇 토큰으로 쪼개지는지 셈(처음 실행 때 토크나이저를 내려받음)
""")
code("""
!python src/inspect_tokenizer.py
!python src/inspect_tokenizer.py --text "녹스 궁 쿨 몇초임"
""")
md("""
- ② 따라 치기 2-T1: Self-Attention을 직접 계산함(빈칸을 채운 뒤 실행). 같은 계산이 실제 층 안에 있는지 `--info`로 확인함
""")
typing_cell("typing/s02/t1_self_attention.py")
code("""
!python typing/s02/t1_self_attention.py
!python src/mini_transformer.py --info
""")
md("""
- ③ Self-Attention과 Cross-Attention 비교 그림(처음 실행 때 번역 모델을 내려받음)
""")
code("""
!python src/viz_attention.py compare --text "결제했는데 아이템이 안 들어왔어요"
show("results/fig/attn_compare.png")
""")

md("""
## 휴식 전 걸어 두기: 생성 설정 비교
- 같은 질문을 설정만 바꿔 여러 번 생성함(코랩 CPU 약 13분, T4면 더 짧음)
- 쉬는 동안 탭을 닫지 않음. 도는 동안 다른 생성·학습 셀은 실행하지 않음
""")
code(r'''
def _run_sweep():
    with open("results/log_gen_sweep.txt", "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "src/gen_params_sweep.py", "--model", "Qwen/Qwen3-0.6B"], stdout=f, stderr=subprocess.STDOUT)
    open("results/bg_done.txt", "w").write("done")


if os.path.exists("results/bg_done.txt"):
    os.remove("results/bg_done.txt")
threading.Thread(target=_run_sweep, daemon=True).start()
print("뒤에서 생성 설정 비교 시작. 진행 확인은 다음 셀")
''')
code("""
tail("results/log_gen_sweep.txt", 5)
print("\\n모두 끝남" if os.path.exists("results/bg_done.txt") else "\\n아직 생성 중")
""")

md("""
## 해보기 ①. MiniGPT 설정과 학습 글 바꾸기
- 기본값으로 한 번, 그다음 값 하나만 바꿔 다시 학습함(한 번 약 95초)
- 생성 설정 비교가 끝난 뒤에 실행함
""")
code("""
!python src/mini_transformer.py --train 1.5
# !python src/mini_transformer.py --train 1.5 --layers 2
# !python src/mini_transformer.py --train 1.5 --corpus nsmc
""")
code("""
display(Markdown(open("results/mini_gpt_runs.md", encoding="utf-8").read()))
show("results/fig/mini_gpt_loss.png")
""")

md("""
## 실습 2. 문맥 내 학습과 고르는 법
- ④ 예시 개수에 따라 Base 모델의 답이 어떻게 바뀌는지, Base와 Instruct가 같은 질문에 어떻게 다르게 답하는지 봄
""")
code("""
sys.path.insert(0, os.path.abspath("src"))
import config, utils, lineage, model_utils
base, base_tok = model_utils.load_llm(config.BASE_MODEL_ID, quant="bf16")
test = utils.read_json(config.data_dir() / "test_seed.json")
train = utils.read_jsonl(config.data_dir() / "train_seed.jsonl")
shots, seen = [], set()
for r in train:                                   # 분류가 서로 다른 예시 7개(분류마다 1개)
    if r["gold"]["category"] not in seen:
        seen.add(r["gold"]["category"]); shots.append(r)
for k in (0, 3, 7):
    r = lineage.icl_accuracy(base, base_tok, test, shots[:k])
    print(f"예시 {k}개 → 형식 준수 {r['format_ok']:.0%} · 정확도 {r['accuracy']:.0%}")
""")
code("""
q = "결제했는데 루멘석이 안 들어왔어요. 어떻게 해야 하나요?"
print("[Base 모델, 그대로 입력]\\n", lineage.chat_raw_vs_template(base, base_tok, q, use_template=False))
inst, inst_tok = model_utils.load_llm(config.SMALL_MODEL_ID, quant="bf16")
print("\\n[Instruct 모델, Chat Template 적용]\\n", lineage.chat_raw_vs_template(inst, inst_tok, q, use_template=True))
""")
md("""
- ⑤ 따라 치기 2-T2: 다음 토큰을 고르는 법(greedy, temperature, top-k, top-p)을 직접 구현함
""")
typing_cell("typing/s02/t2_sampling.py")
code("!python typing/s02/t2_sampling.py")
md("""
- ⑥ 생성 설정 비교 결과(뒤에서 돈 것)와 7B 모델 결과(강사 서버 실측)를 나란히 봄
""")
code("""
for p in ("results/gen_sweep.md", "ref/s02/gen_sweep_main_7b.md"):
    print("=====", p); print(open(p, encoding="utf-8").read() if os.path.exists(p) else "아직 없음")
""")

md("""
## 실습 3. KV Cache와 문맥 주입
- ⑥ KV Cache를 켜고 끌 때의 속도와 메모리를 잼(약 3분)
""")
code("""
!python src/mini_transformer.py --kv
!python src/kv_cache_bench.py --lengths 128 512
show("results/fig/kv_cache.png")
""")

md("""
## 해보기 ②. 팀 생성 설정과 KV Cache 손계산
- D(필수, 팀): 형식형 문의로 내가 고른 설정 하나를 greedy와 비교하고 `GEN_SERVICE`를 정함(한 번 약 60초)
""")
code("""
!python src/gen_params_sweep.py --model Qwen/Qwen3-0.6B --kind 형식형 --temperature 0.5 --top-p 0.8
print(open("results/gen_sweep_mine.md", encoding="utf-8").read() if os.path.exists("results/gen_sweep_mine.md") else "아직 없음")
""")

md("""
## 결과 검증과 인계
- 3회차부터 팀 GPU 서버로 넘어가므로 오늘 결과가 다 있는지 점검함
""")
code("!python src/resume_check.py --handover 2")

md("""
## 결과 내려받기
- 코랩은 연결이 끝나면 파일이 사라지므로 수업을 마치기 전에 실행함
""")
code(r'''
shutil.make_archive("s02_results", "zip", ".", "results")
try:
    from google.colab import files
    files.download("s02_results.zip")
except ImportError:
    print("저장 위치:", os.path.abspath("s02_results.zip"))
''')

md("""
## 문제가 생기면
| 증상 | 조치 |
|---|---|
| `No such file or directory` | 0번 셀부터 다시 실행 |
| 연결이 끊김 | 다시 연결하고 0번 셀부터 실행. 생성 설정 비교는 다시 걸어 둠 |
| GPU를 쓸 수 없다는 안내 | CPU로 그대로 진행 |
| 셀이 끝나지 않음 | 셀 왼쪽 ■를 누르고 TA를 부름 |
| `Rate limited` 문구 | 1~2분 기다리면 이어짐 |
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "toc_visible": True}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = hashlib.md5(f"s02_colab-{i}".encode()).hexdigest()[:8]
    src = c["source"].split("\n")
    c["source"] = [ln + "\n" for ln in src[:-1]] + [src[-1]]
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("생성:", OUT.name, len(cells), "셀")
