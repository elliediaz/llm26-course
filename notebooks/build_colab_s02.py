"""2회차 실습 노트북 V2(코랩·개인 PC·실습 서버 공용) 생성: notebooks/s02_colab.ipynb

V2: 1회차 정리(복습 실습 R) → 실습 D(1회차 남은 부분) → 트랜스포머 실습 1 → 토큰 실습 2 → 다음 토큰 실습 3 → LLM 감성 분류 실습 F → 해보기 ②
V1(11/16 작성본)은 notebooks/_V1/에 보존함
따라 치기 셀은 typing/의 실제 파일 내용을 %%writefile 셀로 넣는다. 다시 생성: python notebooks/build_colab_s02.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "s02_colab.ipynb"

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
    # 학습이 늦을 때 1회차 참고 결과를 복사함(내 결과는 덮어쓰지 않음)
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
## 복습 실습 R. XOR를 직접 나눠 보기
- 단층 · 층만 쌓기 · 층 + 활성화 함수, 세 모델로 AND·OR·XOR를 학습함(약 10초)
""")
code("""
!python src/xor_demo.py
show("results/fig/xor_boundary.png", width=620)
""")
code("""
!python src/xor_demo.py --hidden 2
# !python src/xor_demo.py --hidden 2 --seed 3
# !python src/xor_demo.py --act sigmoid
""")
md("""
확인 값
- XOR 줄: 단층 0.50, 2층·활성화 X 0.50, 2층·ReLU 1.00
- `--hidden 2`: 시작값에 따라 0.50 · 0.75 · 1.00으로 갈림. `--seed`를 바꿔 다시 실행해 봄
""")

md("""
## 실습 D ①. 기억 길이 시험
- 숫자 여러 개 중 맨 앞 숫자를 맞힘(찍으면 0.1). 거리가 멀어질수록 어떻게 되는지 봄(코랩 CPU 약 1분 30초)
""")
code("""
!python src/seq_limits.py
show("results/fig/seq_limits.png")
""")

md("""
## 휴식 전 걸어 두기(실습 D ②)
- 한글 금액을 숫자로 바꾸는 seq2seq 세 구조를 2분씩 학습함(약 7분)
- 쉬는 동안 탭을 닫지 않음. 도는 동안 다른 학습 셀은 실행하지 않음
""")
code(r'''
def _run_jobs():
    with open("results/log_seq2seq.txt", "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, "src/tiny_seq2seq_train.py", "--arch", "all", "--minutes", "2"],
                       stdout=f, stderr=subprocess.STDOUT)
    open("results/bg_done.txt", "w").write("done")


if os.path.exists("results/bg_done.txt"):
    os.remove("results/bg_done.txt")
threading.Thread(target=_run_jobs, daemon=True).start()
print("뒤에서 학습 시작. 진행 확인은 다음 셀")
''')
code("""
tail("results/log_seq2seq.txt", 4)
print("\\n모두 끝남" if os.path.exists("results/bg_done.txt") else "\\n아직 학습 중")
""")

md("""
## 실습 D ③④. seq2seq 세 구조의 결과
- 학습이 덜 끝났으면 `use_ref()`로 참고 결과를 받아 진행함
""")
code("""
tail("results/log_seq2seq.txt", 6)
# use_ref()
""")
code("""
!python src/viz_attention.py tiny --arch gru_attn --text "삼억 칠천이백만 오천 원"
show("results/fig/attn_tiny_gru_attn.png")
""")
md("""
확인 값
- 17자 이상에서 `gru`는 떨어지고(코랩 CPU 약 0.5), `gru_attn`은 약 1.00
- 그림: 출력 숫자마다 입력의 어느 글자를 봤는지
""")

md("""
## 실습 1. Self-Attention을 직접 계산하고 그려 보기
- 따라 치기 2-T1: 빈칸 5개
""")
typing_cell("typing/s02/t1_self_attention.py")
code("!python typing/s02/t1_self_attention.py")
code("!python src/mini_transformer.py --info")
code("""
!python src/viz_attention.py compare --text "결제했는데 아이템이 안 들어왔어요"
show("results/fig/attn_compare.png")
""")
md("""
확인 값
- 가중치 첫 행 1.00, 둘째 행 0.45·0.55, 대각선 위쪽 0, 행마다 합 1
- 그림: Self는 대각선 위쪽이 빈 삼각형, Cross는 번역 결과와 입력의 대응
- 따옴표 안의 문장을 바꿔 다시 그려 봄
""")

md("""
## 실습 2. 여러 종류의 글로 토큰 수 재기
- 처음 실행할 때 토크나이저 세 개를 내려받음(약 45초)
""")
code("!python src/inspect_tokenizer.py")
code(r'''
texts = ["녹스 궁 쿨 몇초임",
         "for i in range(10): print(i)",
         "2026년 11월 17일 오후 6시, 3,850원",
         "오늘 진짜 최고 ㅋㅋㅋ",
         "Transformer는 RNN보다 빠름"]
for t in texts:
    print("=" * 10, t)
    !python src/inspect_tokenizer.py --text "{t}"
''')
md("""
확인 값
- 한국어 글자당 토큰: A.X 0.46 · Qwen3 0.75 · KURE 0.60(기본 표)
- 같은 뜻이라도 모델마다 토큰 수가 다름. 토큰 수 = 비용 · 속도 · 컨텍스트 한도
""")

md("""
## 실습 3. 다음 토큰 분류기와 고르는 법
- 따라 치기 T3: 빈칸 5개. 처음 실행할 때 모델을 내려받아 1~2분 더 걸림
""")
typing_cell("typing/s01/t3_next_token.py")
code("!python typing/s01/t3_next_token.py")
md("""
- 따라 치기 2-T2: 빈칸 3개. greedy · Temperature · Top-k · Top-p별로 30번 뽑음
""")
typing_cell("typing/s02/t2_sampling.py")
code("!python typing/s02/t2_sampling.py")
md("""
확인 값
- 어휘 크기 151,936, 확률 합 1.0
- 30번 뽑기에서 서로 다른 토큰 수: greedy·T 0.3 1개, T 1.0 22개, T 1.5 29개, Top-k 5 4개, Top-p 0.9 20개
""")

md("""
## 실습 F. LLM으로 영화 리뷰 감성 분류
- 1회차 LSTM과 같은 NSMC 리뷰를 학습 없이 분류함. 프롬프트 뒤에 ' 긍정'과 ' 부정'을 붙여 확률을 비교함
- 리뷰 60건, 코랩 CPU 약 1분(T4면 더 짧음)
""")
code("!python src/llm_sentiment.py")
code("""
!python src/llm_sentiment.py --shots 4
show("results/fig/llm_sentiment.png", width=560)
""")
code("""
!python src/llm_sentiment.py --text "배우는 좋은데 각본이 아쉽다" "두 번 봐도 또 울었다"
# !python src/llm_sentiment.py --shots 4 --model Qwen/Qwen3-0.6B-Base
# !python src/llm_sentiment.py --shots 8
""")
md("""
확인 값
- 예시 0개: 정확도 0.50, 모두 '긍정'으로 답함
- 예시 4개: 약 0.68, Base 모델 예시 4개: 약 0.78(1회차 LSTM 0.80)
""")

md("""
## 해보기 ②. 내 질문 10건으로 FAQ 검색기 재기
- 아래 셀의 질문 10건을 고침(표현 바꾸기 5, 오타 2, 은어 2, 무관 질문 1)
- `gold_faq_id`에 정답 번호를 씀(`data/G/faq.json`). 무관 질문은 `null`
- 10건 중 8건 이상 맞히면 완료
""")
code("%%writefile my_queries.json\n" + (ROOT / "data/G/my_queries_template.json").read_text(encoding="utf-8").rstrip("\n"))
code("""
!python src/faq_search.py --queries my_queries.json
# !python src/faq_search.py --method tfidf --queries my_queries.json
""")

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
| 연결이 끊김 | 다시 연결하고 0번 셀부터 실행. 실습 D 학습은 다시 걸거나 `use_ref()` |
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
