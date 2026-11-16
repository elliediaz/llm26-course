"""1회차 실습 노트북(코랩·개인 PC·실습 서버 공용) 생성: notebooks/s01_colab.ipynb

따라 치기 셀은 typing/s01/의 실제 파일 내용을 %%writefile 셀로 넣는다(파일과 노트북이 어긋나지 않게).
고칠 때는 이 파일과 typing/ 원본을 고친 뒤 다시 생성한다: python notebooks/build_colab.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "s01_colab.ipynb"
COLAB_URL = "https://colab.research.google.com/github/elliediaz/llm26-course/blob/main/notebooks/s01_colab.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


def typing_cell(rel: str) -> None:
    body = (ROOT / rel).read_text(encoding="utf-8").rstrip("\n")
    code(f"%%writefile {rel}\n{body}")


md(f"""
# 1회차 실습 노트북: Dense → CNN → RNN·LSTM → seq2seq·어텐션 → 디코더 LLM

코랩, 개인 PC의 주피터, 수업 실습 서버 어디서 열어도 같은 순서로 진행할 수 있게 만든 노트북임.
실습가이드(PDF)의 명령을 셀 단위로 옮겼고, 확인할 값도 같음.

| 할 일 | 방법 |
|---|---|
| 셀 실행 | 셀을 누르고 **Shift + Enter** (또는 셀 왼쪽 ▶) |
| 따라 치기 | `%%writefile`로 시작하는 셀에서 `____`을 채우고 실행 → 파일이 고쳐짐 → 바로 아래 셀로 실행 |
| 그림 보기 | 결과 셀 아래에 그림이 바로 뜸 |
| 내 작업 남기기 | 코랩은 **파일 → Drive에 사본 저장**. 결과 파일은 맨 아래 '결과 내려받기' 셀 |

- 위에서부터 차례로 실행함. 셀 하나가 끝나야(왼쪽 ▶가 다시 보여야) 다음 셀이 돌아감
- 런타임이 끊기면 **0번 셀부터 다시** 실행함(코랩은 연결이 끊기면 만든 파일이 사라짐)
- 코랩 바로 열기: {COLAB_URL}
""")

md("""
## 0. 준비: 저장소 받기와 환경 점검
- 코랩은 런타임 유형 **CPU로 충분함**. T4 GPU가 잡히면 학습이 더 빨라질 뿐 결과는 같음
- 판정이 `통과(...)`로 나오면 됨. `확인 필요`가 나오면 손을 들어 TA를 부름
""")
code(r'''
import os, sys, shutil, subprocess, threading
from IPython.display import Image, Markdown, display

REPO = "https://github.com/elliediaz/llm26-course.git"
if os.path.isfile("../src/config.py"):          # 저장소 안의 notebooks/에서 열었을 때(개인 PC·실습 서버)
    HOME = os.path.abspath("..")
elif os.path.isdir("/workspace/course"):        # 수업 실습 서버(RunPod)
    HOME = "/workspace/course"
else:                                           # 코랩, 또는 저장소 밖에서 연 개인 PC
    HOME = "/content/course" if os.path.isdir("/content") else os.path.abspath("llm26-course")
    if not os.path.isdir(HOME):
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, HOME], check=True)
    else:
        subprocess.run(["git", "-C", HOME, "pull", "-q"])
os.chdir(HOME)
print("작업 폴더:", os.getcwd())
os.makedirs("results/fig", exist_ok=True)


def show(*paths, width=760):
    """그림 파일을 셀 아래에 띄움. 아직 없으면 알려 줌."""
    for p in paths:
        display(Image(p, width=width)) if os.path.exists(p) else print("아직 없음:", p)


def tail(path, n=8):
    """로그 파일의 마지막 n줄(학습이 어디까지 왔는지 확인)."""
    if not os.path.exists(path):
        print("아직 없음:", path)
        return
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    print("\n".join(lines[-n:]))


def use_ref():
    """학습이 늦을 때 1회차 참고 결과를 복사(이미 있는 내 결과는 덮어쓰지 않음)."""
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
%pip install -q koreanize-matplotlib
!python src/check_gpu.py
''')

md("""
## 실습 A. Dense로 이탈(분류)과 플레이 시간(회귀)
- 데이터 `data/G/churn.csv`: 가상 RPG 유저 20,000명의 최근 활동 12개(합성 데이터)
- 같은 몸통에 출력층(머리)만 바꾸면 분류(2칸)와 회귀(1칸)가 됨
- 따라 치기 T1: 아래 셀의 빈칸 6개를 채우고 실행함(막히면 실습가이드 실습 A의 빈칸 답)
""")
typing_cell("typing/s01/t1_dense_heads.py")
code("!python typing/s01/t1_dense_heads.py")
code("""
!python src/nn_dense.py
show("results/fig/nn_dense.png")
""")
md("""
**확인할 것**
- 모양표: `[N,12] → [N,64] → [N,32] →` 분류 `[N,2]` / 회귀 `[N,1]`(마지막 줄만 다름)
- 분류 정확도 약 0.83(기준선 '모두 유지' 0.72), 재현율 약 0.56
- 회귀 평균 오차 약 19분(기준선 '모두 평균' 40분)
- '기준값을 바꾸면' 표: 판정 기준만 바꿔도 재현율과 정밀도가 반대로 움직임
""")

md("""
## 실습 B. CNN으로 옷 사진 10종
- 데이터 `data/common/fashion_mnist/`: Fashion-MNIST(MIT), 28×28 흑백 옷 사진, 학습 6만·시험 1만
- 따라 치기 T2: 층을 지날 때마다 바뀌는 모양을 맞힘(빈칸 6개)
""")
typing_cell("typing/s01/t2_cnn_shapes.py")
code("!python typing/s01/t2_cnn_shapes.py")
code("""
!python src/nn_cnn.py
show("results/fig/nn_cnn_pred.png", "results/fig/nn_cnn_maps.png")
""")
md("""
**확인할 것**
- 모양표: `[N,1,28,28] → [N,16,28,28] → [N,16,14,14] → [N,32,14,14] → [N,32,7,7] → [N,1568] → [N,128] → [N,10]`
- 정확도 MLP 약 0.84, CNN 약 0.88. '가장 많이 헷갈린 쌍'에서 셔츠·티셔츠 찾기
- T2에서 `7`을 `8`로 바꿔 실행하고 오류 문구(`mat1 and mat2 shapes cannot be multiplied`)를 한 번 읽어 봄
""")

md("""
## 해보기 ①
- `tasks/s01_해보기.md`의 A·B에서 골라 **한 번에 하나만** 바꿔 실행하고 결과를 비교함
- 아래 셀의 명령을 고쳐 가며 실행함. 바꾼 것·명령·결과·이유를 기록표에 남김
""")
code("""
!python src/nn_dense.py --hidden 8
# !python src/nn_cnn.py --target group
""")

md("""
## 휴식 전 걸어 두기
- RNN·LSTM 감성 분류와 seq2seq 세 구조 학습을 차례로 **뒤에서** 돌림(실행 즉시 다음 셀로 넘어가도 됨)
- 코랩 CPU에서 약 10분 걸림. 휴식 동안 **브라우저 탭을 닫지 않음**(닫으면 런타임이 끊길 수 있음)
""")
code(r'''
JOBS = [(["src/nn_rnn.py"], "results/log_nn_rnn.txt"),
        (["src/tiny_seq2seq_train.py", "--arch", "all", "--minutes", "2"], "results/log_seq2seq.txt")]


def _run_jobs():
    for args, log in JOBS:
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, *args], stdout=f, stderr=subprocess.STDOUT)
    open("results/bg_done.txt", "w").write("done")


if os.path.exists("results/bg_done.txt"):
    os.remove("results/bg_done.txt")
threading.Thread(target=_run_jobs, daemon=True).start()
print("뒤에서 학습 시작. 진행 확인은 다음 셀")
''')
code("""
tail("results/log_nn_rnn.txt", 4); print("---"); tail("results/log_seq2seq.txt", 4)
print("\\n모두 끝남" if os.path.exists("results/bg_done.txt") else "\\n아직 학습 중")
""")

md("""
## 실습 C. RNN과 LSTM, 내 문장 감성 분류
- 데이터 `data/common/nsmc/`: 네이버 영화 리뷰(NSMC, CC0) 학습 3만·시험 1만 건, 글자 하나를 토큰 하나로 씀
- 휴식 뒤에도 학습이 끝나지 않았으면 `use_ref()`로 참고 결과를 받아 진행함(내 결과는 덮어쓰지 않음)
""")
code("""
tail("results/log_nn_rnn.txt", 30)
# use_ref()      # 학습이 늦을 때만 앞의 # 을 지우고 실행
""")
code("""
!python src/nn_rnn.py --cell lstm --predict "배우들 연기는 좋았는데 결말이 너무 허무하다" "인생 영화 등극 ㅠㅠ 두 번 봤어요"
""")
md("""
**확인할 것**
- 모양표: `[N,T] → 임베딩 [N,T,64] → LSTM [N,T,128] → 마지막 상태 [N,128] → [N,2]`
- 정확도 RNN 약 0.73, LSTM 약 0.80. 81자 이상 리뷰에서 0.66 → 0.76으로 차이가 가장 큼
- 내 문장 3개 이상의 확률. 반어, 부정이 겹친 문장, 신조어로 틀리는 사례 찾기
""")

md("""
## 트랜스포머 이전 구조의 한계(시연)
- 첫 숫자 기억하기: 숫자 L개 중 맨 앞 숫자를 맞히는 10칸 분류(찍으면 0.1)
- VAE: 옷 사진을 숫자 몇 개(z)로 압축했다가 다시 그림
""")
code("!python src/seq_limits.py")
code("""
!python src/nn_vae.py
show("results/fig/nn_vae_recon.png", "results/fig/nn_vae_latent.png")
""")

md("""
## 실습 D. seq2seq와 어텐션
- 한글 금액 수사 → 숫자 표기(예: '삼억 칠천이백만 오천 원' → 372,005,000)
- `gru`는 문장 전체를 벡터 하나에 담고, `gru_attn`은 매 시점 입력 전체를 다시 봄
""")
code("""
tail("results/log_seq2seq.txt", 6)
!python src/viz_attention.py tiny --arch gru_attn --text "삼억 칠천이백만 오천 원"
show("results/fig/attn_tiny_gru_attn.png")
""")
md("""
**확인할 것**
- `gru`는 17자 이상에서 약 0.73으로 떨어지고, `gru_attn`은 길이와 상관없이 약 1.00
- 그림: 출력 숫자마다 입력의 어느 글자를 봤는지
""")

md("""
## 실습 E. 디코더 LLM은 다음 토큰 분류기
- 따라 치기 T3: 빈칸 5개. 처음 실행할 때 모델(Qwen3-0.6B, 약 1.5GB)을 내려받으므로 1~2분 더 걸림
""")
typing_cell("typing/s01/t3_next_token.py")
code("!python typing/s01/t3_next_token.py")
code("""
!python src/seq2seq_demo.py
if os.path.exists("results/structure_compare.md"):
    display(Markdown(open("results/structure_compare.md", encoding="utf-8").read()))
""")
md("""
**확인할 것**
- 어휘 크기 151,936 = 출력층이 고르는 후보 수(클래스 수)
- 상위 5개 후보와 확률, 확률 합 1.0. 마지막 줄 = 분류를 15번 반복해 만든 글
""")

md("""
## 해보기 ② D. 내 질문 10건으로 FAQ 의미 검색기 재기(필수)
- 아래 셀의 질문 10건을 고침: 표현 바꾸기 5, 오타 2, 은어 2, 무관 질문 1
- `gold_faq_id`에 정답 FAQ 번호(`data/G/faq.json`)를 씀. 무관 질문은 `null`
- **10건 중 8건 이상 1순위**이면 완료
""")
code("%%writefile my_queries.json\n" + (ROOT / "data/G/my_queries_template.json").read_text(encoding="utf-8").rstrip("\n"))
code("!python src/faq_search.py --queries my_queries.json")
md("""
- 선택(C): 접속자 예측에서 기준선 이기기. `--epochs`, `--window`, `--hidden`, `--cell` 중 하나씩 바꿈
""")
code("!python src/nn_rnn.py --task ccu")

md("""
## 결과 내려받기
- `results/`를 압축해 내려받음. 코랩은 런타임이 끝나면 파일이 사라지므로 수업을 마치기 전에 실행함
""")
code(r'''
shutil.make_archive("s01_results", "zip", ".", "results")
try:
    from google.colab import files
    files.download("s01_results.zip")
except ImportError:
    print("저장 위치:", os.path.abspath("s01_results.zip"))
''')

md("""
## 문제가 생기면
| 증상 | 조치 |
|---|---|
| `No such file or directory`, `ModuleNotFoundError: config` | 0번 셀을 먼저 실행(작업 폴더가 바뀌지 않은 상태) |
| 런타임 연결이 끊김 | 다시 연결하고 0번 셀부터 실행. 학습이 필요한 실습은 `use_ref()`로 참고 결과를 받음 |
| GPU를 할당할 수 없다는 안내 | CPU로 연결해 그대로 진행(1회차는 CPU로 충분) |
| 셀이 끝나지 않음 | 셀 왼쪽 ■(중지)를 누르고 TA를 부름 |
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "toc_visible": True}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = hashlib.md5(f"s01_colab-{i}".encode()).hexdigest()[:8]
    src = c["source"].split("\n")
    c["source"] = [ln + "\n" for ln in src[:-1]] + [src[-1]]
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("생성:", OUT.name, len(cells), "셀")
