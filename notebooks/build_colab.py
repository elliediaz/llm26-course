"""1회차 실습 노트북(코랩·개인 PC·실습 서버 공용) 생성: notebooks/s01_colab.ipynb

보조 실험 셀은 experiments/s01/의 완성 코드를 옵션만 바꿔 실행한다(빈칸 없음).
고칠 때는 이 파일을 고친 뒤 다시 생성한다: python notebooks/build_colab.py
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


md("""
# 1회차 실습 노트북

실습가이드의 명령을 순서대로 셀로 옮긴 노트북임. 코랩, 개인 PC, 실습 서버에서 똑같이 쓸 수 있음

- 셀은 위에서부터 차례로 Shift + Enter로 실행함
- `!python experiments/...` 셀은 옵션(`--act`, `--kernel` 등)을 바꿔 가며 여러 번 돌려 봄. 결과가 어떻게 달라지는지가 핵심임
- 연결이 끊기면 0번 셀부터 다시 실행함
- 작업을 남기려면 파일 → Drive에 사본 저장. 결과 파일은 맨 아래 셀로 내려받음
""")

md("""
## 0. 준비
- 런타임 유형은 CPU면 됨
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
## 실습 A. Dense로 이탈(분류)과 플레이 시간(회귀)
- 데이터: 가상 RPG 유저 2만 명의 활동 12개(`data/G/churn.csv`)
- 보조 실험: 같은 몸통에 출력층만 2칸(분류)·1칸(회귀). 활성화를 빼거나 몸통을 좁히면 두 점수가 어떻게 되는지 봄
""")
code("!python experiments/s01/dense_heads.py")
code("""
!python experiments/s01/dense_heads.py --act none         # 활성화 없이: 층을 쌓아도 직선 하나
!python experiments/s01/dense_heads.py --hidden 4         # 몸통을 아주 좁게
""")
code("""
!python src/nn_dense.py
show("results/fig/nn_dense.png")
""")
md("""
확인 값
- 모양표 마지막 줄: 분류 `[N,2]`, 회귀 `[N,1]`
- 분류 정확도 약 0.83(기준선 0.72), 회귀 평균 오차 약 19분(기준선 40분)
""")

md("""
## 실습 B. CNN으로 옷 사진 10종
- 데이터: Fashion-MNIST 옷 사진 7만 장(28×28 흑백)
- 보조 실험: 층마다 바뀌는 모양과 파라미터 수. 필터 크기·패딩·풀링을 바꿔 Linear 입력 크기가 어떻게 바뀌는지 봄
""")
code("!python experiments/s01/cnn_shapes.py")
code("""
!python experiments/s01/cnn_shapes.py --kernel 5 --padding 0
!python experiments/s01/cnn_shapes.py --no-pool --compare-mlp
""")
code("""
!python src/nn_cnn.py
show("results/fig/nn_cnn_pred.png", "results/fig/nn_cnn_maps.png")
""")
md("""
확인 값
- 정확도 MLP 약 0.84, CNN 약 0.88
- 가장 많이 헷갈린 쌍: 셔츠와 티셔츠
""")

md("""
## 해보기 ①
- `tasks/s01_해보기.md`에서 골라 값을 하나만 바꿔 실행하고 기록표에 남김
""")
code("""
!python src/nn_dense.py --hidden 8
# !python src/nn_cnn.py --target group
""")

md("""
## 휴식 전 걸어 두기
- 실습 C·D에 쓸 학습을 뒤에서 돌림(약 7분)
- 쉬는 동안 브라우저 탭을 닫지 않음. 도는 동안 다른 학습 셀은 실행하지 않음
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
- 데이터: 네이버 영화 리뷰 4만 건(`data/common/nsmc/`)
- 학습이 덜 끝났으면 `use_ref()`로 참고 결과를 받아 진행함
""")
code("""
tail("results/log_nn_rnn.txt", 30)
# use_ref()
""")
code("""
!python src/nn_rnn.py --cell lstm --predict "배우들 연기는 좋았는데 결말이 너무 허무하다" "인생 영화 등극 ㅠㅠ 두 번 봤어요"
""")
md("""
확인 값
- 정확도 RNN 약 0.73, LSTM 약 0.80(81자 이상 리뷰에서 차이가 가장 큼)
- 따옴표 안의 문장을 내 문장으로 바꿔 틀리는 예를 찾아봄
""")

md("""
## 트랜스포머 이전 구조의 한계(시연)
- 첫 숫자 기억하기(찍으면 0.1), VAE(사진을 숫자 몇 개로 줄였다가 다시 그림)
""")
code("!python src/seq_limits.py      # 코랩 CPU에서 약 1분 30초")
code("""
!python src/nn_vae.py
show("results/fig/nn_vae_recon.png", "results/fig/nn_vae_latent.png")
""")

md("""
## 실습 D. seq2seq와 어텐션
- 한글 금액을 숫자로 바꾸기(예: '삼억 칠천이백만 오천 원' → 372,005,000)
""")
code("""
tail("results/log_seq2seq.txt", 6)
!python src/viz_attention.py tiny --arch gru_attn --text "삼억 칠천이백만 오천 원"
show("results/fig/attn_tiny_gru_attn.png")
""")
md("""
확인 값
- 17자 이상에서 `gru`는 떨어지고(코랩 약 0.56), `gru_attn`은 약 1.00
""")

md("""
## 실습 E. 디코더 LLM은 다음 토큰 분류기
- 보조 실험: 출력층 크기(어휘 수)와 한 토큰씩 고르는 과정. 문장을 바꿔 1등 확률이 높은 단계와 갈리는 단계를 봄
- 처음 실행할 때 모델을 내려받아 1~2분 더 걸림
""")
code("!python experiments/s01/next_token.py")
code('!python experiments/s01/next_token.py --text "환불은 구매 후 며칠" --steps 10 --topk 3')
code("""
!python src/seq2seq_demo.py
if os.path.exists("results/structure_compare.md"):
    display(Markdown(open("results/structure_compare.md", encoding="utf-8").read()))
""")
md("""
확인 값
- 어휘 크기 151,936, 확률 합 1.0
""")

md("""
## 해보기 ② D. 내 질문 10건으로 FAQ 검색기 재기
- 아래 셀의 질문 10건을 고침(표현 바꾸기 5, 오타 2, 은어 2, 무관 질문 1)
- `gold_faq_id`에 정답 번호를 씀(`data/G/faq.json`). 무관 질문은 `null`
- 10건 중 8건 이상 맞히면 완료
""")
code("%%writefile my_queries.json\n" + (ROOT / "data/G/my_queries_template.json").read_text(encoding="utf-8").rstrip("\n"))
code("!python src/faq_search.py --queries my_queries.json")
md("""
- 선택: 접속자 예측에서 기준선 이기기
""")
code("!python src/nn_rnn.py --task ccu")

md("""
## 결과 내려받기
- 코랩은 연결이 끝나면 파일이 사라지므로 수업을 마치기 전에 실행함
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
| `No such file or directory` | 0번 셀부터 다시 실행 |
| 연결이 끊김 | 다시 연결하고 0번 셀부터 실행, 필요하면 `use_ref()` |
| 셀이 끝나지 않음 | 셀 왼쪽 ■를 누르고 TA를 부름 |
| `Rate limited` 문구 | 1~2분 기다리면 이어짐 |
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
