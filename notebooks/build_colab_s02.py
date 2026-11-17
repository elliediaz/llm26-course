"""2회차 실습 노트북 V3(코랩·개인 PC·실습 서버 공용) 생성: notebooks/s02_colab.ipynb

V3: 미션형. M0 XOR 몸풀기 → M1 기억력 감사 → M2 어텐션 탐정 → M3 토큰 요금 견적 → M4 다음 단어 맞히기
    → M5 학습 없는 분류기 경연 → M6 FAQ 검색기(과제)
V1·V2는 notebooks/_V1, notebooks/_V2에 보존함(수강생 저장소에는 올리지 않음)
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
# 2회차 미션 노트북

오늘 실습은 미션 여섯 개로 진행함. 자세한 배경·힌트·기록표는 『2회차 미션 워크북』(PDF)에 있음

- 셀은 위에서부터 차례로 Shift + Enter로 실행함
- `%%writefile` 셀은 빈칸 `____`을 채운 뒤 실행하고, 바로 아래 셀로 결과를 봄
- 연결이 끊기면 0번 셀부터 다시 실행함. 결과 파일은 맨 아래 셀로 내려받음

| 미션 | 의뢰 | 쓰는 것 | 시간 |
|---|---|---|---|
| M0 | 입사 시험: 직선 하나로 안 나뉘는 문제 | `xor_demo.py` | 10분 |
| M1 | 기억력 감사: 긴 문장 앞부분을 잊는 이유 | `seq_limits.py`, seq2seq 학습 | 20분 |
| M2 | 어텐션 탐정: 모델은 문장을 어떻게 읽나 | 따라 치기 2-T1, `attention_probe.py` | 25분 |
| M3 | 토큰 요금 견적: 챗봇 한 달 요금 | `token_cost.py` | 12분 |
| M4 | 다음 단어 맞히기: 속담 이벤트에 써도 되나 | 따라 치기 T3, `next_word_game.py` | 13분 |
| M5 | 학습 없는 분류기 경연 | `llm_classify.py` | 23분 |
| M6 | (과제) FAQ 검색기 재기 | `faq_search.py` | 집에서 |
""")
md("""
## 0. 준비
- 런타임 유형은 T4 GPU 권장(런타임 → 런타임 유형 변경). CPU로도 모든 셀이 돎
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
os.makedirs("prompts", exist_ok=True)


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

# ---------------------------------------------------------------- M0
md("""
## M0. 입사 시험: XOR를 나눠라
- 단층 · 층만 쌓기 · 층 + 활성화 함수, 세 모델로 AND·OR·XOR를 학습함(약 10초)
- 기록: XOR 줄의 정확도 세 개, `--hidden 2`일 때 시드마다 달라지는 값
""")
code("""
!python src/xor_demo.py
show("results/fig/xor_boundary.png", width=620)
""")
code("""
for seed in [0, 1, 3]:
    !python src/xor_demo.py --hidden 2 --seed {seed} | grep XOR
# !python src/xor_demo.py --act sigmoid
""")
md("""
확인 값
- XOR 줄: 단층 0.50, 2층·활성화 X 0.50, 2층·ReLU 1.00
- `--hidden 2`: 시드 0은 0.75, 시드 3은 1.00. 뉴런이 적으면 시작값 운에 따라 갈림
""")

# ---------------------------------------------------------------- M1
md("""
## M1. 기억력 감사: 앞부분을 잊는 이유
의뢰: 고객사 챗봇이 긴 문의의 앞부분을 자꾸 놓친다고 함. RNN·LSTM 계열이 어디까지 기억하는지 재고, 어텐션을 붙이면 달라지는지 확인

1) 기억 길이 시험: 숫자 여러 개 중 맨 앞 숫자를 맞힘(찍으면 0.1, 코랩 CPU 약 1분 30초)
""")
code("""
!python src/seq_limits.py
show("results/fig/seq_limits.png")
""")
md("""
2) 휴식 전 걸어 두기: 한글 금액을 숫자로 바꾸는 seq2seq 세 구조를 2분씩 학습함(약 7분)
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
3) 휴식 후: 세 구조의 결과와 어텐션 그림. 학습이 덜 끝났으면 `use_ref()`로 참고 결과를 받아 진행함
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
- 기억 길이: 거리가 멀어질수록 RNN이 먼저 무너지고 LSTM이 더 버팀
- seq2seq: 17자 이상에서 `gru`는 약 0.5로 떨어지고 `gru_attn`은 약 1.00
- 그림: 출력 숫자마다 입력의 어느 글자를 봤는지가 대각선으로 보임
""")

# ---------------------------------------------------------------- M2
md("""
## M2. 어텐션 탐정: 모델은 문장을 어떻게 읽나
의뢰: "모델이 '그것'이 무엇인지 아는가?"라는 질문에 근거를 들어 답하는 짧은 보고서

1) Self-Attention을 직접 계산함. 따라 치기 2-T1, 빈칸 5개
""")
typing_cell("typing/s02/t1_self_attention.py")
code("!python typing/s02/t1_self_attention.py")
md("""
2) 같은 낱말, 다른 뜻: 과일 배 · 타는 배 · 몸의 배가 모델 안에서 다른 벡터가 되는지 봄(약 20초)
""")
code("""
!python src/attention_probe.py --polysemy
show("results/fig/polysemy.png")
""")
md("""
3) '그것'은 무엇인가: 지시어 문장 12개에서 층별 어텐션을 재고, 모델에게 직접도 물어봄(약 20초)
""")
code("""
!python src/attention_probe.py
show("results/fig/attention_probe.png")
""")
md("""
4) 내 문장으로 수사: 따옴표 안을 바꿔 다시 실행함. 후보는 쉼표로 구분
""")
code("""
!python src/attention_probe.py --text "엄마가 딸에게 전화했다. 그녀는 집에 늦게 온다고 했다." --target 그녀 --cands 엄마,딸
show("results/fig/attention_probe_text.png", width=620)
""")
md("""
확인 값
- 2-T1: 가중치 첫 행 1.00, 둘째 행 0.45·0.55, 대각선 위쪽 0
- 다의어: 0층(임베딩)에서는 '배'끼리 유사도가 모두 1.00. 마지막 층에서는 같은 뜻 0.93, 다른 뜻 0.84로 벌어짐
- 지시어: big·small 쌍에서 '지시어 자리' 값이 똑같음(trophy 쪽 0.41, 뒤 단어를 아직 못 봄). 묻기 정답률 0.50(0.6B 모델의 한계)
""")

# ---------------------------------------------------------------- M3
md("""
## M3. 토큰 요금 견적: 챗봇 한 달 요금
의뢰: 배달앱 고객사가 상담 챗봇의 월 요금 견적을 요청함. 모델(토크나이저)마다 요금이 얼마나 달라지는지 비교

1) 표본 문장 8종 × 토크나이저 4종(처음 실행 때 토크나이저를 내려받아 약 30초)
""")
code("""
!python src/token_cost.py
show("results/fig/token_cost.png")
""")
md("""
2) 쪼개진 모양 보기: 내가 고른 문장을 넣어 봄. `·`은 한 글자가 여러 토큰으로 갈라진 조각
""")
code("""
!python src/token_cost.py --text "점메추 해 줘. 오늘은 갓생 살 거라서" "결제했는데 아이템이 안 들어왔어요"
""")
md("""
3) 견적 조건 바꾸기: 하루 요청 수, 답변 길이, 단가를 우리 서비스에 맞게 바꿔 봄
""")
code("""
!python src/token_cost.py --traffic game --daily 50000 --answer-ratio 5 --price 3000 | tail -8
""")
md("""
확인 값
- 한국어 일상 문장 글자/토큰: A.X 3.06 · Qwen3 1.12 · KURE 1.62 · GPT-2 0.44
- 영어 문장은 네 토크나이저가 모두 같음(19토큰). 차이는 한국어에서 남
- 기본 조건 월 요금: A.X 약 7.7만 원, Qwen3 약 13.9만 원, GPT-2 약 41.9만 원
""")

# ---------------------------------------------------------------- M4
md("""
## M4. 다음 단어 맞히기
의뢰: 마케팅팀이 '속담 이어 말하기' 이벤트에 LLM을 쓰자고 함. 써도 되는지 판정

1) 다음 토큰 분류기 직접 보기: 따라 치기 T3, 빈칸 5개(처음 실행 때 모델을 내려받아 1~2분 더 걸림)
""")
typing_cell("typing/s01/t3_next_token.py")
code("!python typing/s01/t3_next_token.py")
md("""
2) 먼저 맞혀 보기: 아래 앞부분 세 개의 다음 단어를 종이에 적은 뒤 실행함
""")
code("""
!python src/next_word_game.py --prefix "가는 말이 고와야" "The early bird catches the" "오늘 점심 메뉴는"
""")
md("""
3) 속담 50개 채점: 대화용(Instruct)과 원본(Base) 비교(약 2분)
""")
code("""
!python src/next_word_game.py --model both | tail -12
show("results/fig/next_word_game.png", width=560)
""")
md("""
4) 온도 바꿔 보기: 같은 앞부분에서 다섯 번 뽑음
""")
code("""
!python src/next_word_game.py --prefix "우리 팀 프로젝트 이름은" --temp 1.0 --samples 5
# !python src/next_word_game.py --prefix "우리 팀 프로젝트 이름은" --temp 0.3 --samples 5
""")
md("""
확인 값
- T3: 어휘 크기 151,936, 확률 합 1.0
- 속담: 영어는 대부분 맞히고 한국어는 거의 못 맞힘. 학습 데이터에 들어 있던 양의 차이
- 온도 1.0은 매번 다른 이름, 0.3은 거의 같은 이름
""")

# ---------------------------------------------------------------- M5
md("""
## M5. 학습 없는 분류기 경연
의뢰: 고객사 네 곳이 문장 분류기를 요청함. 학습할 시간과 GPU가 없으므로 가중치는 그대로 두고 프롬프트만으로 정확도를 올림

| 과제 | 고객사(가상) | 라벨 | 찍기 수준 |
|---|---|---|---|
| `delivery` | 배달앱 | 맛 · 배달 · 양 · 포장 · 칭찬 | 0.20 |
| `smishing` | 통신사 보안팀 | 정상 · 스미싱 | 0.50 |
| `news` | 포털 뉴스팀 | IT과학 · 경제 · 사회 · 생활문화 · 스포츠 | 0.20 |
| `game` | 게임사 운영팀 | 결제 · 밸런스 · 버그 · 이벤트 · 제재 | 0.20 |
| `nsmc` | (1회차) 영화 리뷰 | 부정 · 긍정 | 0.50 |

규칙
- 개발셋(`--split dev`, 기본값)으로 몇 번이고 고쳐 봄. 시험셋(`--split test`)은 마지막에 한 번만 돌림
- 고칠 수 있는 손잡이는 다섯 개: 지시문(`--prompt`), 예시 수(`--shots`), 라벨 단어(`--words`), 보정(`--calibrate`), 모델(`--model base`)
- 마지막 줄 '제출 줄'을 팀 기록표와 공유 시트에 붙임

1) 몸풀기: 1회차 LSTM(0.80)과 같은 영화 리뷰를 학습 없이 분류(약 1분)
""")
code("!python src/llm_sentiment.py --shots 4")
md("""
2) 우리 팀 과제 고르기
""")
code("""
TASK = "delivery"      # delivery · smishing · news · game · nsmc 중 하나
TAG = "팀이름"          # 리더보드에 남길 이름
!python src/llm_classify.py --list
""")
md("""
3) 기준 점수: 아무것도 고치지 않은 상태
""")
code("""
!python src/llm_classify.py --task {TASK} --show-prompt --tag {TAG}_기준
show(f"results/fig/llm_classify_{TASK}.png", width=420)
""")
md("""
4) 손잡이 돌리기: 한 번에 하나씩 바꾸고 결과를 기록표에 적음
""")
code("""
!python src/llm_classify.py --task {TASK} --shots 5 --tag {TAG}_예시5
""")
code("""
!python src/llm_classify.py --task {TASK} --shots 5 --calibrate --tag {TAG}_예시5_보정
""")
md("""
5) 지시문 고치기: 아래 파일 내용을 우리 과제에 맞게 고쳐 씀. 라벨 정의, 헷갈리는 경우의 기준을 적으면 효과가 큼
""")
code("""
%%writefile prompts/my_prompt.txt
배달 음식 리뷰를 읽고 고객이 무엇에 대해 말하는지 하나만 고름.
맛: 맛, 간, 익힘, 냄새에 대한 불만
배달: 배달 시간, 기사, 도착 과정에 대한 불만
양: 양이 적거나 빠진 품목이 있음
포장: 포장이 새거나 더럽거나 이물질이 나옴
칭찬: 만족하거나 다시 주문하겠다는 내용
""")
code("""
!python src/llm_classify.py --task {TASK} --prompt prompts/my_prompt.txt --shots 5 --calibrate --tag {TAG}_지시문1
""")
md("""
6) 라벨 단어와 모델 바꾸기(선택): 라벨 순서대로 다른 단어를 줌. Base 모델은 대화용으로 다듬기 전의 원본
""")
code("""
# !python src/llm_classify.py --task delivery --shots 5 --calibrate --words "맛없음,늦음,부족,새거나더러움,만족"
# !python src/llm_classify.py --task {TASK} --shots 5 --calibrate --model base --tag {TAG}_base
!python src/llm_classify.py --task {TASK} --shots 5 --calibrate --text "사장님 감사해요 근데 국물이 다 샜어요"
""")
md("""
7) 최종 제출: 개발셋에서 가장 좋았던 설정 그대로 시험셋을 한 번만 돌림
""")
code("""
!python src/llm_classify.py --task {TASK} --split test --prompt prompts/my_prompt.txt --shots 5 --calibrate --tag {TAG}_최종
""")
code("""
import pandas as pd
pd.read_csv("results/leaderboard.csv").tail(12)
""")
md("""
확인 값(Qwen3-0.6B, 시험셋)

| 과제 | 예시 0 | 예시 0 + 보정 | 예시 5 | 예시 5 + 보정 | Base · 예시 5 + 보정 |
|---|---|---|---|---|---|
| delivery | 0.20 | 0.26 | 0.56 | 0.64 | 0.62 |
| news | 0.20 | 0.33 | 0.45 | 0.56 | 0.56 |
| game | 0.24 | 0.39 | 0.43 | 0.61 | 0.71 |
| smishing(예시 2) | 0.53 | 0.58 | 0.98 | 0.98 | 0.78 |

- 위 5)의 예시 지시문 + 예시 5 + 보정(delivery): 개발셋 0.80, 시험셋 0.70
- 예시 0개일 때 한 라벨로만 답하는 쏠림이 보이면 보정부터 켬
- 이 값을 넘기면 지시문이 효과를 낸 것
""")

# ---------------------------------------------------------------- M6
md("""
## M6. (과제) 내 질문 10건으로 FAQ 검색기 재기
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
## 결과 확인과 내려받기
- 오늘 남겨야 할 파일(M0·M1·M2·M3·M5)이 모두 있는지 확인함. '누락'이 있으면 해당 미션 셀을 다시 실행함
- 코랩은 연결이 끝나면 파일이 사라지므로 수업을 마치기 전에 내려받음
""")
code("!python src/resume_check.py --session 2 --quick")
code(r'''
shutil.make_archive("s02_results", "zip", ".", "results")
try:
    from google.colab import files
    files.download("s02_results.zip")
except ImportError:
    print("저장 위치:", os.path.abspath("s02_results.zip"))
''')

md("""
## 더 해 보기(선택)
- 따라 치기 2-T2: greedy · Temperature · Top-k · Top-p별로 30번 뽑기
- 나만의 데이터: `data/s02/README.md` 구조로 라벨당 15건을 써서 `--data`로 돌려 봄
""")
typing_cell("typing/s02/t2_sampling.py")
code("!python typing/s02/t2_sampling.py")

md("""
## 문제가 생기면
| 증상 | 조치 |
|---|---|
| `No such file or directory` | 0번 셀부터 다시 실행 |
| 연결이 끊김 | 다시 연결하고 0번 셀부터 실행. M1 학습은 다시 걸거나 `use_ref()` |
| GPU를 쓸 수 없다는 안내 | CPU로 그대로 진행 |
| 셀이 끝나지 않음 | 셀 왼쪽 ■를 누르고 TA를 부름 |
| `Rate limited` 문구 | 1~2분 기다리면 이어짐 |
| `--words 는 라벨 수만큼` | 라벨 순서대로 같은 개수를 쉼표로 적음 |
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
