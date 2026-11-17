"""6회차 검색 실험 노트북(코랩·개인 PC 공용) 생성: notebooks/s06_colab.ipynb

LLM 없이 검색만 재는 사건 2·3(조각 크기·겹침, 임베딩·키워드·섞기)과 사건 5의 검색 부분을 개인 환경에서 돌림.
LLM이 들어가는 측정(sweep.py mode·top_k, benchmark.py, probe.py)은 팀 GPU 서버에서 함.
다시 생성: python notebooks/build_colab_s06.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "s06_colab.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# 6회차 검색 실험 노트북

사건 2·3은 검색만 재므로 LLM이 필요 없음. 팀 서버는 실행 담당이 LLM 측정에 쓰고, 팀원은 이 노트북으로 각자 검색 손잡이를 돌림

- 자세한 상황·실측 참고값·판단 기록표는 『6회차 사건 워크북』(PDF)에 있음
- 런타임은 **T4 GPU를 권장**함(손잡이 하나에 1분 안쪽). CPU면 손잡이 하나에 3~6분 걸림
- 팀원끼리 손잡이를 나눠 맡으면 빠름(예: 한 명은 조각 크기, 한 명은 겹침, 한 명은 임베딩)

| 사건 | 손잡이 | 명령 |
|---|---|---|
| 2 | 조각 크기 · 겹침 · 분할 방식 | `retrieval_sweep.py size` · `overlap` · `strategy` |
| 3 | 임베딩 · 키워드 · 섞는 비중 | `retrieval_sweep.py embed` · `hybrid`, `hybrid_search.py` |
| 5 | 새 문서를 넣었을 때 검색 결과 | `index_builder.py --with-updates`, `rag.py --search-only` |
""")
md("""
## 0. 준비
- 아래 셀의 `DOMAIN`을 팀 도메인(G·L·S)으로 바꾼 뒤 실행함
- 처음 실행 때 임베딩 모델(KURE-v1 약 2.3GB)을 받음. bge-m3는 사건 3에서 한 번 더 받음
""")
code(r'''
import os, shutil, subprocess
from IPython.display import Image, display

DOMAIN = "G"                                    # 팀 도메인 G · L · S
REPO = "https://github.com/elliediaz/llm26-course.git"
if os.path.isfile("../src/config.py"):          # 저장소 안에서 열었을 때
    HOME = os.path.abspath("..")
else:                                           # 코랩
    HOME = "/content/course" if os.path.isdir("/content") else os.path.abspath("llm26-course")
    if not os.path.isdir(HOME):
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, HOME], check=True)
    else:
        subprocess.run(["git", "-C", HOME, "pull", "-q"])
os.chdir(HOME)
os.environ["COURSE_DOMAIN"] = DOMAIN
os.makedirs("results/fig", exist_ok=True)
print("작업 폴더:", os.getcwd(), "· 도메인:", DOMAIN)


def show(*paths, width=760):
    for p in paths:
        display(Image(p, width=width)) if os.path.exists(p) else print("아직 없음:", p)
''')
code(r'''
%pip install -q koreanize-matplotlib sentence-transformers rank-bm25
!python src/index_builder.py --stats
''')
md("""
확인
- 두 분할 전략(fixed · sentence)의 조각 수와 길이가 나옴. 문서가 짧은 도메인은 조각이 문서 수와 비슷함
- 팀 문항 20건까지 넣어 재려면 팀 서버의 `data/<도메인>/test_dataset.json`을 왼쪽 파일 창으로 올려 같은 자리에 덮어씀. 그대로 두면 시드 10건으로 잼
""")

md("""
## 사건 2. 검색이 엉뚱한 문서를 가져옴
- 고정 길이 조각의 크기와 겹침을 바꿔 Recall@1·3·5, MRR, 조각 수를 비교함
- 표 오른쪽의 `R@3 K` · `R@3 R` · `R@3 F`는 유형별 Recall@3임. 평균이 같아도 유형마다 다르게 움직일 수 있음
""")
code("""
!python src/retrieval_sweep.py size 200 500 1000 2000
show("results/fig/retrieval_size.png")
""")
code("""
!python src/retrieval_sweep.py overlap 0 100 250 --size 500
show("results/fig/retrieval_overlap.png")
""")
code("""
!python src/retrieval_sweep.py strategy fixed sentence
# !python src/retrieval_sweep.py size 300 800 --tag more          # 다른 값을 더 볼 때(결과 파일 이름을 나눔)
""")
md("""
판단
- 고른 분할 방식과 크기·겹침, 근거 숫자 두 개(Recall@3, MRR, 조각 수 중)를 워크북 기록표에 적음
- 고른 값은 팀 서버의 `src/config.py` `CHUNK`·`CHUNK_STRATEGY`에 적고 실행 담당이 `index_builder.py`로 인덱스를 다시 만듦
""")

md("""
## 사건 3. 이름과 숫자가 든 질문에서 놓침
- 임베딩 두 개(KURE-v1, bge-m3)와 키워드 검색(BM25)을 비교하고, 벡터 점수와 키워드 점수를 섞는 비중을 바꿈
- `hybrid` 값 0은 벡터만, 1은 키워드만임
""")
code("""
!python src/retrieval_sweep.py embed nlpai-lab/KURE-v1 BAAI/bge-m3 keyword
show("results/fig/retrieval_embed.png")
""")
code("""
!python src/retrieval_sweep.py hybrid 0 0.3 0.5 0.7 1
show("results/fig/retrieval_hybrid.png")
""")
md("""
- 아래 셀은 점수를 섞지 않고 순위를 합치는 방식(RRF)임. 인덱스 파일이 있어야 하므로 먼저 인덱스를 만듦(GPU 수십 초, CPU 1~2분)
""")
code("""
!python src/index_builder.py --strategy sentence --backend numpy
!python src/hybrid_search.py --strategy sentence
""")
md("""
판단
- 우리 도메인에 맞는 검색 방식(벡터 · 키워드 · 섞기)과 근거. 유형별 Recall@3에서 어느 유형이 이득을 보고 어느 유형이 손해를 보는지 함께 적음
""")

md("""
## 사건 5(검색 부분). 새 문서를 넣으면 검색 결과가 바뀌나
- 새 문서는 `data/<도메인>/corpus_updates/`에 있음. 같은 질문의 검색 결과를 새 문서를 넣기 전후로 비교함
- 답이 실제로 바뀌는지는 팀 서버에서 `rag.py`(LLM)로 봄. 여기서는 근거 문서가 바뀌는지만 봄
""")
code(r'''
Q = {"G": "겨울 성좌 축제에서 성좌의 등불 받으려면 별빛 조각 몇 개 모아야 해요?",
     "L": "한빛로 달빛편의점에서 2027년 3월 2일부터 4주 동안 주 3일, 20시부터 다음 날 2시까지 6시간씩 일했어요. "
          "시급 10,700원이고 주휴수당은 받았는데 야간수당은 따로 없었어요. 덜 받은 돈이 있나요?",
     "S": "2026 월드 인비테이셔널 우승팀이 어디예요?"}[DOMAIN]
!cat data/$COURSE_DOMAIN/corpus_updates/updates_manifest.json
print("\n[새 문서 넣기 전]")
!python src/rag.py --q "{Q}" --strategy sentence --search-only
''')
code(r'''
!python src/index_builder.py --strategy sentence --backend numpy --with-updates
print("\n[새 문서 넣은 뒤]")
!python src/rag.py --q "{Q}" --strategy sentence --search-only
!python src/index_builder.py --strategy sentence --backend numpy        # 원래 인덱스로 되돌림
''')

md("""
## 결과 내려받기
- 코랩은 연결이 끝나면 파일이 사라지므로 수업을 마치기 전에 내려받아 팀 서버의 `results/`에 올림
""")
code(r'''
shutil.make_archive(f"s06_retrieval_{DOMAIN}", "zip", ".", "results")
try:
    from google.colab import files
    files.download(f"s06_retrieval_{DOMAIN}.zip")
except ImportError:
    print("저장 위치:", os.path.abspath(f"s06_retrieval_{DOMAIN}.zip"))
''')

md("""
## 문제가 생기면
| 증상 | 조치 |
|---|---|
| `No module named 'rank_bm25'` | 준비의 두 번째 셀을 다시 실행 |
| 임베딩 받기가 오래 걸림 | 처음 한 번만 그럼. 기다리는 동안 `data/<도메인>/corpus/` 문서를 읽음 |
| `인덱스가 없습니다` | `hybrid_search.py` 앞 셀의 `index_builder.py`를 먼저 실행 |
| 도메인을 바꿨는데 결과가 그대로 | 0번 셀의 `DOMAIN`을 고치고 0번 셀부터 다시 실행 |
| 연결이 끊김 | 다시 연결하고 0번 셀부터 실행. 받아 둔 모델은 다시 받음 |
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "toc_visible": True}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = hashlib.md5(f"s06_colab-{i}".encode()).hexdigest()[:8]
    src = c["source"].split("\n")
    c["source"] = [ln + "\n" for ln in src[:-1]] + [src[-1]]
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("생성:", OUT.name, len(cells), "셀")
