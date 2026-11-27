# GPU 기반 오픈소스 LLM·AI Agent 서비스 개발 PBL: 실습 저장소

> **실습 서버 : 운영 종료(2026-11-27).** 개인 서버와 팀 서버는 모두 지웠음. 학생별 실습 결과는 강사가 보관하며, 자세한 안내는 [ACCESS.md](ACCESS.md)에 있음
> **6 · 7회차 실습은 노트북에서** : 내 서버에서 `notebooks/s06_my_rag.ipynb`(내 문서 챗봇) · `notebooks/s07_my_lora.ipynb`(내 데이터로 학습) · `notebooks/s07_final_studio.ipynb`(7회차 최종 스튜디오 : 부품 코드 조립 · API · 응용)를 열고 셀을 차례로 실행함. 조절판으로 설정을 바꾸고, 끝날 때 `lab.pack()`으로 결과를 묶어 내려받음. 안내 : [6회차](tasks/s06_사건카드.md) · [7회차](tasks/s07_사건카드.md)
> 5회차 내용을 혼자 다시 해 보려면 `notebooks/s05_my_prompt.ipynb`(지시문 · 예시 · 생성 설정 · 형식 · 문제 입력)를 엶. 설명은 handouts 의 `05회차_개인보충_워크북_V6.pdf`
> **8 · 9 · 10회차(차승희 교수) 실습** : `yu-gpu-ai/` 폴더의 노트북 `03` · `04` · `05`를 엶. 차승희 교수 저장소(https://github.com/victoriouscha/yu-gpu-ai)를 고치지 않고 그대로 넣은 폴더임. 노트북을 열 때 잡히는 커널(Python 3)을 그대로 쓰고 `LLM-PBL (course)`로 바꾸지 않음. 9 · 10회차의 웹 화면은 내 실습 접속 주소에서 `-8888`을 `-8000`으로 바꾼 주소로 엶
> **수업 자료(발표 PDF·워크북·안내서): [handouts/](handouts/)**
> Pod 경로: `/workspace/course` · 커널: **LLM-PBL (course)** · 터미널: `source /workspace/activate.sh`
> 모델·버전 기준일 2026-10. 설정은 `src/config.py` 한 곳에서만 바꿈
>
> **코랩·개인 PC로 실습**: [1회차 노트북을 코랩에서 열기](https://colab.research.google.com/github/elliediaz/llm26-course/blob/main/notebooks/s01_colab.ipynb) · [2회차 노트북을 코랩에서 열기](https://colab.research.google.com/github/elliediaz/llm26-course/blob/main/notebooks/s02_colab.ipynb) · [6회차 검색 실험 노트북을 코랩에서 열기](https://colab.research.google.com/github/elliediaz/llm26-course/blob/main/notebooks/s06_colab.ipynb)(런타임 T4 GPU 권장).
> 개인 PC의 주피터에서는 저장소를 받은 뒤 `notebooks/s01_colab.ipynb`·`s02_colab.ipynb`를 그대로 열어 씀

## 구조

```
course/
├─ setup/        Pod 환경 구성(venv·커널·모델 시딩): 수업 전에 미리 실행해 둠
├─ src/          단순 실행 스크립트(아래 표): 완성 코드를 실행하고 결과를 해석
├─ tasks/        해보기·사건 카드: 상황을 주고 값과 설정을 바꿔 판단하게 하는 카드(회차별)
├─ notebooks/    5 · 6 · 7회차 개인 샌드박스 노트북(s05_my_prompt · s06_my_rag · s07_my_lora · s07_final_studio, 원본은 build_sandbox.py), 코랩·개인 PC 공용 실습 노트북(1회차 s01_colab, 2회차 s02_colab, 6회차 검색 실험 s06_colab), 1·2회차 발전사 확인 노트북(원본은 notebooks/src/*.py)
├─ yu-gpu-ai/   8 · 9 · 10회차(차승희 교수) 실습 자료. 원본 저장소를 그대로 둔 폴더라 파일을 고치지 않음
├─ data/
│  ├─ common/    공개 데이터(Fashion-MNIST·NSMC 리뷰), 금액 수사 생성기, 토큰 비교 문장
│  ├─ s02/       2회차 미션 데이터(배달 리뷰·스미싱 문자·뉴스 제목·게임 문의·속담·토큰 표본·지시어 문장)
│  └─ G · L · S  도메인 데이터(README에 출처·라이선스·기준일, 6회차부터 각자 고름). G 게임 운영, L 알바 급여 정산, S 가상 스포츠 기록실
├─ prompts/      2회차 M5 지시문 파일(예시_delivery.txt), 5회차 덧붙일 규칙 파일(s05/)
├─ experiments/  회차별 보조 실험(sNN/): 원리를 눈으로 확인하는 짧은 완성 코드, 옵션으로 조건을 바꿔 봄
├─ results/      모든 산출물(JSON·표·그림): 볼륨에 남음
├─ ref/          회차별 참고 결과(학습이 늦을 때 `bash ref/s01/use_ref.sh`)
├─ index/        6회차 벡터 인덱스
├─ adapters/     7회차 LoRA Adapter
└─ runs/         학습 중간 파일·체크포인트(공유 대상 아님)
```

실습에 쓰는 코드와 기본 데이터는 모두 이 저장소 안에 있음. 인터넷에서 따로 내려받는 것은 Hugging Face 모델 가중치뿐이며, 이것도 서버를 만들 때 미리 받아 둠

## 회차별 스크립트

| 회차 | 스크립트 | 하는 일 |
|---|---|---|
| 공통 | `config.py` `utils.py` `model_utils.py` `run_model.py` `check_gpu.py` `resume_check.py` | 설정, 측정 도구, 모델 로딩·생성, 환경 점검, 복귀·인계 점검 |
| 1 | `nn_tools.py` | 신경망 실습 공통 도구: 층별 모양 추적표, 분류·회귀 공용 학습 루프, 데이터 로더 |
| 1 | `nn_dense.py` | Dense: 유저 이탈(분류)·다음 주 플레이 시간(회귀)을 같은 특징 추출부(백본), 다른 출력층으로 |
| 1 | `nn_cnn.py` | MLP vs CNN: 옷 사진 10종 분류, 첫 층 필터·특징 맵 그림 |
| 1 | `nn_rnn.py` | RNN vs LSTM: 영화 리뷰 감성 분류(길이별 정확도), 시간별 동시 접속자 회귀 |
| 1 | `seq_limits.py` | 첫 숫자 기억하기: 거리가 멀어질 때 RNN·LSTM·어텐션 비교(트랜스포머 이전 구조의 한계) |
| 1 | `nn_vae.py` | VAE: 옷 사진을 숫자 몇 개로 압축했다가 다시 그리기(재구성·사이 잇기·잠재 공간 지도) |
| 1 | `what_head.py` | 수업 모델의 출력층으로 분류·회귀(점수) 판정 |
| 1 | `faq_search.py` | FAQ 의미 검색기(임베딩 vs TF-IDF): 점수 매기기 + 고르기 |
| 1 | `cls_vs_reg.py` | 같은 문제(금액 수사 → 숫자)를 회귀·분류로 풀어 비교 |
| 1 | `ticket_classifier.py` | 고전 분류기·회귀기로 티켓 판정(category·action 분류, priority 분류 vs 회귀) |
| 1 | `tiny_seq2seq_train.py` `viz_attention.py` `seq2seq_demo.py` | 세 구조(GRU·GRU+어텐션·트랜스포머) 학습, Attention 시각화, 구조별 비교표 |
| 1·2 | `lineage.py` | 발전사 노트북용 함수(n-gram, BoW, BPE, 빈칸 vs 다음 토큰, 문맥 내 학습) |
| 2 | `xor_demo.py` | M0: AND·OR·XOR를 단층·층만 쌓기·층 + 활성화로 학습해 결정 경계 비교 |
| 2 | `attention_probe.py` | M2: 다의어 낱말의 층별 벡터 유사도, 지시어 문장의 층별 어텐션, 모델에게 직접 묻기 |
| 2 | `token_cost.py` | M3: 토크나이저 4종의 토큰 수와 글자/토큰, 챗봇 월 요금 추정 |
| 2 | `next_word_game.py` | M4: 다음 토큰 후보와 확률, 속담 이어 말하기 채점(Instruct vs Base), 온도별 뽑기 |
| 2 | `llm_sentiment.py` `llm_classify.py` | M5: 학습 없이 프롬프트로 분류(영화 리뷰, 데이터 5종), 예시·보정·라벨 단어, 리더보드 |
| 2 | `inspect_tokenizer.py` `mini_transformer.py` `gen_params_sweep.py` `kv_cache_bench.py` | 토큰 비교, 미니 트랜스포머, 생성 설정 비교표, KV Cache 측정(부록·5·6회차) |
| 5 | `prompt_utils.py` `scorer.py` `benchmark.py` `scoreboard.py` `dataset_tool.py` `judge.py` | 프롬프트(평가·학습 공용), 채점, 하네스, 성능 기록표, 문항 점검·병합, LLM 채점(도전) |
| 5 | `field_report.py` | 필드별 점수를 분류·추출로 나눠 보고 1회차 분류기와 같은 문항에서 비교 |
| 5~7 | `sweep.py` | 설정 비교 실험: 설정 하나(형식 강제·예시 수·템플릿·온도·최대 길이·규칙 파일·모델·정밀도, 6회차 top-k·문맥 길이, 7회차 Adapter)를 바꿔 같은 문항으로 비교표·그림 |
| 5~7 | `probe.py` | 문제 입력 점검: 오타·은어, 섞인 문의, 프롬프트 주입, 무관 질문, 영어, 긴 입력, 형식 유도, 정보 부족(`data/<도메인>/probes.json`) |
| 6 | `retrieval.py` `index_builder.py` `rag.py` `hybrid_search.py` `rerank.py` | 분할·인덱스·검색, 검색 품질, 검색 기반 생성·오답 분류, 하이브리드·재순위화(도전) |
| 6 | `retrieval_sweep.py` | 검색 설정 비교 실험: LLM 없이 청크 크기·중첩·분할 방식·임베딩·키워드 비중을 바꿔 Recall@k·MRR 비교(코랩 `notebooks/s06_colab.ipynb`) |
| 7 | `build_train_data.py` `train_lora.py` `plot_loss.py` `merge_adapter.py` | 학습 데이터, LoRA/QLoRA 학습, Loss 판독, Adapter 병합(도전) |
| 6~7 | `lab.py` `lab_train.py` | 개인 샌드박스 도구 : 노트북의 조절판(GPU 계기판, 내 재료, 확인 질문, 자르기, 의미 지도, 검색, 설정 실험, 모델 비교, 학습 목표, 학습 데이터, 학습, 전후 비교, 네 구성, 내 솔루션)과 LoRA 학습 실행기. 개인 자료는 `my/` 폴더에 남음 |
| 6~7 | `my_start.py` `pack_results.py` | 도메인 과제를 명령으로 할 때 : 내 도메인·개인 시험지(시드 10 + 문항 풀 20) 준비와 진행 상태 확인, 결과 묶어 내려받기 |

## 실습 코드 세 종류

| 종류 | 위치 | 하는 일 |
|---|---|---|
| 실행·비교 | `src/` | 완성 스크립트를 옵션을 바꿔 가며 실행 → 결과 숫자·그림을 비교해 해석 |
| 보조 실험 | `experiments/sNN/` | 원리를 눈으로 확인하는 짧은 완성 코드(층별 모양, 다음 토큰 확률, Self-Attention, 생성 설정, Chat Template, 채점 규칙). 옵션으로 조건을 바꿔 봄 |
| 해보기·사건 카드 | `tasks/` | 상황(문제)이 주어지고, 설정(하이퍼파라미터)을 바꿔 돌린 결과로 판단해 근거와 함께 기록(5회차까지 팀, 6회차부터 개인) |

## 빠른 시작

```bash
source /workspace/activate.sh
python src/check_gpu.py                 # 1회차: 서버 확인
python src/nn_dense.py                  # 1회차 실습 A
python src/nn_cnn.py                    # 1회차 실습 B
python experiments/s01/next_token.py    # 1회차 실습 E
python src/benchmark.py --limit 3       # 하네스 동작 확인(5회차부터)
python src/sweep.py shots 0 2 --limit 5  # 설정 비교 실험(예시 개수 0 대 2)
```
