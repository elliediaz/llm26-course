# GPU 기반 오픈소스 LLM·AI Agent 서비스 개발 PBL: 실습 저장소

> Pod 경로: `/workspace/course` · 커널: **LLM-PBL (course)** · 터미널: `source /workspace/activate.sh`
> 모델·버전 기준일 2026-10. 설정은 `src/config.py` 한 곳에서만 바꿈

## 구조

```
course/
├─ setup/        Pod 환경 구성(venv·커널·모델 시딩): 수업 전 강사가 실행
├─ src/          단순 실행 스크립트(아래 표): 완성 코드를 실행하고 결과를 해석
├─ typing/       따라 치기 코드: 빈칸(____)을 채워 직접 실행(회차별 sNN/)
├─ tasks/        과제 카드: 팀·개인에게 던지는 주제(회차별)
├─ notebooks/    1·2회차 발전사 확인 노트북(.ipynb), 원본은 notebooks/src/*.py
├─ data/
│  ├─ common/    수사 변환 학습 데이터 생성기, 토큰 비교 문장
│  └─ G · L · S  팀 도메인 데이터(README에 출처·라이선스·기준일)
├─ results/      모든 산출물(JSON·표·그림): 볼륨에 남음
├─ ref/          회차별 참고 결과(학습이 늦을 때 `bash ref/s01/use_ref.sh`)
├─ index/        6회차 벡터 인덱스
├─ adapters/     7회차 LoRA Adapter
└─ runs/         학습 중간 파일(공유 대상 아님)
```

## 회차별 스크립트

| 회차 | 스크립트 | 하는 일 |
|---|---|---|
| 공통 | `config.py` `utils.py` `model_utils.py` `run_model.py` `check_gpu.py` `resume_check.py` | 설정, 측정 도구, 모델 로딩·생성, 환경 점검, 복귀·인계 점검 |
| 1 | `what_head.py` | 수업 모델의 출력층으로 분류·회귀(점수) 판정 |
| 1 | `faq_search.py` | FAQ 의미 검색기(임베딩 vs TF-IDF): 점수 매기기 + 고르기 |
| 1 | `cls_vs_reg.py` | 같은 문제(금액 수사 → 숫자)를 회귀·분류로 풀어 비교 |
| 1 | `ticket_classifier.py` | 고전 분류기·회귀기로 티켓 판정(category·action 분류, priority 분류 vs 회귀) |
| 1 | `tiny_seq2seq_train.py` `viz_attention.py` `seq2seq_demo.py` | 세 구조(GRU·GRU+어텐션·트랜스포머) 학습, Attention 시각화, 구조별 비교표 |
| 1·2 | `lineage.py` | 발전사 노트북용 함수(n-gram, BoW, BPE, 빈칸 vs 다음 토큰, 문맥 내 학습 등) |
| 2 | `inspect_tokenizer.py` `mini_transformer.py` `gen_params_sweep.py` `kv_cache_bench.py` | 토큰 비교, 미니 트랜스포머, 생성 설정 비교표, KV Cache 측정 |
| 5 | `prompt_utils.py` `scorer.py` `benchmark.py` `scoreboard.py` `dataset_tool.py` `judge.py` | 프롬프트(평가·학습 공용), 채점, 하네스, 성능 기록표, 문항 점검·병합, LLM 채점(도전) |
| 5 | `field_report.py` | 필드별 점수를 분류·추출로 나눠 보고 1회차 분류기와 같은 문항에서 비교 |
| 6 | `retrieval.py` `index_builder.py` `rag.py` `hybrid_search.py` `rerank.py` | 분할·인덱스·검색, 검색 품질, 검색 기반 생성·오답 분류, 하이브리드·재순위화(도전) |
| 7 | `build_train_data.py` `train_lora.py` `plot_loss.py` `merge_adapter.py` | 학습 데이터, LoRA/QLoRA 학습, Loss 판독, Adapter 병합(도전) |

## 실습 코드 세 종류

| 종류 | 위치 | 하는 일 |
|---|---|---|
| 따라 치기 | `typing/sNN/` | 강사 화면을 보며 빈칸을 채워 실행(10~40줄). 원리를 손으로 확인 |
| 단순 실행 | `src/` | 완성 스크립트 실행 → 결과 숫자·그림 해석 |
| 과제(주제 던지기) | `tasks/` | 팀·개인 주제. 결과와 근거를 기록 |

## 빠른 시작

```bash
source /workspace/activate.sh
python src/check_gpu.py                 # 1회차 ①
python typing/s01/t1_next_token.py      # 1회차 ②(빈칸을 채운 뒤)
python src/faq_search.py                # 1회차 ③
python src/benchmark.py --limit 3       # 하네스 동작 확인(5회차부터)
```

## 규칙

- 모델 이름·생성 설정·경로를 스크립트에 직접 쓰지 않음 → `config.py`
- 평가는 `GEN_EVAL`(greedy) 고정. 바꾸면 이전 측정과 비교할 수 없음
- 한 번에 하나만 바꿔서 잼(프롬프트·채점 규칙·데이터를 동시에 바꾸지 않음)
- 실존 개인정보 입력 금지. 데이터 폴더의 README에 적힌 라이선스를 따름
- 결과 파일은 `results/`에 남기고 팀 저장소에 커밋(데이터는 트랙 A만)
