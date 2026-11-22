# G 도메인 데이터: 게임 운영 AI(프로젝트 루멘)

| 항목 | 내용 |
|---|---|
| 출처 | 자체 제작 가상 게임 DB(`build/game_db.json`). 실존 게임과 무관함 |
| 라이선스 | CC BY 4.0(자체 저작). 학생 저장소·포트폴리오에 데이터 포함 가능 |
| 기준일 | 2026-10-29 |
| 생성 | `python data/G/build/make_game_pkg.py` (시드 고정, 다시 실행해도 같은 결과) |
| 개인정보 | 없음. 닉네임은 가상 |

## 파일

| 파일 | 회차 | 내용 |
|---|---|---|
| `churn.csv` | 1 | 유저 20,000명의 최근 활동 12개 → 다음 14일 이탈(`churn_14d`, 분류) · 다음 7일 일평균 플레이 분(`next7_play_min`, 회귀). 생성 `build/make_churn_ccu.py` |
| `ccu_hourly.csv` | 1 | 2026년 시간별 동시 접속자 8,760행(하루·주 주기, 수요일 06~08시 점검 0명, 업데이트 직후 급증). 생성 같은 스크립트 |
| `faq.json`, `faq_queries.json` | 1 | FAQ 25건, 시험 질의 10건(표현 변형·오타·은어·무관 질문) |
| `gen_prompts.json` | 2 | 생성 설정 비교용 프롬프트 6건(사실형·창작형·형식형 각 2) |
| `schema.json`, `scoring.yaml` | 5~7 | 출력 JSON 형식, 필드별 채점 규칙 |
| `test_seed.json` | 5 | 시드 10건(K4·R3·F3) |
| `my_item_template.json` | 5 | 직접 추가하는 문항 20건 양식(K8·R7·F5)과 작성 규칙 |
| `test_dataset.json` | 5~7 | `benchmark.py` 입력. 시드 10건에 문항 풀(`item_pool.json`)에서 뽑은 20건을 합쳐 30건(`my_start.py`) |
| `corpus/*.md`, `corpus_manifest.json` | 6 | 위키형 문서 63건 |
| `train_seed.jsonl` | 7 | 학습 시드 185건(`input`·`gold` 쌍. 메시지 형식은 `build_train_data.py`가 평가와 같은 프롬프트로 만듦) |
| `tools_spec.md` | 8~10 | Tool 후보 |

## 평가셋과 학습셋 분리
- 패치 변경 항목·상품·영웅·장비·닉네임을 평가용과 학습용으로 나누고, 문장 틀도 따로 쓴다.
- `build_train_data.py`가 입력 문장 유사도(문자 3-gram 자카드)로 한 번 더 검사한다.


## churn.csv 열

| 열 | 뜻 | 열 | 뜻 |
|---|---|---|---|
| days_since_join | 가입 경과일 | guild | 길드 가입(1)·미가입(0) |
| login_days_14 | 최근 14일 중 접속한 날 | friends | 친구 수 |
| avg_play_min | 최근 일평균 플레이 분 | tickets_30 | 최근 30일 고객 문의 수 |
| level | 캐릭터 레벨 | pvp_win | PvP 승률 |
| level_up_14 | 최근 14일 레벨업 횟수 | last_login_gap | 마지막 접속 후 경과일 |
| pay_30 | 최근 30일 결제액(원) | event_join_30 | 최근 30일 이벤트 참여 수 |
| **churn_14d** | 정답(분류): 다음 14일 동안 접속하지 않음 1 | **next7_play_min** | 정답(회귀): 다음 7일 일평균 플레이 분 |

- 실제 게임 운영 지표의 모양을 흉내 낸 합성 데이터임: 접속이 뜸하고 문의가 잦고 길드가 없으면 이탈이 늘고, 결제는 일부 유저에 몰림. 이탈 비율 약 28%
