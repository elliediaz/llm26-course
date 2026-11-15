# G 도메인 데이터: 게임 운영 AI(프로젝트 루멘)

| 항목 | 내용 |
|---|---|
| 출처 | 강사 자체 제작 가상 게임 DB(`build/game_db.json`). 실존 게임과 무관함 |
| 라이선스 | CC BY 4.0(강사 저작). 학생 저장소·포트폴리오에 데이터 포함 가능 |
| 기준일 | 2026-10-29 |
| 생성 | `python data/G/build/make_game_pkg.py` (시드 고정, 다시 실행해도 같은 결과) |
| 개인정보 | 없음. 닉네임은 가상 |

## 파일

| 파일 | 회차 | 내용 |
|---|---|---|
| `faq.json`, `faq_queries.json` | 1 | FAQ 25건, 시험 질의 10건(표현 변형·오타·은어·무관 질문) |
| `gen_prompts.json` | 2 | 생성 설정 비교용 프롬프트 6건(사실형·창작형·형식형 각 2) |
| `schema.json`, `scoring.yaml` | 5~7 | 출력 JSON 형식, 필드별 채점 규칙 |
| `test_seed.json` | 5 | 강사 시드 10건(K4·R3·F3) |
| `team_item_template.json` | 5 | 팀 추가 20건 양식(K8·R7·F5)과 작성 규칙 |
| `test_dataset.json` | 5~7 | `benchmark.py` 입력. 처음에는 시드 10건, 5회차에 팀 20건을 합쳐 30건 |
| `corpus/*.md`, `corpus_manifest.json` | 6 | 위키형 문서 63건 |
| `train_seed.jsonl` | 7 | 학습 시드 185건(`input`·`gold` 쌍. 메시지 형식은 `build_train_data.py`가 평가와 같은 프롬프트로 만듦) |
| `tools_spec.md` | 8~10 | Tool 후보 |

## 평가셋과 학습셋 분리
- 패치 변경 항목·상품·영웅·장비·닉네임을 평가용과 학습용으로 나누고, 문장 틀도 따로 쓴다.
- `build_train_data.py`가 입력 문장 유사도(문자 3-gram 자카드)로 한 번 더 검사한다.

