# L 도메인 데이터: 알바 급여 정산(회귀형)

| 항목 | 내용 |
|---|---|
| 과제 | 가상 상권 '한빛로'의 아르바이트 근무 상황 → 급여 정산 JSON(쟁점·주 소정근로시간·최저시급·주휴수당·미지급액·근거 조문) |
| 타입 | **회귀형**: 채점 가중치 10 중 6이 수치 필드(허용 오차 안에서 맞으면 정답) |
| 출처 | 강사 제작 가상 사업장 24곳(`build/wage_db.json`), 최저임금 고시 요약, 강사 작성 해설. 실존 상호·인물과 무관함 |
| 라이선스 | CC BY 4.0(강사 저작). 최저임금 문서는 공공정보 요약 |
| 기준일 | 2026-11-01(최저시급 2026년 10,320원, 2027년 10,700원) |
| 정답 계산 | 모든 정답은 규칙엔진 `build/wage_rules.py`가 계산함. 법률 자문이 아님 |
| 생성 | `python data/L/build/make_wage_pkg.py` (시드 고정, 다시 실행해도 같은 결과) |

## 파일

| 파일 | 회차 | 내용 |
|---|---|---|
| `schema.json`, `scoring.yaml` | 5~7 | 출력 JSON 형식, 필드별 채점 규칙(분류 2 · 회귀 4 · 추출 1) |
| `test_seed.json` | 5 | 강사 시드 10건(K4·R3·F3) |
| `team_item_template.json` | 5 | 팀 추가 20건 양식(K8·R7·F5)과 작성 규칙 |
| `test_dataset.json` | 5~7 | `benchmark.py` 입력. 처음에는 시드 10건, 5회차에 팀 20건을 합쳐 30건 |
| `corpus/*.md`, `corpus_manifest.json` | 6 | 사업장 문서 24, 최저임금 2, 해설 6(조문 원문은 6회차 전에 추가) |
| `train_seed.jsonl` | 7 | 학습 시드 180건(학습용 사업장 ws15~ws24, 평가와 다른 문장 틀) |
| `tools_spec.md` | 8~10 | Tool 후보(급여 계산기·사업장 조회·최저시급 조회) |

## 팀 문항의 정답 만들기

```bash
python data/L/build/wage_rules.py --year 2027 --weeks 4 --days 3 --hours 6 --wage 10500 --size 3
# {"issue": "최저임금미달", ..., "unpaid_krw": 168480, "law_article": "최저임금법 제6조"}
```

- 야간 근무는 `--night <하루 중 22시~06시 시간>`, 주휴수당을 받았으면 `--holiday-paid yes`, 가산수당을 받았으면 `--premium-paid yes`
- 사업장 문항(K)은 `corpus/ws01.md`~`ws14.md`의 시급·직원 수를 넣어 계산함

## 평가셋과 학습셋 분리
- 사업장을 평가용(ws01~ws14)과 학습용(ws15~ws24)으로 나누고, 문장 틀도 따로 씀
- `build_train_data.py`가 입력 문장 유사도(문자 3-gram 자카드)로 한 번 더 검사함
