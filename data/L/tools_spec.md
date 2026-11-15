# L 도메인 Tool 후보(8~10회차)

> 기준일 2026-11-01. 시그니처는 제안이며 차승희 박사 회차에서 확정한다. 데이터 원본은 `build/wage_db.json`, 계산 규칙은 `build/wage_rules.py`.

| Tool | 입력 | 출력 | 쓰임 |
|---|---|---|---|
| `calc_wage` | year, weeks, days, hours_per_day, night_per_day, wage, size, holiday_paid, premium_paid | 정산 JSON(issue·weekly_hours·min_wage·weekly_holiday_pay·unpaid_krw·law_article) | 회귀형 필드를 LLM이 직접 계산하지 않고 계산기에 맡김. R 유형에서 효과가 가장 클 것으로 예상 |
| `lookup_workplace` | 사업장 이름 | 업종, 상시근로자 수, 연도별 약정 시급 | K 유형: 입력에 없는 시급·규모 조회 |
| `get_min_wage` | year | 최저시급(원) | 2027년 고시처럼 모델이 모르는 값 조회 |
