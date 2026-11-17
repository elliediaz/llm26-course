# S 도메인 Tool 후보 (8~10회차 인계용)

> 기준일 2026-11-08. 시그니처는 제안이며 차승희 박사 회차에서 확정함. 데이터 원본은 `build/sports_db.json`

| Tool | 입력 | 출력 | 용도 |
|---|---|---|---|
| `standings(league, as_of)` | 리그, 날짜 | 그날 기준 순위표 | 월별 순위·순위 질문(K) 확인 |
| `player_stats(name)` | 선수 등록명 | 2026 시즌 기록 JSON | 개인 기록 질문(K) 확인 |
| `calc(metric, **args)` | 승률·게임차·타율·평균자책점·KDA와 재료 수치 | 규약 표기 값 | 계산 질문(R)에서 LLM 산술 오류 보완(`build/s_rules.py`) |
| `normalize_entity(text)` | 약칭이 섞인 질문 | 정식 이름 후보와 리그 | 약칭·동명 판단(F) |
| `schedule(league, date)` | 리그, 날짜 | 경기 일정·결과 | 기준일 뒤 결과 여부 판단 |
