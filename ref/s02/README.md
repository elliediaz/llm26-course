# 2회차 참고 결과

- `gen_sweep_main_7b.md`·`.json`: 메인 LLM(skt/A.X-4.0-Light, bf16)으로 돌린 생성 설정 비교표. 2026-11-15 팀 GPU 서버(RTX PRO 4500 32GB)에서 `python src/gen_params_sweep.py`를 그대로 실행한 결과(6개 조합 × 프롬프트 6건 × 3회, 2.3분)
- 2026-11-16: JSON 판정을 두 열로 나눠 `.json`에 저장된 생성 결과(samples)에서 다시 계산함. **JSON(느슨)** 코드 울타리(```json … ```)를 벗기면 JSON으로 읽히는 비율, **JSON만(엄격)** 울타리·설명 없이 JSON만 낸 비율. 예전 표의 'JSON' 한 열은 배열 답에 울타리가 붙으면 실패로 세어 온도에 따른 하락이 실제보다 크게 보였음
- 개인 CPU 서버에서는 같은 비교를 `python src/gen_params_sweep.py --model Qwen/Qwen3-0.6B`로 돌림(11/16 실측 653초). 7B를 CPU에서 돌리면 초당 1토큰 정도라 수업 시간 안에 끝나지 않음
- `gen_sweep_qwen3_06b.md`·`.json`: 같은 비교를 개인 CPU 서버에서 Qwen3-0.6B로 돌린 결과(2026-11-16, 653초). 걸어 둔 비교가 수업 시간 안에 끝나지 않았을 때 이 표로 실습 2 ⑥을 진행함
- 두 표를 나란히 놓고 "작은 모델과 메인 LLM에서 같은 문의의 분류 값이 얼마나 달라지는가"를 본 뒤 팀의 `GEN_SERVICE`를 정함. 생성 결과 원문은 `.json`의 `samples`에 있음
