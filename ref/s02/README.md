# 2회차 참고 결과

- `gen_sweep_main_7b.md`·`.json`: 메인 LLM(skt/A.X-4.0-Light, bf16)으로 돌린 생성 설정 비교표. 2026-11-15 팀 GPU 서버(RTX PRO 4500 32GB)에서 `python src/gen_params_sweep.py`를 그대로 실행한 결과(6개 조합 × 프롬프트 6건 × 3회, 2.3분)
- 개인 CPU 서버에서는 같은 비교를 `python src/gen_params_sweep.py --model Qwen/Qwen3-0.6B`로 돌림(약 13분). 7B를 CPU에서 돌리면 초당 1토큰 정도라 수업 시간 안에 끝나지 않음
- 두 표를 나란히 놓고 "작은 모델과 메인 LLM에서 다양성·JSON 비율이 어떻게 다른가"를 본 뒤 팀의 `GEN_SERVICE`를 정함
