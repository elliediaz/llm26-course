"""[따라 치기 5-T1] Chat Template 적용 (5회차 코드 구조)

실행: python typing/s05/t1_chat_template.py      (빈칸 ____을 채운 뒤)
확인: ① 역할(system·user·assistant)이 특수 토큰으로 감싸여 한 줄의 글로 바뀜
      ② add_generation_prompt=True 면 끝에 'assistant 차례' 표시가 붙음 → 모델이 답을 시작
      ③ 템플릿 없이 넣은 것과 토큰 수 비교
"""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
msgs = [{"role": "system", "content": "게임 고객 문의를 읽고 상담 티켓 JSON 하나만 출력하시오."},
        {"role": "user", "content": "결제했는데 다이아가 안 들어왔어요"}]

text = tok.____(msgs, tokenize=False, ____=True, enable_thinking=False)
print(text)
plain = "\n\n".join(m["content"] for m in msgs)
print("템플릿 적용 토큰 수:", len(tok(text)["input_ids"]), " / 내용만 이어 붙인 토큰 수:", len(tok(plain)["input_ids"]))
