"""2회차 해보기 F: 생성하지 않고 LLM을 분류기로 쓰기(후보 라벨의 로그확률 비교)

  python experiments/s02/label_score.py                       # G 시드 10건 category, 라벨 로그확률 합
  python experiments/s02/label_score.py --norm mean           # 합 대신 평균: 짧은 라벨(기타)로 쏠리는지
  python experiments/s02/label_score.py --model skt/A.X-4.0-Light   # 큰 모델(GPU 서버)

볼 것 : 라벨마다 토큰 수가 다르면 합은 짧은 라벨에 유리함. 합과 평균의 정확도, 예측이 한 라벨로 쏠리는지
메모(CPU, Qwen3-0.6B) : 합 0.20, 평균 0.10. 작은 모델은 생성 없는 분류에 약함. 1회차 ticket_classifier.py(시드 10건 0.80)와 비교
더 다듬은 판(보정·예시·데이터 5종) : src/llm_classify.py
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

LABELS = ["버그", "밸런스", "결제", "계정", "이벤트", "제재", "기타"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def label_logprob(model, tok, prompt: str, label: str, norm: str = "sum") -> float:
    p_ids = tok(prompt, return_tensors="pt")["input_ids"]
    full = tok(prompt + " " + label, return_tensors="pt")["input_ids"].to(DEV)
    logp = torch.log_softmax(model(full).logits[0, :-1].float(), dim=-1)   # 위치 t의 출력 = t+1 번째 토큰 예측
    target = full[0, 1:]
    start = p_ids.shape[1] - 1                                            # 라벨이 시작되는 예측 위치
    lp = logp[start:].gather(1, target[start:, None])
    return (lp.mean() if norm == "mean" else lp.sum()).item()   # sum은 토큰이 적은(짧은) 라벨에 유리함


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--norm", choices=["sum", "mean"], default="sum", help="라벨 토큰 로그확률의 합(기본) 또는 평균")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(DEV).eval()
    items = json.load(open(Path(__file__).resolve().parents[2] / "data" / "G" / "test_seed.json", encoding="utf-8"))
    hit = 0
    for it in items:
        prompt = ("게임 고객 문의를 다음 중 하나로 분류하시오: " + ", ".join(LABELS)
                  + f"\n문의: {it['input']}\n분류:")
        scores = {lab: label_logprob(model, tok, prompt, lab, a.norm) for lab in LABELS}
        pred = max(scores, key=scores.get)
        hit += pred == it["gold"]["category"]
        print(f"{it['id']}  정답 {it['gold']['category']:4s} 예측 {pred:4s}  {it['input'][:30]}")
    print(f"\ncategory 정확도(시드 {len(items)}건): {hit / len(items):.2f}  (1회차 ticket_classifier.py 결과와 비교)")


if __name__ == "__main__":
    main()
