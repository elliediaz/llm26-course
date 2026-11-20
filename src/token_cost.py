"""2회차 미션 M3: 토큰 요금 계산기. 같은 문장도 모델(토크나이저)에 따라 토큰 수와 요금이 달라짐

  python src/token_cost.py                                # 표본 문장 8종 × 토크나이저 4종
  python src/token_cost.py --text "점메추 해 줘" "Recommend lunch"   # 쪼개진 모양 보기
  python src/token_cost.py --traffic delivery --daily 20000 --price 2000   # 우리 서비스 한 달 요금 추정

토크나이저 4종
  A.X-4.0-Light  한국어 데이터를 많이 넣은 국내 모델(4회차 이후 메인 후보)
  Qwen3-0.6B     다국어 모델(2회차 실습용)
  KURE-v1        한국어 임베딩 모델(XLM-RoBERTa 계열, 6회차 검색용)
  GPT-2          2019년 영어 중심 모델(비교 기준)
요금은 가상 단가(백만 토큰당 원)로 계산함. 2,000원은 비교용으로 정한 값일 뿐이고, 실제 API 단가는 서비스마다 다르며 입력·출력 단가도 다름
결과: results/token_cost.json, results/fig/token_cost.png
"""
from __future__ import annotations

import argparse

from transformers import AutoTokenizer

import config
import utils

DATA = config.ROOT / "data" / "s02"
TOKENIZERS = {"A.X-4.0-Light": config.MODEL_ID, "Qwen3-0.6B": config.SMALL_MODEL_ID,
              "KURE-v1": config.EMBED_MODEL, "GPT-2": "openai-community/gpt2"}
SYSTEM = ("너는 배달앱 고객센터 상담원이다. 고객의 리뷰나 문의를 읽고 불만 유형을 먼저 파악한 뒤, 정중하고 짧게 답한다. "
          "환불·재배달·쿠폰 지급은 직접 약속하지 말고 담당 부서로 연결한다고 안내한다. 개인정보(전화번호·주소·카드번호)는 "
          "묻지 않는다. 음식 사진이 필요하면 앱의 사진 첨부 기능을 안내한다. 답변은 세 문장을 넘기지 않는다. 고객이 화가 나 "
          "있으면 먼저 사과하고, 칭찬 리뷰에는 감사 인사와 함께 다음 주문을 기다린다고 답한다.")
TRAFFIC = {"delivery": "delivery_reviews.json", "smishing": "smishing.json", "news": "news_ynat.json", "game": "game_inquiry.json"}


def pieces(tok, text: str) -> list[str]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out = []
    for i in ids:
        p = tok.decode([i])
        out.append(p.replace("\ufffd", "·") or "_")      # ·: 한 글자가 여러 토큰으로 갈라진 청크, _: 띄어쓰기만 담은 토큰
    return out


def load_all(names):
    toks = {}
    for n in names:
        try:
            toks[n] = AutoTokenizer.from_pretrained(TOKENIZERS[n])
        except Exception as e:                         # 내려받기 실패 시 그 모델만 건너뜀
            print(f"[건너뜀] {n}: {type(e).__name__}")
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*", help="쪼개진 모양을 볼 문장")
    ap.add_argument("--models", nargs="*", default=list(TOKENIZERS), choices=list(TOKENIZERS))
    ap.add_argument("--traffic", default="delivery", choices=["delivery", "smishing", "news", "game"],
                    help="요금 추정에 쓸 문장 묶음(data/s02)")
    ap.add_argument("--daily", type=int, default=10000, help="하루 요청 수")
    ap.add_argument("--answer-ratio", type=float, default=3.0, help="답변 길이 = 질문 토큰 × 이 값")
    ap.add_argument("--system", help="요청마다 붙는 지시문(시스템 프롬프트) 텍스트 파일. 없으면 기본 상담원 지시문")
    ap.add_argument("--price", type=float, default=2000, help="가상 단가: 백만 토큰당 원")
    a = ap.parse_args()
    toks = load_all(a.models)

    if a.text:
        for t in a.text:
            print(f"\n{t}  ({len(t)}자)")
            for n, tok in toks.items():
                p = pieces(tok, t)
                print(f"  {n:14s} {len(p):3d}토큰  " + "|".join(p))
        return

    zoo = utils.read_json(DATA / "token_zoo.json")["categories"]
    rows, per_cat = [], {}
    for c in zoo:
        row = {"종류": c["category"], "글자": sum(len(t) for t in c["texts"])}
        for n, tok in toks.items():
            row[n] = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in c["texts"])
        rows.append(row)
        per_cat[c["category"]] = {n: round(row["글자"] / row[n], 2) for n in toks}
    print("토큰 수(문장 2개 합): 적을수록 싸고 빠르고 길게 넣을 수 있음")
    utils.print_table(rows)
    print("\n토큰 하나가 담는 글자 수(클수록 효율적)")
    utils.print_table([{"종류": k, **v} for k, v in per_cat.items()])

    d = utils.read_json(DATA / TRAFFIC[a.traffic])
    traffic = [r["text"] for r in d["dev"] + d["test"]]
    system = open(a.system, encoding="utf-8").read().strip() if a.system else SYSTEM
    cost = []
    for n, tok in toks.items():
        sys_n = len(tok(system, add_special_tokens=False)["input_ids"])
        q = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in traffic) / len(traffic)
        per_req = sys_n + q + q * a.answer_ratio
        month = per_req * a.daily * 30
        cost.append({"토크나이저": n, "지시문": sys_n, "질문 평균": f"{q:.1f}", "요청당": f"{per_req:.0f}",
                     "월 토큰": f"{month / 1e6:.1f}백만", "월 요금": f"{month / 1e6 * a.price:,.0f}원"})
    print(f"\n월 요금 추정(토큰 수): 하루 {a.daily:,}건 · 지시문 {len(system)}자 · 답변 = 질문×{a.answer_ratio} · "
          f"가상 단가 {a.price:,.0f}원/백만 토큰")
    utils.print_table(cost)
    utils.write_json(utils.result_path("token_cost.json"), {"rows": rows, "chars_per_token": per_cat, "cost": cost, "args": vars(a)})

    import matplotlib.pyplot as plt
    utils.korean_font()
    cats = list(per_cat)
    names = list(toks)
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = ["#c0622b", "#2b5c9e", "#6a9a5b", "#999999"]
    for i, n in enumerate(names):
        ax.bar([x + i * w for x in range(len(cats))], [per_cat[c][n] for c in cats], w, label=n, color=colors[i % 4])
    ax.set_xticks([x + w * (len(names) - 1) / 2 for x in range(len(cats))], cats, fontsize=8, rotation=15)
    ax.set_ylabel("토큰 하나당 글자 수"); ax.legend(fontsize=8)
    ax.set_title("같은 문장, 토크나이저마다 다른 효율(높을수록 토큰이 적게 듦)", fontsize=10)
    fig.tight_layout()
    out = utils.fig_path("token_cost.png")
    fig.savefig(out, dpi=130)
    print("그림:", out)


if __name__ == "__main__":
    main()
