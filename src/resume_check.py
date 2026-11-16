"""복귀 점검·인계 점검: 회차 시작 전에 환경과 이전 산출물이 그대로 있는지 확인한다.

  python src/resume_check.py              # 5·6·7회차 도입(복귀 점검): GPU·모델 로딩·이전 산출물
  python src/resume_check.py --handover 2 # 2회차 끝(차승희 박사 회차 인계 전): 남겨야 할 파일 확인
  python src/resume_check.py --handover 7 # 7회차 끝(8회차 인계 전)
  python src/resume_check.py --quick      # 모델 로딩 없이 파일만

판정이 '누락'이면 강사·TA에게 알린다. 볼륨(/workspace)에 있는 파일은 Pod를 꺼도 남는다.
"""
from __future__ import annotations

import argparse

import config
import utils

R, D = config.PATHS["results"], config.data_dir()
CHECKS = {
    "1회차 산출물": [R / "env_check.json", R / "faq_eval.json"],
    "2회차 산출물": [R / "xor_demo.json", R / "seq_limits.json", R / "llm_sentiment_Qwen3-0.6B_4shot.json"],   # V2(11/17)
    "5회차 산출물": [D / "test_dataset.json", R / "base.json", R / "scoreboard.json"],
    "6회차 산출물": [R / "rag.json", R / "retrieval_compare.json"],
}
HANDOVER = {
    2: ["1회차 산출물", "2회차 산출물"],
    7: ["5회차 산출물", "6회차 산출물"],
}


def check_files(groups) -> list[dict]:
    rows = []
    for g in groups:
        for p in CHECKS[g]:
            rows.append({"구분": g, "파일": str(p.relative_to(config.ROOT)), "상태": "있음" if p.exists() else "누락"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handover", type=int, choices=list(HANDOVER))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print(f"[설정] 도메인 {config.DOMAIN} · 모델 {config.MODEL_ID} · {config.QUANT}")
    print(f"[설정] GEN_SERVICE {config.GEN_SERVICE} · 이유: {config.GEN_SERVICE_REASON}")
    groups = HANDOVER[a.handover] if a.handover else list(CHECKS)
    rows = check_files(groups)
    if a.handover == 7:
        ads = sorted(p.parent for p in config.PATHS["adapters"].glob("*/*/adapter_config.json"))
        rows.append({"구분": "7회차 산출물", "파일": "adapters/<팀>/<run>/", "상태": f"{len(ads)}개" if ads else "누락"})
        idx = sorted(config.PATHS["index"].glob(f"{config.DOMAIN}_*/meta.json"))
        rows.append({"구분": "6회차 산출물", "파일": "index/", "상태": f"{len(idx)}개" if idx else "누락"})
    utils.print_table(rows)
    if not a.quick and not a.handover:
        import torch
        import model_utils
        print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'}")
        model, tok = model_utils.load_llm()
        g = model_utils.generate(model, tok, [{"role": "user", "content": "한 문장으로 인사해 줘."}], {"do_sample": False, "max_new_tokens": 30})
        print(f"생성 확인: {g['text']!r} ({g['latency']}s)")
    miss = [r for r in rows if r["상태"] == "누락"]
    print("\n판정:", "이상 없음" if not miss else f"누락 {len(miss)}건: 해당 회차 스크립트를 다시 실행하거나 강사·TA에게 알릴 것")


if __name__ == "__main__":
    main()
