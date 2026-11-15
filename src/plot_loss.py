"""7회차 실습 1 ④: Loss 곡선 그리기와 판독.

  python src/plot_loss.py --log results/train_log_team3_r16.json

판독표(대략적 기준: 최종 판단은 반드시 같은 하네스로 재평가해서 한다)
  꾸준히 내려감            정상
  거의 안 내려감           학습률이 낮거나 데이터 형식(템플릿)이 어긋남 → 데이터부터 확인
  널뛰기                   학습률이 높음 → 낮춘다
  학습 Loss 만 계속 내려감  과적합(외움) → 에폭을 줄이거나 데이터를 늘린다
  0 근처로 급락            데이터가 너무 쉽거나 중복이 많음 → 데이터 점검
Loss가 낮다고 좋은 모델이 아니다. Loss는 학습 데이터에 맞춘 정도일 뿐이다.
"""
from __future__ import annotations

import argparse
import statistics

import utils


def judge(train: list[float], valid: list[float]) -> list[str]:
    msgs = []
    if len(train) < 4:
        return ["기록이 너무 짧음: 판단 보류"]
    first, last = statistics.mean(train[:3]), statistics.mean(train[-3:])
    drop = (first - last) / first if first else 0
    diffs = [abs(b - a) for a, b in zip(train, train[1:])]
    if drop < 0.1:
        msgs.append(f"학습 Loss 감소 {drop:.0%}: 거의 안 내려감: 템플릿·데이터 형식·학습률 확인")
    else:
        msgs.append(f"학습 Loss {first:.3f} → {last:.3f} ({drop:.0%} 감소)")
    if diffs and statistics.mean(diffs) > 0.25 * statistics.mean(train):
        msgs.append("변동이 큼(널뛰기): 학습률을 낮춰 볼 것")
    if last < 0.02:
        msgs.append("0 근처까지 내려감: 데이터가 너무 쉽거나 중복이 많을 수 있음")
    if len(valid) >= 3 and valid[-1] > min(valid) * 1.1:
        msgs.append(f"검증 Loss가 최저({min(valid):.3f}) 이후 다시 오름({valid[-1]:.3f}): 과적합 신호")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    data = utils.read_json(a.log)
    hist = data["log_history"]
    tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    va = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    for m in judge([x for _, x in tr], [x for _, x in va]):
        print(" -", m)
    utils.korean_font()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(*zip(*tr), label="학습 Loss")
    if va:
        ax.plot(*zip(*va), marker="o", label="검증 Loss")
    info = data.get("info", {})
    ax.set(xlabel="step", ylabel="Loss", title=f"{info.get('team', '')}/{info.get('run', '')} r={info.get('r')} · {info.get('n_train')}건")
    ax.legend()
    fig.tight_layout()
    out = utils.fig_path(f"loss_{info.get('team', 'x')}_{info.get('run', 'x')}.png")
    fig.savefig(out, dpi=150)
    print(f"그림: {out}")


if __name__ == "__main__":
    main()
