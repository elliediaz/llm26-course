"""5~7회차: 성능 기록표: 같은 평가 기준으로 잰 네 구성(Baseline·RAG·LoRA·RAG+LoRA)을 한 표에 나란히 둔다.

  python src/scoreboard.py                    # 표 보기(results/scoreboard.md 갱신)
  python src/scoreboard.py --target 0.75      # 5회차: 개선 목표치 기록
  python src/scoreboard.py --swap G-T-07 G-P-K03 --reason "Baseline 85%로 높아 어려운 문항으로 교체"
  python src/scoreboard.py --plot             # 7회차: 4구성 막대그래프(results/fig/scoreboard.png)

30문항이면 1문항이 3.3%p 다. 3문항(10%p) 미만 차이는 "노이즈일 수 있음"으로 적는다.
"""
from __future__ import annotations

import argparse

import config
import utils

PATH = config.PATHS["results"] / "scoreboard.json"
COLS = ["구성", "측정 시각", "모델", "방식", "Task Acc", "JSON", "EM", "K", "R", "F", "p50(s)", "p95(s)", "VRAM(GB)", "비고"]


def load() -> dict:
    return utils.read_json(PATH) if PATH.exists() else {"rows": [], "target": None, "swaps": []}


def save(sb: dict) -> None:
    utils.write_json(PATH, sb)
    render(sb)


def add_row(payload: dict, note: str = "") -> None:
    sb = load()
    m, c = payload["metrics"], payload["config"]
    row = {
        "구성": payload["label"], "측정 시각": payload["measured_at"][5:16], "모델": c["model_id"].split("/")[-1],
        "방식": c["strategy"], "Task Acc": m["task_accuracy"], "JSON": m["json_success"], "EM": m["exact_match"],
        "K": m["by_type"].get("K"), "R": m["by_type"].get("R"), "F": m["by_type"].get("F"),
        "p50(s)": m["latency"]["p50"], "p95(s)": m["latency"]["p95"], "VRAM(GB)": m["vram_peak_gb"],
        "비고": note, "n": m["n"], "result": payload["name"],
    }
    sb["rows"].append(row)
    save(sb)
    print(f"[성능 기록표] '{row['구성']}' 행 추가 → {config.PATHS['results'] / 'scoreboard.md'}")


def render(sb: dict) -> str:
    lines = [f"# 성능 기록표: 도메인 {config.DOMAIN}", ""]
    if sb.get("target"):
        lines += [f"- 개선 목표: Task Accuracy {sb['target']}", ""]
    lines.append(utils.print_table(sb["rows"], COLS) if sb["rows"] else "(아직 측정 없음)")
    base = next((r for r in sb["rows"] if r["구성"] == "Baseline"), None)
    if base:
        lines += ["", "## Baseline 대비 변화(Task Accuracy)", ""]
        n = base.get("n") or 30
        for r in sb["rows"]:
            if r is base:
                continue
            d = r["Task Acc"] - base["Task Acc"]
            flag = " (노이즈일 수 있음: 3문항 미만 차이)" if abs(d) < 3 / n else ""
            lines.append(f"- {r['구성']}({r['방식']}): {d:+.3f}{flag}")
    if sb.get("swaps"):
        lines += ["", "## 문항 교체 이력", ""] + [f"- {s['out']} → {s['in']}: {s['reason']}" for s in sb["swaps"]]
    text = "\n".join(lines) + "\n"
    (config.PATHS["results"] / "scoreboard.md").write_text(text, encoding="utf-8")
    return text


def plot(sb: dict) -> None:
    utils.korean_font()
    import matplotlib.pyplot as plt
    rows = sb["rows"]
    if not rows:
        print("측정 없음")
        return
    labels = [f"{r['구성']}\n({r['방식']})" for r in rows]
    metrics = ["Task Acc", "JSON", "EM"]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(rows)), 4))
    w = 0.8 / len(metrics)
    for j, m in enumerate(metrics):
        ax.bar([i + j * w for i in range(len(rows))], [r[m] or 0 for r in rows], w, label=m)
    ax.set_xticks([i + w for i in range(len(rows))], labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"성능 기록표: 도메인 {config.DOMAIN}")
    ax.legend()
    fig.tight_layout()
    out = utils.fig_path("scoreboard.png")
    fig.savefig(out, dpi=150)
    print(f"저장: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, help="개선 목표 Task Accuracy")
    ap.add_argument("--swap", nargs=2, metavar=("OUT_ID", "IN_ID"), help="문항 교체 이력 기록")
    ap.add_argument("--reason", default="")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    sb = load()
    if a.target is not None:
        sb["target"] = a.target
    if a.swap:
        sb["swaps"].append({"out": a.swap[0], "in": a.swap[1], "reason": a.reason, "at": utils.now()})
    save(sb)
    print(render(sb))
    if a.plot:
        plot(sb)


if __name__ == "__main__":
    main()
