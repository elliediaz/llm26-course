"""내 실습 결과 묶기: 서버(볼륨)는 과정이 끝나면 지워지므로 남길 파일을 zip 한 개로 묶어 내려받음.

  python src/pack_results.py                  # 결과·판단 기록·내 설정·내 문서·학습 데이터(my/ 폴더 포함) → my_results_<이름>_<날짜>.zip
  python src/pack_results.py --with-adapter   # Adapter 가중치(개당 약 40MB)까지 넣음

묶은 파일은 Jupyter 왼쪽 파일 창에서 오른쪽 단추 → Download 로 내려받음. 10회차 Portfolio 의 근거 자료로 씀.
"""
from __future__ import annotations

import argparse
import datetime
import zipfile

import config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-adapter", action="store_true")
    a = ap.parse_args()
    root, d = config.ROOT, config.data_dir()
    files = [p for p in config.PATHS["results"].rglob("*") if p.is_file() and p.suffix in (".json", ".md", ".csv", ".png", ".txt")]
    files += [p for p in (root / "my_config.json", d / "test_dataset.json") if p.exists()]
    files += [p for p in (d / "corpus_updates").glob("my_*")] + [p for p in d.glob("my_*") if p.is_file()]
    files += [p for p in (root / "prompts").rglob("my_*")]
    for cfg in config.PATHS["adapters"].glob("*/*/adapter_config.json"):
        files += [p for p in cfg.parent.iterdir() if p.is_file() and (a.with_adapter or p.suffix in (".json", ".md"))]
    my = root / "my"                                        # 개인 샌드박스: 문서 · 설정 · 확인 질문 · 학습 데이터 · 기록 · 보고서
    if my.exists():
        files += [p for p in my.rglob("*") if p.is_file() and "adapters" not in p.relative_to(my).parts
                  and p.stat().st_size < 20 * 2**20]
        for cfg in (my / "adapters").glob("*/adapter_config.json"):
            files += [p for p in cfg.parent.iterdir() if p.is_file() and (a.with_adapter or p.suffix in (".json", ".md"))]
    out = root / f"my_results_{config.MY_NAME}_{datetime.date.today():%m%d}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(root))
    size = out.stat().st_size
    print(f"{out.name} : 파일 {len(set(files))}개, " + (f"{size / 1e6:.1f}MB" if size >= 1e6 else f"{max(1, round(size / 1e3))}KB"))
    print("Jupyter 왼쪽 파일 창(course 폴더)에서 이 파일을 오른쪽 단추 → Download 로 내려받음")


if __name__ == "__main__":
    main()
