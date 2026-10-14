#!/usr/bin/env python3
"""
결과 요약표·그래프 생성 (카드 4: C31, C34)
원자료 폴더의 results/metrics.csv 에서 직접 계산하므로, 표의 모든 숫자는 원자료로 되짚어 갈 수 있다.
이 스크립트는 새 실험을 하지 않으며 exp 폴더를 수정하지 않는다. 결과는 analysis/ 폴더에 저장한다.

사용법:
  python summarize.py --exp exp1
  python summarize.py --exp exp2
"""
import argparse
import csv
import os
import statistics as st

METHODS = [("threshold", "임계값 방식"), ("logreg", "로지스틱 회귀")]
COLS = ["recall_high", "recall_mid", "recall_low", "fpr", "drop_high_to_low"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="exp1 또는 exp2 폴더")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()

    with open(os.path.join(args.exp, "results", "metrics.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.exp))

    stats = {}
    for key, _ in METHODS:
        sel = [r for r in rows if r["method"] == key]
        stats[key] = {c: [float(r[c]) for r in sel] for c in COLS}
    n_rep = len(stats["threshold"]["fpr"])

    # ---- 표 (평균 ± 표준편차, 반복 횟수 n_rep)
    header = ["방식", "높음 재현율", "중간 재현율", "낮음 재현율", "오탐률", "하락 폭(높음→낮음)"]
    table = []
    for key, label in METHODS:
        line = [label]
        for c in COLS:
            v = stats[key][c]
            line.append(f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
        table.append(line)
    md = [f"### {name} 결과 (평균 ± 표준편차, {n_rep}회 반복)", "",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in table]

    # ---- 회차별 하락 폭 차이 (1차 판정에 쓰이는 값)
    diffs = [a - b for a, b in zip(stats["threshold"]["drop_high_to_low"], stats["logreg"]["drop_high_to_low"])]
    md += ["", "회차별 (임계값 방식 하락 폭 − 분류기 하락 폭): " + ", ".join(f"{d:.3f}" for d in diffs),
           f"평균: {st.mean(diffs):.4f}  (사전에 정한 기준: 0.10 이상이면 지지, 1차 실험에만 적용)"]
    text = "\n".join(md)
    print(text)
    with open(os.path.join(args.out, f"table_{name}.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # ---- 회차별 원자료를 그대로 옮긴 표 (부록용)
    with open(os.path.join(args.out, f"per_rep_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rep", "method"] + COLS)
        for r in rows:
            w.writerow([r["rep"], r["method"]] + [r[c] for c in COLS])

    # ---- 그래프 (matplotlib 이 없으면 표만 만든다)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib 이 없어 그래프는 만들지 않았습니다: python -m pip install matplotlib)")
        return
    levels = ["recall_high", "recall_mid", "recall_low"]
    labels = ["High entropy", "Mid entropy", "Low entropy"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.36
    for i, (key, _) in enumerate(METHODS):
        means = [st.mean(stats[key][c]) for c in levels]
        errs = [st.stdev(stats[key][c]) if n_rep > 1 else 0 for c in levels]
        xs = [j + (i - 0.5) * width for j in range(3)]
        ax.bar(xs, means, width, yerr=errs, capsize=3, label={"threshold": "Entropy threshold", "logreg": "Logistic regression"}[key])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall (mean ± SD over %d runs)" % n_rep)
    ax.set_title(f"{name}: recall by tunnel entropy level")
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = os.path.join(args.out, f"recall_{name}.png")
    fig.savefig(path, dpi=200)
    print(f"\n그래프 저장: {path}")


if __name__ == "__main__":
    main()
