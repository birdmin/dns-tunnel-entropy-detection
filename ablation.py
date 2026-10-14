#!/usr/bin/env python3
"""
특징 제거 분석 (카드 4: 다르게 나온 이유를 데이터로 확인) - 사후 탐색적 분석

질문: 로지스틱 회귀가 엔트로피를 낮춘 터널링을 어느 정도 잡은 것은 엔트로피 덕분인가,
      아니면 길이·숫자 비율·레이블 개수 같은 다른 특징 덕분인가?

방법: exp 폴더의 원자료(data/rep*_samples.csv)와 그 안의 split 열(학습/시험 구분)을 그대로 쓴다.
      특징 조합만 바꿔 로지스틱 회귀를 다시 학습하고, 시험 데이터의 단계별 재현율·오탐률을 구한다.
      (모델 설정은 본 실험과 같다: StandardScaler + LogisticRegression, 시드 동일)

이 분석은 사후 탐색적 분석이며 가설의 지지/기각 판정에 쓰지 않는다. exp 폴더는 수정하지 않고
결과는 analysis/ 폴더에 저장한다.

사용법:
  python ablation.py --exp exp1
  python ablation.py --exp exp2
"""
import argparse
import csv
import glob
import os
import statistics as st

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATS = ["entropy", "length", "digit_ratio", "n_labels"]
VARIANTS = {
    "all (본 실험과 동일)": FEATS,
    "entropy만": ["entropy"],
    "entropy 제외": [f for f in FEATS if f != "entropy"],
    "length 제외": [f for f in FEATS if f != "length"],
    "digit_ratio 제외": [f for f in FEATS if f != "digit_ratio"],
    "n_labels 제외": [f for f in FEATS if f != "n_labels"],
}
LEVELS = ["high", "mid", "low"]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.exp))
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.exp, "data", "rep*_samples.csv")))
    saved = {(int(r["rep"]), r["method"]): r for r in read(os.path.join(args.exp, "results", "metrics.csv"))}

    result = {v: {k: [] for k in LEVELS + ["fpr"]} for v in VARIANTS}
    check = []
    for fp in files:
        rows = read(fp)
        rep = int(rows[0]["rep"])
        seed = int(rows[0]["seed"])
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "test"]
        y_train = np.array([1 if r["label"] == "tunnel" else 0 for r in train])
        for vname, feats in VARIANTS.items():
            X_train = np.array([[float(r[f]) for f in feats] for r in train])
            X_test = np.array([[float(r[f]) for f in feats] for r in test])
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            for lv in LEVELS:
                m = np.array([r["level"] == lv for r in test])
                result[vname][lv].append(float(pred[m].mean()))
            mb = np.array([r["label"] == "benign" for r in test])
            result[vname]["fpr"].append(float(pred[mb].mean()))
        # 본 실험의 저장값과 같은지 점검(전체 특징 버전)
        s = saved[(rep, "logreg")]
        v = result["all (본 실험과 동일)"]
        check.append(max(abs(v["high"][-1] - float(s["recall_high"])), abs(v["mid"][-1] - float(s["recall_mid"])),
                         abs(v["low"][-1] - float(s["recall_low"])), abs(v["fpr"][-1] - float(s["fpr"]))))

    n = len(files)
    print(f"[{name}] {n}회 평균 (전체 특징 버전과 저장된 본 실험 값의 최대 차이: {max(check):.1e})")
    header = ["특징 조합", "높음 재현율", "중간 재현율", "낮음 재현율", "오탐률"]
    lines = []
    for vname in VARIANTS:
        r = result[vname]
        lines.append([vname] + [f"{st.mean(r[k]):.3f}" for k in LEVELS + ["fpr"]])
    w = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(header)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(header)))
    for l in lines:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(l)))

    with open(os.path.join(args.out, f"ablation_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header + ["반복 횟수"])
        for l in lines:
            wr.writerow(l + [n])
    print(f"\n저장: {os.path.join(args.out, f'ablation_{name}.csv')}")


if __name__ == "__main__":
    main()
