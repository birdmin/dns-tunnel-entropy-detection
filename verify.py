#!/usr/bin/env python3
"""
결과 재확인 스크립트 (C26, C29)
1) 원자료 건수가 설계와 일치하는지
2) 저장된 엔트로피 등 특징값이 접두사 문자열에서 다시 계산한 값과 같은지
3) predictions.csv 에서 다시 계산한 재현율·오탐률이 metrics.csv 와 같은지
사용법: python verify.py --out <실험 출력 폴더>
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict


def shannon(s):
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in Counter(s).values())


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("OK   " if cond else "FAIL ") + msg)
        ok = ok and cond

    cfg = json.load(open(os.path.join(args.out, "results", "config.json"), encoding="utf-8"))
    n_b = cfg["args"]["n_benign"]
    n_t = cfg["args"]["n_tunnel_per_level"]
    seeds = cfg["seeds"]

    samples = {}
    for rep in range(1, len(seeds) + 1):
        rows = read(os.path.join(args.out, "data", f"rep{rep}_samples.csv"))
        samples[rep] = {r["id"]: r for r in rows}
        c = Counter(r["level"] for r in rows)
        check(len(rows) == n_b + 3 * n_t, f"rep{rep}: 총 {len(rows)}건 (설계 {n_b + 3 * n_t}건)")
        check(c["benign"] == n_b and all(c[l] == n_t for l in ("high", "mid", "low")),
              f"rep{rep}: 정상 {c['benign']}, 높음 {c['high']}, 중간 {c['mid']}, 낮음 {c['low']}")
        bad = sum(abs(shannon(r["prefix"].replace(".", "")) - float(r["entropy"])) > 1e-5 for r in rows)
        check(bad == 0, f"rep{rep}: 엔트로피 재계산 불일치 {bad}건")

    preds = read(os.path.join(args.out, "results", "predictions.csv"))
    metrics = read(os.path.join(args.out, "results", "metrics.csv"))
    group = defaultdict(list)
    for p in preds:
        group[(int(p["rep"]), p["method"])].append(p)
    for m in metrics:
        key = (int(m["rep"]), m["method"])
        rows = group[key]
        stat = defaultdict(list)
        for p in rows:
            r = samples[key[0]][p["id"]]
            stat["fpr" if r["label"] == "benign" else "recall_" + r["level"]].append(int(p["pred"]))
        for name in ("recall_high", "recall_mid", "recall_low", "fpr"):
            val = sum(stat[name]) / len(stat[name])
            check(abs(val - float(m[name])) < 1e-5, f"rep{key[0]} {key[1]} {name} 재계산 {val:.4f} = 저장 {m[name]}")
    print("\n결과:", "모두 일치" if ok else "불일치 있음")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
