#!/usr/bin/env python3
"""
DNS 터널링 탐지 실험: 엔트로피 임계값 방식 vs 로지스틱 회귀 분류기

[고정된 가설]
터널링 DNS 질의의 도메인 엔트로피를 낮추면, 엔트로피 임계값 방식의 탐지율(재현율)은
크게 떨어지지만 여러 특징을 함께 쓰는 로지스틱 회귀 분류기의 탐지율은 상대적으로 덜 떨어진다.

[고정된 판정 기준: 실행 전에 확정, 결과를 본 뒤 수정하지 않는다]
5회 반복 평균에서, (임계값 방식의 재현율 하락 폭) - (분류기의 재현율 하락 폭) >= 0.10 이면 지지,
그렇지 않으면 기각.  하락 폭 = 재현율(엔트로피 '높음') - 재현율(엔트로피 '낮음')
이 판정은 1차 실험(--train-levels high)에만 적용한다.
2차 실험(--train-levels all)은 탐색적이며 같은 지표를 기술만 한다(지지/기각 판정에 사용하지 않음).

사용법은 README.md 참고.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from collections import Counter

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 고정 설계값
DEFAULT_SEEDS = [11, 22, 33, 44, 55]          # 반복 5회 (회차마다 데이터를 새로 생성)
LEVEL_BINS = {                                 # 접두사 문자열의 샤넌 엔트로피(bit/문자) 구간
    "high": (4.0, 4.6),
    "mid": (3.4, 3.8),
    "low": (2.7, 3.1),
}
LEVEL_K = {"high": (24, 36), "mid": (10, 18), "low": (5, 10)}  # 사용 문자 종류 수 범위
TUNNEL_LEN = (16, 40)                          # 터널링 접두사 길이(점 제외) - 단계와 무관하게 동일
ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"
SERVICE = ["www", "mail", "api", "cdn", "static", "img", "login", "shop", "blog", "news",
           "m", "app", "docs", "support", "secure", "portal", "dev", "test", "files", "media"]
WORDS = ["green", "river", "blue", "cloud", "north", "star", "fast", "secure", "center", "online",
         "global", "update", "mobile", "image", "video", "store", "account", "service", "market",
         "travel", "health", "school", "music", "game", "photo", "design", "data", "smart",
         "city", "world", "prime", "alpha", "delta", "metro", "cafe", "bank", "cargo", "daily"]
FEATURES = ["entropy", "length", "digit_ratio", "n_labels"]
LEVELS = ["high", "mid", "low"]


def shannon(s):
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in Counter(s).values())


def features(prefix):
    flat = prefix.replace(".", "")
    return {
        "entropy": shannon(flat),
        "length": len(flat),
        "digit_ratio": sum(ch.isdigit() for ch in flat) / len(flat),
        "n_labels": prefix.count(".") + 1,
    }


def make_tunnel_prefix(level, rng):
    lo, hi = LEVEL_BINS[level]
    kmin, kmax = LEVEL_K[level]
    for _ in range(20000):
        length = rng.randint(*TUNNEL_LEN)
        alphabet = rng.sample(ALNUM, rng.randint(kmin, kmax))
        s = "".join(rng.choice(alphabet) for _ in range(length))
        if lo <= shannon(s) <= hi:
            break
    else:
        raise RuntimeError(f"엔트로피 구간 {level} 샘플을 만들지 못했습니다")
    n_labels = rng.randint(1, 3)
    cuts = sorted(rng.sample(range(1, length), n_labels - 1))
    parts = [s[a:b] for a, b in zip([0] + cuts, cuts + [length])]
    return ".".join(parts)


def make_benign_normal(rng):
    t = rng.random()
    if t < 0.3:
        return rng.choice(SERVICE)
    if t < 0.7:
        return "-".join(rng.sample(WORDS, rng.randint(2, 3)))
    return rng.choice(SERVICE) + "." + rng.choice(WORDS)


def make_benign_hard(rng):
    """정상이지만 무작위처럼 보이는 이름(해시형 CDN 호스트 등)"""
    length = rng.randint(12, 32)
    alphabet = rng.choice(["0123456789abcdef", ALNUM])
    return "".join(rng.choice(alphabet) for _ in range(length))


def load_tranco(path, top_n):
    domains = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip().isdigit():   # 헤더 줄(rank,domain 등)은 건너뜀
                domains.append(row[1].strip().lower())
            if len(domains) >= top_n:
                break
    return domains


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(rep, seed, args, base_domains):
    rng = random.Random(seed)
    rows = []

    n_hard = round(args.n_benign * args.hard_ratio)
    if base_domains:
        reg = rng.sample(base_domains, args.n_benign)
    else:  # 시험용(--pilot): 합성 도메인
        reg = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 10))) + ".test"
               for _ in range(args.n_benign)]
    for i in range(args.n_benign):
        hard = i < n_hard
        prefix = make_benign_hard(rng) if hard else make_benign_normal(rng)
        rows.append({"label": "benign", "level": "benign", "benign_type": "hard" if hard else "normal",
                     "prefix": prefix, "qname": f"{prefix}.{reg[i]}"})

    for level in LEVELS:
        for _ in range(args.n_tunnel_per_level):
            prefix = make_tunnel_prefix(level, rng)
            dom = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 9))) + ".example"
            rows.append({"label": "tunnel", "level": level, "benign_type": "",
                         "prefix": prefix, "qname": f"{prefix}.{dom}"})

    # 학습/시험 분할
    #  1차(기본, --train-levels high): 학습 = 정상 절반 + 높음 터널링 절반 (방어자는 낮춘 회피 형태를 모른다고 가정)
    #  2차(탐색적, --train-levels all): 학습 = 정상 절반 + 높음·중간·낮음 터널링 각 절반
    # 1차 분할의 난수 사용 순서는 2차 옵션과 무관하게 유지된다(2차용 추가 셔플은 그 뒤에 수행).
    idx_benign = [i for i, r in enumerate(rows) if r["label"] == "benign"]
    idx_high = [i for i, r in enumerate(rows) if r["level"] == "high"]
    rng.shuffle(idx_benign)
    rng.shuffle(idx_high)
    train = set(idx_benign[: len(idx_benign) // 2]) | set(idx_high[: len(idx_high) // 2])
    if args.train_levels == "all":
        for lv in ("mid", "low"):
            idx = [i for i, r in enumerate(rows) if r["level"] == lv]
            rng.shuffle(idx)
            train |= set(idx[: len(idx) // 2])
    for i, r in enumerate(rows):
        r["id"] = f"r{rep}_{i:05d}"
        r["rep"] = rep
        r["seed"] = seed
        r["split"] = "train" if i in train else "test"
        for k, v in features(r["prefix"]).items():
            r[k] = round(v, 6)
    return rows


def choose_threshold(train_rows):
    """학습 데이터에서 균형 정확도 (재현율+특이도)/2 가 최대인 엔트로피 임계값"""
    ent = np.array([r["entropy"] for r in train_rows])
    y = np.array([1 if r["label"] == "tunnel" else 0 for r in train_rows])
    best_t, best_score = None, -1.0
    for t in sorted(set(ent)):
        pred = ent >= t
        tpr = pred[y == 1].mean()
        tnr = (~pred)[y == 0].mean()
        score = (tpr + tnr) / 2
        if score > best_score:
            best_t, best_score = float(t), float(score)
    return best_t


def run_rep(rep, seed, rows, outdir):
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    X_train = np.array([[r[f] for f in FEATURES] for r in train])
    y_train = np.array([1 if r["label"] == "tunnel" else 0 for r in train])
    X_test = np.array([[r[f] for f in FEATURES] for r in test])

    thr = choose_threshold(train)
    pred_a = (X_test[:, 0] >= thr).astype(int)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    clf.fit(X_train, y_train)
    pred_b = clf.predict(X_test)

    preds = []
    metrics = []
    for method, pred in (("threshold", pred_a), ("logreg", pred_b)):
        for r, p in zip(test, pred):
            preds.append({"rep": rep, "id": r["id"], "method": method, "pred": int(p)})
        rec = {}
        for level in LEVELS:
            m = np.array([r["level"] == level for r in test])
            rec[level] = float(pred[m].mean())
        mb = np.array([r["label"] == "benign" for r in test])
        fpr = float(pred[mb].mean())
        metrics.append({"rep": rep, "method": method,
                        "threshold": round(thr, 6) if method == "threshold" else "",
                        "recall_high": round(rec["high"], 6), "recall_mid": round(rec["mid"], 6),
                        "recall_low": round(rec["low"], 6), "fpr": round(fpr, 6),
                        "drop_high_to_low": round(rec["high"] - rec["low"], 6)})
    return preds, metrics


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranco", help="Tranco 인기 도메인 목록 CSV 경로 (rank,domain 형식)")
    ap.add_argument("--tranco-date", help="Tranco 목록을 받은 날짜 (예: 2026-10-14), --tranco 사용 시 필수")
    ap.add_argument("--top-n", type=int, default=5000)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--n-benign", type=int, default=1000)
    ap.add_argument("--n-tunnel-per-level", type=int, default=1000)
    ap.add_argument("--hard-ratio", type=float, default=0.3)
    ap.add_argument("--train-levels", choices=["high", "all"], default="high",
                    help="high: 1차(본 결과) / all: 2차(탐색적, 판정 없음)")
    ap.add_argument("--pilot", action="store_true", help="시험 실행(합성 정상 도메인 허용). 결과는 본 결과로 쓰지 않는다.")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if not args.pilot:
        if not args.tranco or not args.tranco_date:
            sys.exit("본 실행에는 --tranco 와 --tranco-date 가 필요합니다. (시험 실행은 --pilot)")
        if seeds != DEFAULT_SEEDS or args.n_benign != 1000 or args.n_tunnel_per_level != 1000:
            sys.exit("본 실행은 고정된 설계값(시드 5개, 1000/1000)만 허용합니다. 바꿨다면 시험 실행(--pilot)으로 하세요.")

    base_domains = load_tranco(args.tranco, args.top_n) if args.tranco else None
    if base_domains is not None and len(base_domains) < args.n_benign:
        sys.exit("Tranco 목록이 정상 도메인 수보다 적습니다.")

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)

    sample_fields = ["id", "rep", "seed", "label", "level", "benign_type", "prefix", "qname",
                     "entropy", "length", "digit_ratio", "n_labels", "split"]
    all_preds, all_metrics = [], []
    for rep, seed in enumerate(seeds, start=1):
        rows = build_dataset(rep, seed, args, base_domains)
        write_csv(os.path.join(args.out, "data", f"rep{rep}_samples.csv"), rows, sample_fields)
        preds, metrics = run_rep(rep, seed, rows, args.out)
        all_preds += preds
        all_metrics += metrics
        print(f"[rep {rep}] seed={seed} 샘플 {len(rows)}건 생성, 시험 {len(preds)//2}건 평가")

    write_csv(os.path.join(args.out, "results", "predictions.csv"), all_preds, ["rep", "id", "method", "pred"])
    mfields = ["rep", "method", "threshold", "recall_high", "recall_mid", "recall_low", "fpr", "drop_high_to_low"]
    write_csv(os.path.join(args.out, "results", "metrics.csv"), all_metrics, mfields)

    # ---- 고정된 판정 기준 적용
    drop = {m: [r["drop_high_to_low"] for r in all_metrics if r["method"] == m] for m in ("threshold", "logreg")}
    diffs = [a - b for a, b in zip(drop["threshold"], drop["logreg"])]
    mean_diff = float(np.mean(diffs))
    summary = {
        "pilot": bool(args.pilot),
        "train_levels": args.train_levels,
        "analysis_type": "1차(본 결과)" if args.train_levels == "high" else "2차(탐색적, 지지/기각 판정에 사용하지 않음)",
        "reps": len(seeds),
        "mean_drop_threshold": float(np.mean(drop["threshold"])),
        "mean_drop_logreg": float(np.mean(drop["logreg"])),
        "std_drop_threshold": float(np.std(drop["threshold"], ddof=1)) if len(seeds) > 1 else None,
        "std_drop_logreg": float(np.std(drop["logreg"], ddof=1)) if len(seeds) > 1 else None,
        "mean_diff_threshold_minus_logreg": mean_diff,
        "rule": "mean_diff >= 0.10 이면 지지, 아니면 기각",
        "verdict": ("지지" if mean_diff >= 0.10 else "기각") if args.train_levels == "high" else "판정 없음(탐색적)",
    }
    with open(os.path.join(args.out, "results", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    config = {
        "args": vars(args), "seeds": seeds, "level_bins": LEVEL_BINS, "level_k": LEVEL_K,
        "tunnel_len": TUNNEL_LEN, "features": FEATURES,
        "tranco_sha256": sha256(args.tranco) if args.tranco else None,
        "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
    }
    with open(os.path.join(args.out, "results", "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.pilot:
        print("\n※ 시험 실행입니다. 이 결과는 본 결과로 사용하지 마세요.")


if __name__ == "__main__":
    main()
