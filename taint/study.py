"""Validation-only threshold selection and held-out detector diagnostics."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from taint.replay import evaluate_rows


def calibration(rows, bins=10):
    if not rows:
        return {"brier": None, "ece": None, "bins": []}
    buckets = [[] for _ in range(bins)]
    for row in rows:
        buckets[min(bins - 1, int(row["score"] * bins))].append(row)
    results, ece = [], 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        confidence = sum(r["score"] for r in bucket) / len(bucket)
        frequency = sum(r["attack_present"] for r in bucket) / len(bucket)
        ece += len(bucket) / len(rows) * abs(confidence - frequency)
        results.append({"lower": index / bins, "upper": (index + 1) / bins,
                        "n": len(bucket), "mean_score": confidence, "attack_frequency": frequency})
    return {"brier": sum((r["score"] - r["attack_present"]) ** 2 for r in rows) / len(rows),
            "ece": ece, "bins": results}


def select_threshold(rows, max_clean_flag_rate=0.05):
    if not math.isfinite(max_clean_flag_rate) or not 0 <= max_clean_flag_rate <= 1:
        raise ValueError("max_clean_flag_rate must be between zero and one")
    validation = [r for r in rows if r["split"] == "validation"]
    if {r["attack_present"] for r in validation} != {False, True}:
        raise ValueError("validation requires clean and attack examples")
    # Include the first representable float above each score so tied observations
    # cannot be split into an unrealistically favourable decision rule.
    thresholds = {0.0, 1.0}
    for row in validation:
        thresholds.add(row["score"])
        above = math.nextafter(row["score"], math.inf)
        if above <= 1:
            thresholds.add(above)
    candidates = []
    clean = [r for r in validation if not r["attack_present"]]
    attacked = [r for r in validation if r["attack_present"]]
    for threshold in thresholds:
        false_rate = sum(r["score"] >= threshold for r in clean) / len(clean)
        recall = sum(r["score"] >= threshold for r in attacked) / len(attacked)
        if false_rate <= max_clean_flag_rate:
            candidates.append((recall, -false_rate, threshold))
    if not candidates:
        raise ValueError("no threshold satisfies the validation clean-flag budget")
    recall, negative_false_rate, threshold = max(candidates)
    return {"threshold": threshold, "validation_recall": recall,
            "validation_clean_flag_rate": -negative_false_rate,
            "max_clean_flag_rate": max_clean_flag_rate, "validation_n": len(validation)}


def group_bootstrap_accuracy(rows, threshold, repeats=500, seed=0):
    if type(repeats) is not int or repeats < 20:
        raise ValueError("at least 20 bootstrap repeats required")
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    if len(groups) < 2:
        return None
    rng, samples = random.Random(seed), []
    names = sorted(groups)
    for _ in range(repeats):
        selected = [r for name in rng.choices(names, k=len(names)) for r in groups[name]]
        samples.append(sum((r["score"] >= threshold) == r["attack_present"] for r in selected) / len(selected))
    samples.sort()
    return {"method": "family-cluster percentile bootstrap", "repeats": repeats,
            "families": len(groups), "seed": seed,
            "accuracy_95_percent_interval": [samples[int(0.025 * (repeats - 1))], samples[int(0.975 * (repeats - 1))]]}


def study(rows, max_clean_flag_rate=0.05, repeats=500, seed=0):
    evaluate_rows(rows)  # Validate every split and reject family leakage first.
    selection = select_threshold(rows, max_clean_flag_rate)
    threshold = selection["threshold"]
    test = [r for r in rows if r["split"] == "test"]
    by_source = []
    for source, model in sorted({(r["source"], r["model"]) for r in test}):
        subset = [r for r in test if (r["source"], r["model"]) == (source, model)]
        by_source.append({"source": source, "model": model, "metrics": evaluate_rows(subset, threshold)})
    return {"selection": selection, "held_out": evaluate_rows(rows, threshold),
            "calibration": calibration(test),
            "uncertainty": group_bootstrap_accuracy(test, threshold, repeats, seed),
            "by_source_model": by_source,
            "score_interpretation": "Brier and ECE treat score as an attack probability; inspect whether the detector supports that interpretation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--max-clean-flag-rate", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    report = study(rows, args.max_clean_flag_rate, args.bootstrap, args.seed)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["held_out"], indent=2))


if __name__ == "__main__":
    main()
