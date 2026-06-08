"""TAINT command-line interface.

Provides ``taint eval`` (and ``taint gen``) so a reviewer can clone, install, and
run the offline benchmark end to end:

    pip install -e . && pytest && taint eval

All commands are fully offline and deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from taint.dataset import build_benchmark, save_traces
from taint.eval import evaluate, format_table, reports_to_dict
from taint.harness import Harness
from taint.policy import CostModel, ElevationPolicy


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the full evaluation and print the metrics table."""
    dataset = build_benchmark(n=args.n, test_frac=args.test_frac, seed=args.seed)
    policy = ElevationPolicy(
        flag_threshold=args.flag_threshold, block_threshold=args.block_threshold
    )
    reports = evaluate(
        dataset=dataset, threshold=args.threshold, policy=policy, cost=CostModel()
    )

    bal = dataset.class_balance("test")
    print(
        f"TAINT offline benchmark  |  train={len(dataset.train)} "
        f"test={len(dataset.test)}  "
        f"(followed={bal.get(True, 0)} clean={bal.get(False, 0)})\n"
    )
    print(format_table(reports))
    print(
        "\nNote: metrics are from the OFFLINE deterministic mock agent, not a "
        "real LLM. See README safety note."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_size": len(dataset.train),
            "test_size": len(dataset.test),
            "test_class_balance": {str(k): v for k, v in bal.items()},
            "threshold": args.threshold,
            "policy": {
                "flag_threshold": args.flag_threshold,
                "block_threshold": args.block_threshold,
            },
            "reports": reports_to_dict(reports),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote results to {out}")
    return 0


def _cmd_gen(args: argparse.Namespace) -> int:
    """Generate raw traces and save them to a JSONL file."""
    harness = Harness()
    traces = harness.generate(n=args.n)
    path = save_traces(traces, args.out)
    followed = sum(1 for t in traces if t.injection_followed)
    print(
        f"Generated {len(traces)} traces (followed={followed} "
        f"clean={len(traces) - followed}) -> {path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``taint`` CLI."""
    parser = argparse.ArgumentParser(
        prog="taint",
        description="TAINT — prompt-injection provenance detection (offline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="evaluate all detectors on the benchmark")
    p_eval.add_argument("--n", type=int, default=200, help="number of traces")
    p_eval.add_argument("--test-frac", type=float, default=0.4, dest="test_frac")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--threshold", type=float, default=0.5)
    p_eval.add_argument("--flag-threshold", type=float, default=0.4, dest="flag_threshold")
    p_eval.add_argument("--block-threshold", type=float, default=0.7, dest="block_threshold")
    p_eval.add_argument("--out", type=str, default="results/metrics.json")
    p_eval.set_defaults(func=_cmd_eval)

    p_gen = sub.add_parser("gen", help="generate raw labeled traces to JSONL")
    p_gen.add_argument("--n", type=int, default=200)
    p_gen.add_argument("--out", type=str, default="data/traces/traces.jsonl")
    p_gen.set_defaults(func=_cmd_gen)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
