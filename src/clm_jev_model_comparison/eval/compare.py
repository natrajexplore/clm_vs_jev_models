"""Collect results/runs/*/metrics.json into one markdown table (printed and written to results/summary.md).

    uv run python -m clm_jev_model_comparison.eval.compare
"""

import argparse
import json
from pathlib import Path

COLUMNS = ["task", "method", "n", "accuracy", "macro_f1", "ece", "brier", "nll",
           "labeled_examples_used", "latency_ms", "cost_usd", "run"]


def _row(path: Path) -> dict:
    m = json.loads(path.read_text())
    latency = m["latency"]["p50_ms"] if "latency" in m else m.get("ms_per_example")
    return {**m, "latency_ms": latency, "run": path.parent.name}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.0f}"
    return "" if v is None else str(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--out", default="results/summary.md")
    args = parser.parse_args()

    rows = sorted((_row(p) for p in Path(args.runs).glob("*/metrics.json")), key=lambda r: (r["task"], r["method"], r["run"]))
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in COLUMNS) + " |" for r in rows]
    note = ("\nlatency_ms: Jev = p50 per-request network latency; CLM = amortized local CPU embedding time per example. "
            "Not the same measurement; compare with care.\n")
    table = "\n".join(lines) + "\n" + note
    Path(args.out).write_text(table)
    print(table)


if __name__ == "__main__":
    main()
