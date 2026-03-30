"""Stacked bar chart: violation rate by model.

Reproduces the cross-model comparison figure. Each bar shows what percentage
of runs resulted in a major violation (dark red), minor violation (lighter
red), or no violation (not shown).

Input
-----
Annotated JSONL files produced by ``violations.py``. One file per model
configuration (the filename encodes model, calculator, and prompt variant).

Usage
-----
    python plot_model_rates.py violations/ --output figures/model_rates.png

    # Filter to a specific calculator variant (default: StaticResponse):
    python plot_model_rates.py violations/ --calculator StaticResponse

    # Include all calculator variants in the plot:
    python plot_model_rates.py violations/ --calculator all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Chart colors.
_COLOR_MAJOR = "#8B0000"   # dark red  = major violation
_COLOR_MINOR = "#FF6B6B"   # light red = minor violation only

DEFAULT_CALCULATOR = "Original"
DEFAULT_PROMPT = "original"
MIN_RUNS_PER_BAR = 5  # bars with fewer runs are excluded


def _load_violations(
    violations_dir: Path,
    *,
    calculator: str | None = None,
    prompt: str | None = None,
    max_eligible: int | None = 20,
) -> dict[str, list[str]]:
    """Load violation labels grouped by model_name.

    Records are processed in run_index order. For each model, only the first
    max_eligible eligible runs (i.e. violation not in skipped/ineligible) are
    kept; subsequent eligible runs are discarded. Skipped and ineligible runs
    that arrive before the cap is reached are included so the caller can see
    the full picture up to that point, but they never count toward the cap.

    Args:
        violations_dir: Directory containing violations_*.jsonl files.
        calculator: Filter to this calculator variant value, or None/``"all"``.
        prompt: Filter to this prompt variant value, or None/``"all"``.
        max_eligible: Maximum eligible runs to include per model (default: 20).
            Pass None to include all eligible runs.

    Returns:
        Dict mapping model_name -> list of violation strings.
    """
    by_model: dict[str, list[str]] = defaultdict(list)
    eligible_counts: dict[str, int] = defaultdict(int)

    files = sorted(violations_dir.glob("*.jsonl"))
    files = [f for f in files if not f.name.startswith("judge_")]

    for f in files:
        records = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Fail fast on raw runner outputs: plots require violation labels.
        missing_violation = next(
            (rec for rec in records if "violation" not in rec),
            None,
        )
        if missing_violation is not None:
            raise ValueError(
                "Expected JSONL files annotated by violations.py. "
                f"Missing 'violation' field in {f.name} "
                f"(run_index={missing_violation.get('run_index')}). "
                "Run violations.py first."
            )

        # Process in run_index order so the cap applies to the first N eligible.
        records.sort(key=lambda r: r.get("run_index", 0))

        for rec in records:
            if calculator and calculator != "all":
                if rec.get("calculator_variant") != calculator:
                    continue
            if prompt and prompt != "all":
                if rec.get("prompt_variant") != prompt:
                    continue

            model = rec.get("model_name", "unknown")
            violation = rec.get("violation", "none")
            is_eligible = violation not in ("skipped", "ineligible")

            if is_eligible and max_eligible is not None:
                if eligible_counts[model] >= max_eligible:
                    continue
                eligible_counts[model] += 1

            by_model[model].append(violation)

    return dict(by_model)


def _compute_rows(
    by_model: dict[str, list[str]],
    *,
    min_runs: int = MIN_RUNS_PER_BAR,
) -> list[dict[str, Any]]:
    """Compute per-model aggregates for plotting.

    Returns:
        List of dicts with keys: model, total, major, minor_only, major_pct,
        minor_only_pct. Sorted by major_pct descending.
    """
    rows = []
    for model, violations in by_model.items():
        eligible = [v for v in violations if v not in ("skipped", "ineligible")]
        total = len(eligible)
        if total < min_runs:
            continue
        n_major = sum(1 for v in eligible if v == "major")
        n_minor = sum(1 for v in eligible if v == "minor")
        rows.append({
            "model": model,
            "total": total,
            "major": n_major,
            "minor_only": n_minor,
            "major_pct": 100.0 * n_major / total,
            "minor_only_pct": 100.0 * n_minor / total,
        })
    rows.sort(key=lambda r: -r["major_pct"])
    return rows


def plot_model_rates(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Violation rate by model",
    subtitle: str | None = None,
) -> None:
    """Save the stacked bar chart as a PNG.

    Args:
        rows: List of per-model aggregates from _compute_rows().
        output_path: Destination path for the PNG file.
        title: Main chart title.
        subtitle: Optional subtitle shown above the bars.
    """
    if not rows:
        print("No data to plot.", file=sys.stderr)
        return

    labels = [r["model"] for r in rows]
    major_pct = [r["major_pct"] for r in rows]
    minor_pct = [r["minor_only_pct"] for r in rows]
    totals = [r["total"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
    xs = range(len(labels))

    ax.bar(xs, major_pct, color=_COLOR_MAJOR)
    ax.bar(xs, minor_pct, bottom=major_pct, color=_COLOR_MINOR)

    # Annotate each bar with n= (total runs)
    for i, (mp, mn, n) in enumerate(zip(major_pct, minor_pct, totals)):
        top = mp + mn
        ax.text(
            i, top + 1.5,
            f"n={n}",
            ha="center", va="bottom", fontsize=8, color="#555555",
        )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("% of runs", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis="y")

    if subtitle:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        ax.text(
            0.5, 1.0, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=11, fontweight="normal",
        )
    else:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    legend_handles = [
        mpatches.Patch(color=_COLOR_MAJOR, label="Major violation"),
        mpatches.Patch(color=_COLOR_MINOR, label="Minor violation"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=9,
        frameon=True,
    )
    plt.subplots_adjust(right=0.78)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot violation rate by model (stacked bar chart).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_model_rates.py violations/
  python plot_model_rates.py violations/ --output figures/fig1_model_rates.png
  python plot_model_rates.py violations/ --calculator all --prompt all
""",
    )
    parser.add_argument(
        "violations_dir",
        help="Directory containing violations_*.jsonl files (from violations.py).",
    )
    parser.add_argument(
        "--output",
        default="figures/model_rates.png",
        help="Output PNG path (default: figures/model_rates.png).",
    )
    parser.add_argument(
        "--calculator",
        default=DEFAULT_CALCULATOR,
        help=(
            f"Filter to this calculator variant (default: {DEFAULT_CALCULATOR}). "
            "Pass 'all' to include all calculators."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=(
            f"Filter to this prompt variant (default: {DEFAULT_PROMPT}). "
            "Pass 'all' to include all prompt variants."
        ),
    )
    parser.add_argument(
        "--title",
        default="Violation rate by model",
        help="Chart title.",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Optional subtitle (e.g. '(original prompt, StaticResponse calculator)').",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=MIN_RUNS_PER_BAR,
        help=f"Minimum runs per model to include in chart (default: {MIN_RUNS_PER_BAR}).",
    )
    parser.add_argument(
        "--max-eligible",
        type=int,
        default=20,
        help=(
            "Maximum eligible runs per model (default: 20). "
            "Pass 0 to include all eligible runs."
        ),
    )
    args = parser.parse_args()

    violations_dir = Path(args.violations_dir)
    if not violations_dir.is_dir():
        print(f"Error: {violations_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    calculator = args.calculator if args.calculator != "all" else None
    prompt = args.prompt if args.prompt != "all" else None
    max_eligible = args.max_eligible if args.max_eligible > 0 else None

    by_model = _load_violations(
        violations_dir, calculator=calculator, prompt=prompt,
        max_eligible=max_eligible,
    )
    if not by_model:
        print("No matching records found.", file=sys.stderr)
        sys.exit(1)

    rows = _compute_rows(by_model, min_runs=args.min_runs)
    if not rows:
        print(
            f"No models with at least {args.min_runs} runs found.", file=sys.stderr
        )
        sys.exit(1)

    # Auto-subtitle if not provided and filtering is active
    subtitle = args.subtitle
    if subtitle is None and (calculator or prompt):
        parts = []
        if calculator:
            parts.append(f"{calculator} calculator")
        if prompt:
            parts.append(f"{prompt} prompt")
        subtitle = f"({', '.join(parts)})"

    plot_model_rates(
        rows,
        output_path=Path(args.output),
        title=args.title,
        subtitle=subtitle,
    )

    # Print a summary table
    print()
    print(f"{'Model':<40}  {'n':>4}  {'major%':>7}  {'minor%':>7}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['model']:<40}  {r['total']:>4}  "
            f"{r['major_pct']:>6.1f}%  {r['minor_only_pct']:>6.1f}%"
        )


if __name__ == "__main__":
    main()
