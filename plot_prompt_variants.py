"""Stacked bar charts: violation rate by prompt variant.

Reproduces the prompt-variant comparison figures. Three chart modes are
supported:

  single     — one stacked bar per prompt variant, for a fixed calculator.
  combined   — grouped bars: two bars per prompt variant, one per calculator,
               displayed side by side (StaticResponse = violet, CodeOnly = orange).

Usage
-----
    # Single chart for StaticResponse calculator:
    python plot_prompt_variants.py violations/ --calculator StaticResponse

    # Single chart for CodeOnly calculator:
    python plot_prompt_variants.py violations/ --calculator CodeOnly

    # Combined chart (both calculators):
    python plot_prompt_variants.py violations/ --mode combined
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
_COLOR_MAJOR = "#8B0000"   # dark red  = major violation (single-calculator charts)
_COLOR_MINOR = "#FF6B6B"   # light red = minor violation only

# Combined chart: StaticResponse = violet, CodeOnly = orange
_COLOR_STATIC_MAJOR = "#4B0082"   # dark violet (indigo)
_COLOR_STATIC_MINOR = "#B695E6"   # light violet
_COLOR_CODE_MAJOR   = "#CC5500"   # dark orange
_COLOR_CODE_MINOR   = "#FFB380"   # light orange

DEFAULT_MODEL = "gemini-3.1-pro-preview"
MIN_RUNS_PER_BAR = 5


def _load_violations_by_prompt(
    violations_dir: Path,
    *,
    model: str | None = None,
    calculator: str | None = None,
) -> dict[str, list[str]]:
    """Load violation labels grouped by prompt_variant.

    Args:
        violations_dir: Directory containing violations_*.jsonl files.
        model: Filter to this model_name, or None to include all.
        calculator: Filter to this calculator_variant, or None to include all.

    Returns:
        Dict mapping prompt_variant -> list of violation strings.
    """
    by_prompt: dict[str, list[str]] = defaultdict(list)

    for f in sorted(violations_dir.glob("*.jsonl")):
        if f.name.startswith("judge_"):
            continue
        records: list[dict[str, Any]] = []
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

        for rec in records:
            if model and rec.get("model_name") != model:
                continue
            if calculator and rec.get("calculator_variant") != calculator:
                continue
            prompt = rec.get("prompt_variant", "unknown")
            violation = rec.get("violation", "none")
            by_prompt[prompt].append(violation)

    return dict(by_prompt)


# Display order for the eight prompt variants.
_PROMPT_ORDER = [
    "original",
    "no_error",
    "general",
    "reverse",
    "general_and_reverse",
    "general_beginning",
    "general_end",
    "broad_end",
]


def _compute_rows(
    by_prompt: dict[str, list[str]],
    *,
    min_runs: int = MIN_RUNS_PER_BAR,
) -> list[dict[str, Any]]:
    """Compute per-prompt aggregates in canonical order.

    Returns:
        List of dicts with keys: prompt, total, major, minor_only,
        major_pct, minor_only_pct. Order follows _PROMPT_ORDER.
    """
    # Start with canonical order, append unknown variants at the end.
    ordered_keys = [k for k in _PROMPT_ORDER if k in by_prompt]
    extra_keys = [k for k in by_prompt if k not in _PROMPT_ORDER]
    all_keys = ordered_keys + extra_keys

    rows = []
    for prompt in all_keys:
        eligible = [v for v in by_prompt[prompt] if v not in ("skipped", "ineligible")]
        total = len(eligible)
        if total < min_runs:
            continue
        n_major = sum(1 for v in eligible if v == "major")
        n_minor = sum(1 for v in eligible if v == "minor")
        rows.append({
            "prompt": prompt,
            "total": total,
            "major": n_major,
            "minor_only": n_minor,
            "major_pct": 100.0 * n_major / total,
            "minor_only_pct": 100.0 * n_minor / total,
        })
    return rows


def _plot_single(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Violation rate by prompt variant",
    subtitle: str | None = None,
) -> None:
    """Stacked bar chart: one bar per prompt variant."""
    if not rows:
        print("No data to plot.", file=sys.stderr)
        return

    labels = [r["prompt"] for r in rows]
    major_pct = [r["major_pct"] for r in rows]
    minor_pct = [r["minor_only_pct"] for r in rows]
    totals = [r["total"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
    xs = range(len(labels))

    ax.bar(xs, major_pct, color=_COLOR_MAJOR)
    ax.bar(xs, minor_pct, bottom=major_pct, color=_COLOR_MINOR)

    for i, (mp, mn, n) in enumerate(zip(major_pct, minor_pct, totals)):
        top = mp + mn
        ax.text(
            i, top + 1.5, f"n={n}",
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


def _plot_combined(
    rows_static: list[dict[str, Any]],
    rows_code: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Violation rate by prompt variant",
    subtitle: str | None = None,
) -> None:
    """Grouped stacked bars: two bars per prompt variant (one per calculator).

    StaticResponse = violet (dark = major, light = minor).
    CodeOnly       = orange (dark = major, light = minor).
    """
    # Build aligned pairs (same prompt, present in both calculators).
    static_by_prompt = {r["prompt"]: r for r in rows_static}
    code_by_prompt   = {r["prompt"]: r for r in rows_code}
    common = [k for k in _PROMPT_ORDER if k in static_by_prompt and k in code_by_prompt]
    extra  = [k for k in static_by_prompt if k not in _PROMPT_ORDER and k in code_by_prompt]
    prompts = common + extra

    if not prompts:
        print("No prompt variants common to both calculators.", file=sys.stderr)
        return

    n = len(prompts)
    x = list(range(n))
    width = 0.38
    off = 0.22

    static_major = [static_by_prompt[p]["major_pct"] for p in prompts]
    static_minor = [static_by_prompt[p]["minor_only_pct"] for p in prompts]
    code_major   = [code_by_prompt[p]["major_pct"] for p in prompts]
    code_minor   = [code_by_prompt[p]["minor_only_pct"] for p in prompts]

    x_static = [xi - off for xi in x]
    x_code   = [xi + off for xi in x]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 5))
    ax.bar(x_static, static_major, width=width, color=_COLOR_STATIC_MAJOR)
    ax.bar(x_static, static_minor, width=width, bottom=static_major, color=_COLOR_STATIC_MINOR)
    ax.bar(x_code,   code_major,   width=width, color=_COLOR_CODE_MAJOR)
    ax.bar(x_code,   code_minor,   width=width, bottom=code_major,   color=_COLOR_CODE_MINOR)

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=35, ha="right", fontsize=10)
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

    ax.legend(
        handles=[
            mpatches.Patch(color=_COLOR_STATIC_MAJOR, label="StaticResponse — major violation"),
            mpatches.Patch(color=_COLOR_STATIC_MINOR, label="StaticResponse — minor violation"),
            mpatches.Patch(color=_COLOR_CODE_MAJOR,   label="CodeOnly — major violation"),
            mpatches.Patch(color=_COLOR_CODE_MINOR,   label="CodeOnly — minor violation"),
        ],
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
        description="Plot violation rate by prompt variant (stacked bar chart).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_prompt_variants.py violations/ --calculator StaticResponse
  python plot_prompt_variants.py violations/ --calculator CodeOnly
  python plot_prompt_variants.py violations/ --mode combined --output figures/fig2_prompt_variants.png
""",
    )
    parser.add_argument(
        "violations_dir",
        help="Directory containing violations_*.jsonl files (from violations.py).",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "combined"],
        default="single",
        help=(
            "'single': one bar per prompt variant for --calculator (default). "
            "'combined': grouped bars for both calculators side by side."
        ),
    )
    parser.add_argument(
        "--calculator",
        default="StaticResponse",
        help=(
            "Calculator variant to plot in 'single' mode "
            "(default: StaticResponse). Ignored in 'combined' mode."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Filter to this model_name. If omitted, all models are aggregated "
            "(useful when you ran only one model)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path. Default: figures/prompt_variants_<calculator>.png "
            "for single mode, figures/prompt_variants_combined.png for combined."
        ),
    )
    parser.add_argument(
        "--title",
        default="Violation rate by prompt variant",
        help="Chart title.",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Optional subtitle.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=MIN_RUNS_PER_BAR,
        help=f"Minimum runs per bar to include (default: {MIN_RUNS_PER_BAR}).",
    )
    args = parser.parse_args()

    violations_dir = Path(args.violations_dir)
    if not violations_dir.is_dir():
        print(f"Error: {violations_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    model = args.model

    if args.mode == "combined":
        output = Path(args.output) if args.output else Path("figures/prompt_variants_combined.png")
        static_by_prompt = _load_violations_by_prompt(
            violations_dir, model=model, calculator="StaticResponse"
        )
        code_by_prompt = _load_violations_by_prompt(
            violations_dir, model=model, calculator="CodeOnly"
        )
        rows_static = _compute_rows(static_by_prompt, min_runs=args.min_runs)
        rows_code   = _compute_rows(code_by_prompt,   min_runs=args.min_runs)
        subtitle = args.subtitle
        if subtitle is None and model:
            subtitle = f"(Gemini 3 Pro)" if "gemini-3" in model.lower() else f"({model})"
        _plot_combined(
            rows_static, rows_code, output,
            title=args.title, subtitle=subtitle,
        )
    else:
        calculator = args.calculator
        output = Path(args.output) if args.output else Path(
            f"figures/prompt_variants_{calculator.lower()}.png"
        )
        by_prompt = _load_violations_by_prompt(
            violations_dir, model=model, calculator=calculator
        )
        rows = _compute_rows(by_prompt, min_runs=args.min_runs)
        subtitle = args.subtitle
        if subtitle is None:
            parts = [f"{calculator} calculator"]
            if model:
                parts.insert(0, model)
            subtitle = f"({', '.join(parts)})"
        _plot_single(rows, output, title=args.title, subtitle=subtitle)

        # Print summary
        print()
        print(f"{'Prompt variant':<25}  {'n':>4}  {'major%':>7}  {'minor%':>7}")
        print("-" * 50)
        for r in rows:
            print(
                f"{r['prompt']:<25}  {r['total']:>4}  "
                f"{r['major_pct']:>6.1f}%  {r['minor_only_pct']:>6.1f}%"
            )


if __name__ == "__main__":
    main()
