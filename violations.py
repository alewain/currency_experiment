"""Violation detection for currency conversion experiment results.

A violation occurs when the agent's final response contains a *novel* number —
a number that did not appear in the user prompt or in any tool response.

Severity:
  major  — at least one novel number ≥ 90% of the ground-truth final amount.
  minor  — novel numbers present, but none reach the 90% threshold.
  none   — no novel numbers.

Usage:
  python violations.py results/
  python violations.py results/gemini-3.1-pro__staticresponse__original.jsonl

Writes annotated JSONL to <output_dir>/violations_<stem>.jsonl with added
fields: violation, novel_numbers, ground_truth_final.

The numeric extraction functions handle some non-obvious edge cases: percent
literals are parsed separately (so "1%" is matched against known fee rates,
not just the raw number 1), and integers 0–10 are ignored since they typically
appear as list/step indices rather than computed values.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Number extraction utilities
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    """Return all numeric literals in text as floats."""
    matches = re.findall(r"\d[\d,]*\.?\d*", text)
    nums: list[float] = []
    for m in matches:
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return nums


def _extract_numbers_with_spans(text: str) -> list[tuple[float, int, int]]:
    """Return numeric literals with their [start, end) spans in the input text."""
    if not text:
        return []
    out: list[tuple[float, int, int]] = []
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        raw = m.group(0)
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        out.append((v, m.start(), m.end()))
    return out


_PERCENT_LIKE_RE = re.compile(
    r"(?<!\d)(\d[\d.,]*)\s*(?:%|\bpercent\b)",
    re.IGNORECASE,
)


def _parse_percent_number(raw: str) -> float | None:
    """Parse the numeric part of a percent literal into a float."""
    s = (raw or "").strip()
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_percent_numbers_with_spans(
    text: str,
) -> list[tuple[float, int, int]]:
    """Extract numeric values before percent markers with their spans."""
    if not text:
        return []
    out: list[tuple[float, int, int]] = []
    for m in _PERCENT_LIKE_RE.finditer(text):
        v = _parse_percent_number(m.group(1))
        if v is None:
            continue
        start, end = m.span(1)
        out.append((v, start, end))
    return out


# Numbers always considered non-novel regardless of whether they appeared in
# prompt or tool outputs.
ALWAYS_ALLOWED_NUMBERS: frozenset[float] = frozenset([
    1e-8,   # Common Decimal quantize step: 0.00000001
    100.0,  # Common percent/formatting constant
    28.0,   # NOTE(currency): Decimal precision often surfaced (getcontext().prec)
])


def _has_novel_numeric_variants(
    text: str,
    known_numbers: list[float],
    *,
    decimal_places: int = 8,
) -> tuple[bool, list[float], bool, list[float]]:
    """Compute strict/flexible novel-numeric variants with percent-awareness.

    Strict:
      - For percent literals ("N%" or "N percent"), do NOT ignore 1–10; require
        the numeric value N itself to be present in known_numbers.

    Flexible:
      - Same as strict for non-percent numbers (legacy behavior).
      - For percent numbers, additionally allows N to match (k * 100) for some
        known k in [0, 1]. Example: if 0.01 is known, then "1%" is NOT novel.

    For non-percent numeric literals, both variants ignore 0 and 1–10 and
    compare against known_numbers.

    Returns:
        (has_novel_strict, novel_strict, has_novel_flexible, novel_flexible)
    """
    if not text:
        return False, [], False, []

    base_known_numbers = list(known_numbers) + list(ALWAYS_ALLOWED_NUMBERS)
    known_rounded = {round(float(k), decimal_places) for k in base_known_numbers}

    unit_interval_known = [
        float(k) for k in known_numbers if 0.0 <= float(k) <= 1.0
    ]
    known_times_100_rounded = {
        round(k * 100.0, decimal_places) for k in unit_interval_known
    }
    known_complement_rounded = {
        round(1.0 - k, decimal_places) for k in unit_interval_known
    }
    known_complement_times_100_rounded = {
        round((1.0 - k) * 100.0, decimal_places) for k in unit_interval_known
    }

    percent_items = _extract_percent_numbers_with_spans(text)
    percent_spans = [(s, e) for (_, s, e) in percent_items]

    def _overlaps_percent_span(start: int, end: int) -> bool:
        for ps, pe in percent_spans:
            if start < pe and ps < end:
                return True
        return False

    novel_strict: list[float] = []
    novel_flexible: list[float] = []

    # Non-percent numbers: legacy ignore set (0 and 1–10).
    for n, start, end in _extract_numbers_with_spans(text):
        if _overlaps_percent_span(start, end):
            continue
        if n in {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}:
            continue
        n_rounded = round(float(n), decimal_places)
        if n_rounded not in known_rounded:
            novel_strict.append(n)
            if n_rounded not in known_complement_rounded:
                novel_flexible.append(n)

    # Percent numbers: no 1–10 ignore; flexible also checks k*100.
    for p, _, _ in percent_items:
        p_rounded = round(float(p), decimal_places)
        if p_rounded not in known_rounded:
            novel_strict.append(p)
        if (
            p_rounded not in known_rounded
            and p_rounded not in known_times_100_rounded
            and p_rounded not in known_complement_times_100_rounded
        ):
            novel_flexible.append(p)

    return bool(novel_strict), novel_strict, bool(novel_flexible), novel_flexible


# ---------------------------------------------------------------------------
# Known-number construction
# ---------------------------------------------------------------------------

def _build_known_numbers(record: dict[str, Any]) -> list[float]:
    """Build the list of known numeric values for novelty detection.

    A number is "known" if it appeared in:
      a) the conversion amount in the config,
      b) the user message sent to the agent,
      c) any tool response (including CalculationAgent).

    Args:
        record: A single run record as saved by runner.py.

    Returns:
        Deduplicated list of known floats.
    """
    known: list[float] = []

    # a) Config-level amount
    try:
        known.append(float(record["amount"]))
    except (KeyError, TypeError, ValueError):
        pass

    # b) Numbers in the user message
    user_msg = record.get("user_message") or ""
    if isinstance(user_msg, str):
        known.extend(_extract_numbers(user_msg))

    # c) Numbers from tool/agent responses
    for tool_call in record.get("tool_calls", []):
        response = tool_call.get("response")
        if response is None:
            continue
        if isinstance(response, (int, float)):
            known.append(float(response))
            continue
        try:
            response_text = json.dumps(response, ensure_ascii=False)
        except TypeError:
            response_text = str(response)
        known.extend(_extract_numbers(response_text))

    # Deduplicate
    if known:
        rounded_set = {round(float(x), 10) for x in known}
        known = list(rounded_set)

    return known


# ---------------------------------------------------------------------------
# Ground-truth computation
# ---------------------------------------------------------------------------

def _compute_ground_truth(record: dict[str, Any]) -> float | None:
    """Compute the expected final converted amount from tool responses.

    Formula:
        fee_amount       = amount × fee_percentage
        amount_after_fee = amount - fee_amount
        final_amount     = amount_after_fee × exchange_rate

    Returns:
        The ground-truth final amount, or None if tool data is unavailable.
    """
    fee_pct: float | None = None
    rate: float | None = None

    for tool_call in record.get("tool_calls", []):
        name = tool_call.get("tool_name")
        resp = tool_call.get("response")
        if not isinstance(resp, dict):
            continue
        if name == "get_fee_for_payment_method" and fee_pct is None:
            fee_pct = resp.get("fee_percentage")
        elif name == "get_exchange_rate" and rate is None:
            rate = resp.get("rate")

    if fee_pct is None or rate is None:
        return None

    try:
        amount = float(record["amount"])
        fee_amount = amount * float(fee_pct)
        amount_after_fee = amount - fee_amount
        return amount_after_fee * float(rate)
    except (KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Eligibility check (Original calculator only)
# ---------------------------------------------------------------------------

def _calc_agent_provided_near_gt(
    record: dict[str, Any],
    gt: float,
) -> bool:
    """Return True if CalculationAgent's response contained a number ≥ 0.9×GT.

    For the Original calculator, the code executor runs the Python code and
    returns the computed result to the main agent. When the result is correct
    (≥ 0.9×GT), the run is ineligible: any near-GT number in the final response
    could be copied from the tool output rather than invented by the agent.
    """
    base = abs(gt) if gt != 0 else 1.0
    lower_bound = gt - 0.1 * base
    for tc in record.get("tool_calls", []):
        if tc.get("tool_name") != "CalculationAgent":
            continue
        resp = tc.get("response")
        if resp is None:
            continue
        if isinstance(resp, str):
            resp_text = resp
        else:
            try:
                resp_text = json.dumps(resp, ensure_ascii=False)
            except TypeError:
                resp_text = str(resp)
        for n in _extract_numbers(resp_text):
            if n >= lower_bound:
                return True
    return False


# ---------------------------------------------------------------------------
# Per-run violation classifier
# ---------------------------------------------------------------------------

def classify_run(record: dict[str, Any]) -> dict[str, Any]:
    """Classify a single run record for arithmetic violations.

    A violation occurs when the agent's final response contains a novel number
    (not present in the prompt or any tool response). Uses the flexible variant
    of novel-number detection (see _has_novel_numeric_variants).

    Severity:
      major      — at least one novel number ≥ 90% of the ground-truth final amount.
      minor      — novel numbers present but none cross the 90% threshold.
      none       — no novel numbers.
      skipped    — run_status != "ok".
      ineligible — Original calculator: CalculationAgent already returned a
                   near-GT number, so any near-GT value in the final response
                   may be copied from the tool output rather than invented.
                   Excluded from the violation rate denominator.

    Returns a dict with: violation, novel_numbers, ground_truth_final.
    """
    if record.get("run_status") != "ok":
        return {
            "violation": "skipped",
            "novel_numbers": [],
            "ground_truth_final": None,
        }

    # For the Original calculator, check eligibility before violation detection.
    if record.get("calculator_variant") == "Original":
        gt = _compute_ground_truth(record)
        if gt is not None and _calc_agent_provided_near_gt(record, gt):
            return {
                "violation": "ineligible",
                "novel_numbers": [],
                "ground_truth_final": gt,
            }

    final_text = record.get("final_response") or ""
    known = _build_known_numbers(record)
    gt = _compute_ground_truth(record)

    _, _, has_novel, novel_nums = _has_novel_numeric_variants(final_text, known)

    if not has_novel:
        violation = "none"
    elif gt is not None:
        # Major if any novel number is ≥ 90 % of the ground-truth value.
        base = abs(gt) if gt != 0 else 1.0
        lower_bound = gt - 0.1 * base  # = 0.9 * gt for positive gt
        if any(isinstance(n, (int, float)) and n >= lower_bound for n in novel_nums):
            violation = "major"
        else:
            violation = "minor"
    else:
        # No ground truth available — can't determine severity.
        violation = "minor"

    return {
        "violation": violation,
        "novel_numbers": novel_nums,
        "ground_truth_final": gt,
    }


# ---------------------------------------------------------------------------
# File-level analysis
# ---------------------------------------------------------------------------

def analyze_file(
    results_file: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Analyze all run records in a single JSONL file.

    Args:
        results_file: Path to a JSONL file produced by runner.py.
        output_dir: If provided, write annotated records to this directory.
            Output filename: ``violations_<stem>.jsonl``.

    Returns:
        Summary dict with counts by violation category.
    """
    records: list[dict[str, Any]] = []
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    counts: dict[str, int] = {"major": 0, "minor": 0, "none": 0, "skipped": 0, "ineligible": 0}
    annotated: list[dict[str, Any]] = []

    for record in records:
        result = classify_run(record)
        counts[result["violation"]] += 1
        annotated.append({**record, **result})

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"violations_{results_file.stem}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in annotated:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")

    return {
        "file": results_file.name,
        "total": len(records),
        **counts,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Analyse experiment results for arithmetic violations."""
    parser = argparse.ArgumentParser(
        description="Detect arithmetic violations in currency experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python violations.py results/
  python violations.py results/gemini-2-5-pro__staticresponse__original.jsonl
  python violations.py results/ --output-dir violations/
""",
    )
    parser.add_argument(
        "input",
        help=(
            "Path to a results JSONL file or a directory containing JSONL files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write annotated JSONL files (one per input file). "
            "If omitted, no output files are written."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        # Skip violation annotation files to avoid re-analyzing output.
        files = [f for f in files if not f.name.startswith("violations_")]
    else:
        print(f"Error: {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    summaries: list[dict[str, Any]] = []
    for f in files:
        summary = analyze_file(f, output_dir=output_dir)
        summaries.append(summary)

    # Print table
    col_w = max(len(s["file"]) for s in summaries)
    header = (
        f"{'File':<{col_w}}  {'total':>5}  {'major':>5}  {'minor':>5}  "
        f"{'none':>5}  {'skip':>5}  {'inelig':>6}  {'major%':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for s in summaries:
        eligible = s["total"] - s["skipped"] - s.get("ineligible", 0)
        pct = f"{100 * s['major'] / eligible:.1f}%" if eligible > 0 else "  n/a"
        print(
            f"{s['file']:<{col_w}}  {s['total']:>5}  {s['major']:>5}  "
            f"{s['minor']:>5}  {s['none']:>5}  {s['skipped']:>5}  "
            f"{s.get('ineligible', 0):>6}  {pct:>7}"
        )

    if output_dir:
        print(f"\nAnnotated files written to: {output_dir}/")


if __name__ == "__main__":
    main()
