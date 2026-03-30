"""Experiment runner for the currency conversion experiment.

The main entry points are run_single() and run_batch(). Results are saved
incrementally as JSONL (one record per line). Each config gets a deterministic
filename derived from its model/calculator/prompt values, so resume across
interrupted runs is automatic.

Small changes to the event extraction or outcome classification logic can
affect which runs are classified as failures and how tool calls are attributed,
so these functions should be treated carefully.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.genai import types

from agents import (
    CalculatorVariant,
    PromptVariant,
    build_currency_agent,
    DEFAULT_USER_MESSAGE,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Defines a single experimental condition.

    Attributes:
        model_name: Model identifier passed to build_currency_agent().
        prompt_variant: Which main-agent system prompt to use.
        calculator_variant: Which CalculationAgent variant to use.
        n_runs: Number of repetitions (default: 20).
        seed_base: Base seed; run i uses seed_base + i (default: 999).
        max_calls_per_run: Safety limit on LLM calls per run (default: 30).
        amount: Conversion amount in base_currency (default: 1250.0).
        base_currency: ISO 4217 source currency code (default: "USD").
        target_currency: ISO 4217 target currency code (default: "INR").
        payment_method: Payment method string (default: "Bank Transfer").
    """
    model_name: str
    prompt_variant: PromptVariant
    calculator_variant: CalculatorVariant
    n_runs: int = 20
    seed_base: int = 999
    max_calls_per_run: int = 30
    amount: float = 1250.0
    base_currency: str = "USD"
    target_currency: str = "INR"
    payment_method: str = "Bank Transfer"


# ---------------------------------------------------------------------------
# ADK plugins
# ---------------------------------------------------------------------------

class CallLimitPlugin(BasePlugin):
    """Limits LLM calls for the main agent to prevent infinite loops."""

    def __init__(self, max_calls: int = 30, agent_name: str = "currency_agent"):
        super().__init__(name="call_limit")
        self.max_calls = max_calls
        # NOTE: in the original tutorial notebook this agent is called
        # "enhanced_currency_agent". See agents.py.
        self.agent_name = agent_name
        self.call_count = 0

    async def before_model_callback(self, callback_context=None, **kwargs):
        if getattr(callback_context, "agent_name", None) == self.agent_name:
            self.call_count += 1
            if self.call_count > self.max_calls:
                raise RuntimeError(
                    f"Call limit reached: {self.call_count} > {self.max_calls}"
                )
        return None

    async def before_tool_callback(self, callback_context=None, **kwargs):
        return None


class ToolErrorToStatusPlugin(BasePlugin):
    """Converts tool errors into structured error responses.

    Prevents tool errors from aborting the entire run. Instead, the agent
    receives a tool response dict with status="error", which the prompt's
    error-handling logic can process normally.
    """

    def __init__(self):
        super().__init__(name="tool_error_to_status_plugin")

    async def on_tool_error_callback(
        self,
        *,
        tool,
        tool_args,
        tool_context,
        error,
        **kwargs,
    ):
        tool_name = getattr(tool, "name", None)
        return {
            "status": "error",
            "error_message": "Tool not found",
            "tool_name": tool_name,
        }


# ---------------------------------------------------------------------------
# Event extraction functions
# Work on live ADK Event objects (attribute access, not dict .get()).
# Handle parallel tool calls, multi-part final responses, and the distinction
# between thinking parts (thought=True) and visible text.
# ---------------------------------------------------------------------------

def _extract_tool_calls_from_events(events) -> list[dict]:
    """Extract tool call information from events.

    Matches each function_response with its function_call by id,
    to support multiple parallel tool calls in the same turn.

    Args:
        events: List of Event objects from runner.

    Returns:
        List of dicts with keys: tool_name, parameters, response.
    """
    tool_calls: list[dict] = []
    call_index_by_id: dict[str, int] = {}

    for event in events:
        content = getattr(event, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue

        for part in parts:
            # Function call
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                call_id = getattr(fc, "id", None)
                idx = len(tool_calls)
                tool_calls.append({
                    "tool_name": getattr(fc, "name", None),
                    "parameters": dict(getattr(fc, "args", {})),
                    "response": None,
                })
                if call_id is not None:
                    call_index_by_id[call_id] = idx

            # Function response
            elif hasattr(part, "function_response") and part.function_response:
                fr = part.function_response
                resp_id = getattr(fr, "id", None)
                response_obj = getattr(fr, "response", None)

                if resp_id is not None and resp_id in call_index_by_id:
                    # Associate with the correct call by id
                    tool_calls[call_index_by_id[resp_id]]["response"] = response_obj
                elif tool_calls:
                    # Fallback for legacy cases without id
                    tool_calls[-1]["response"] = response_obj

    return tool_calls


def _extract_thinking_from_events(events) -> list[dict]:
    """Extract thinking steps (thought=True parts) from events.

    Args:
        events: List of Event objects from runner.

    Returns:
        List of dicts with keys: event_index, author, text.
        Empty list if no thinking steps found (e.g. non-Gemini models).
    """
    thinking_steps = []

    for i, event in enumerate(events):
        if not hasattr(event, "content") or not event.content:
            continue
        content = event.content
        if not hasattr(content, "parts") or not content.parts:
            continue

        for part in content.parts:
            if (
                hasattr(part, "thought") and part.thought
                and hasattr(part, "text") and part.text
            ):
                thinking_steps.append({
                    "event_index": i,
                    "author": getattr(event, "author", None),
                    "text": part.text,
                })

    return thinking_steps


def _extract_final_response_from_events(events) -> str:
    """Return the final user-visible answer text, skipping thinking parts.

    Searches backwards through events to find the last event that contains
    any text parts, then returns only the non-thinking visible parts from
    that event. If the last event with text has only thinking parts
    (thought=True), returns empty string without searching earlier events.

    Multiple visible parts within the chosen event are joined with newlines
    to preserve answers that span multiple parts (e.g. prose + code block).

    Returns:
        The final user-visible text, or empty string if none found.
    """
    if not events:
        return ""

    for event in reversed(events):
        if not hasattr(event, "content") or not event.content:
            continue
        content = event.content
        if not hasattr(content, "parts") or not content.parts:
            continue

        text_parts = [
            part for part in content.parts
            if hasattr(part, "text") and part.text
        ]
        if not text_parts:
            continue

        non_thinking_texts = [
            part.text for part in text_parts
            if not (hasattr(part, "thought") and part.thought)
        ]
        if non_thinking_texts:
            return "\n".join(non_thinking_texts)

        # Last event with text has only thought=True parts: stop here.
        return ""

    return ""


# ---------------------------------------------------------------------------
# Run outcome classification
# ---------------------------------------------------------------------------

def _derive_run_outcome(
    final_response: str,
    tool_calls: list[dict],
    events,
    *,
    call_limit_reached: bool = False,
) -> tuple[str, str | None]:
    """Return (run_status, run_failure_reason) for a completed run.

    Args:
        final_response: The extracted final answer text (may be empty).
        tool_calls: Extracted tool calls for the run.
        events: Raw Event objects from the runner.
        call_limit_reached: Whether the run was stopped by CallLimitPlugin.

    Returns:
        Tuple of (run_status, run_failure_reason) where run_status is
        "ok" or "failed", and run_failure_reason is None when status is "ok".
    """
    has_tool_calls = bool(tool_calls)

    if not events:
        return "failed", "empty_events"
    if call_limit_reached:
        return "failed", "call_limit_reached"
    if final_response != "":
        return "ok", None

    # No final response — classify by finish reason
    finish_reason = None
    if events:
        last_event = events[-1]
        if hasattr(last_event, "finish_reason"):
            finish_reason = last_event.finish_reason

    if finish_reason == types.FinishReason.MAX_TOKENS:
        reason = "finish_max_tokens"
    elif finish_reason == types.FinishReason.SAFETY:
        reason = "finish_safety"
    elif finish_reason == types.FinishReason.RECITATION:
        reason = "finish_recitation"
    elif finish_reason == types.FinishReason.OTHER:
        reason = "finish_other"
    else:
        reason = "stop_no_tool_no_text" if not has_tool_calls else "stop_tool_only"

    return "failed", reason


# ---------------------------------------------------------------------------
# Filename and path utilities
# ---------------------------------------------------------------------------

def _slug(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    value = value.lower()
    allowed = []
    for ch in value:
        if ch.isalnum() or ch in {".", "-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-")


def _results_path(output_dir: Path, cfg: ExperimentConfig) -> Path:
    """Return the JSONL path for a given config (deterministic, no timestamp)."""
    model_slug = _slug(cfg.model_name)
    calc_slug = _slug(cfg.calculator_variant.value)
    prompt_slug = _slug(cfg.prompt_variant.value)
    return output_dir / f"{model_slug}__{calc_slug}__{prompt_slug}.jsonl"


def _config_path(output_dir: Path, cfg: ExperimentConfig) -> Path:
    """Return the config JSON path for a given config."""
    results = _results_path(output_dir, cfg)
    return results.parent / (results.stem + "_config.json")


def _load_completed_indices(results_file: Path) -> set[int]:
    """Return the set of run indices already saved in a results file."""
    completed: set[int] = set()
    if not results_file.exists():
        return completed
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "run_index" in record:
                    completed.add(record["run_index"])
            except json.JSONDecodeError:
                pass
    return completed


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

async def run_single(
    cfg: ExperimentConfig,
    run_index: int,
    user_message: str = DEFAULT_USER_MESSAGE,
) -> dict[str, Any]:
    """Execute one run of the experiment and return a result record.

    Args:
        cfg: Experiment configuration.
        run_index: Logical index for this run (used as seed offset and label).
        user_message: The message sent to the agent (default: the standard
            currency conversion request from the tutorial notebook).

    Returns:
        Dict with all run information ready to be serialised to JSONL.
    """
    seed = cfg.seed_base + run_index

    # Build a fresh agent for this run (seed is baked into generate_content_config)
    agent = build_currency_agent(
        model_name=cfg.model_name,
        prompt_variant=cfg.prompt_variant,
        calculator_variant=cfg.calculator_variant,
        seed=seed,
    )

    call_limit_plugin = CallLimitPlugin(
        max_calls=cfg.max_calls_per_run,
        agent_name="currency_agent",
    )
    tool_error_plugin = ToolErrorToStatusPlugin()

    runner = InMemoryRunner(
        agent=agent,
        plugins=[call_limit_plugin, tool_error_plugin],
    )

    call_limit_reached = False
    try:
        events = await runner.run_debug(user_message, verbose=False)
    except RuntimeError as e:
        if "Call limit reached" in str(e):
            call_limit_reached = True
            print(f"  [run {run_index}] Call limit reached: {e}")
            events = []
        else:
            raise

    # Extract structured information from the event stream
    tool_calls     = _extract_tool_calls_from_events(events)
    final_response = _extract_final_response_from_events(events)
    thinking       = _extract_thinking_from_events(events)

    run_status, run_failure_reason = _derive_run_outcome(
        final_response=final_response,
        tool_calls=tool_calls,
        events=events,
        call_limit_reached=call_limit_reached,
    )

    return {
        "run_index":          run_index,
        "seed":               seed,
        "model_name":         cfg.model_name,
        "prompt_variant":     cfg.prompt_variant.value,
        "calculator_variant": cfg.calculator_variant.value,
        "amount":             cfg.amount,
        "base_currency":      cfg.base_currency,
        "target_currency":    cfg.target_currency,
        "payment_method":     cfg.payment_method,
        "user_message":       user_message,
        "tool_calls":         tool_calls,
        "final_response":     final_response,
        "thinking":           thinking,
        "run_status":         run_status,
        "run_failure_reason": run_failure_reason,
        "total_calls":        call_limit_plugin.call_count,
    }


# ---------------------------------------------------------------------------
# Batch runner with resume
# ---------------------------------------------------------------------------

async def run_batch(
    configs: list[ExperimentConfig],
    output_dir: str | Path = "results",
    resume: bool = True,
) -> None:
    """Run multiple experiment configurations, saving results incrementally.

    For each config, results are appended to a JSONL file whose name is
    derived deterministically from the config (no timestamps). If resume=True
    and the file already exists, run indices already present in the file are
    skipped automatically.

    Args:
        configs: List of ExperimentConfig instances to run.
        output_dir: Directory where JSONL and config files are written.
        resume: If True (default), skip indices already saved on disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(configs)
    for exp_i, cfg in enumerate(configs, 1):
        results_file = _results_path(output_dir, cfg)
        config_file  = _config_path(output_dir, cfg)

        print(f"\n{'='*70}")
        print(
            f"[{exp_i}/{total}] {cfg.model_name} | "
            f"{cfg.calculator_variant.value} | {cfg.prompt_variant.value}"
        )
        print(f"  Output: {results_file}")
        print(f"{'='*70}")

        # Determine which indices still need to be run
        if not resume and results_file.exists():
            results_file.unlink()
        completed = _load_completed_indices(results_file) if resume else set()
        pending = [i for i in range(cfg.n_runs) if i not in completed]

        if not pending:
            print(f"  All {cfg.n_runs} runs already complete. Skipping.")
            continue
        if completed:
            print(
                f"  Resuming: {len(completed)} done, "
                f"{len(pending)} remaining."
            )

        # Write config file (overwrite — config is fixed per file)
        config_file.write_text(
            json.dumps(
                {**asdict(cfg),
                 "prompt_variant":     cfg.prompt_variant.value,
                 "calculator_variant": cfg.calculator_variant.value,
                 "results_file":       str(results_file),
                 "created_at":         datetime.now().isoformat()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Run pending indices sequentially, saving after each
        for run_index in pending:
            print(f"  Run {run_index + 1}/{cfg.n_runs} (index {run_index}) ...",
                  end=" ", flush=True)
            try:
                record = await run_single(cfg, run_index)
            except Exception as e:
                print(f"ERROR: {e}")
                record = {
                    "run_index":          run_index,
                    "model_name":         cfg.model_name,
                    "prompt_variant":     cfg.prompt_variant.value,
                    "calculator_variant": cfg.calculator_variant.value,
                    "run_status":         "failed",
                    "run_failure_reason": f"exception: {e}",
                }

            with open(results_file, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

            status = record.get("run_status", "?")
            reason = record.get("run_failure_reason", "")
            suffix = f" ({reason})" if reason else ""
            print(f"{status}{suffix}")

        ok_count = sum(
            1 for r in _load_completed_indices(results_file)
            if r is not None
        )
        print(f"  Done. {results_file.name}")


# ---------------------------------------------------------------------------
# Predefined batch configurations (ready to run)
# ---------------------------------------------------------------------------

def cross_model_configs(
    models: list[str],
    *,
    calculator_variant: CalculatorVariant = CalculatorVariant.ORIGINAL,
    prompt_variant: PromptVariant = PromptVariant.ORIGINAL,
    n_runs: int = 20,
) -> list[ExperimentConfig]:
    """Configs for comparing violation rates across multiple models.

    Uses a single calculator and prompt variant (defaults: Original +
    original), varying only the model. The Original calculator matches
    the setup used in Section 2.4 of the post.

    Args:
        models: List of model name strings.
        calculator_variant: Calculator to use (default: Original).
        prompt_variant: Prompt variant (default: original).
        n_runs: Runs per model (default: 20).
    """
    return [
        ExperimentConfig(
            model_name=m,
            prompt_variant=prompt_variant,
            calculator_variant=calculator_variant,
            n_runs=n_runs,
        )
        for m in models
    ]


def prompt_variant_configs(
    model_name: str,
    *,
    calculator_variants: list[CalculatorVariant] | None = None,
    prompt_variants: list[PromptVariant] | None = None,
    n_runs: int = 20,
) -> list[ExperimentConfig]:
    """Configs for comparing violation rates across prompt and calculator variants.

    Produces the cartesian product of calculator_variants × prompt_variants
    for a single model. This is the Appendix B setup, so the default
    calculator variants are StaticResponse and CodeOnly (not Original).

    Args:
        model_name: Model to test.
        calculator_variants: Calculators to include
            (default: StaticResponse and CodeOnly).
        prompt_variants: Prompts to include (default: all eight variants).
        n_runs: Runs per condition (default: 20).
    """
    if calculator_variants is None:
        calculator_variants = [
            CalculatorVariant.STATIC_RESPONSE,
            CalculatorVariant.CODE_ONLY,
        ]
    if prompt_variants is None:
        prompt_variants = list(PromptVariant)

    return [
        ExperimentConfig(
            model_name=model_name,
            prompt_variant=pv,
            calculator_variant=cv,
            n_runs=n_runs,
        )
        for cv, pv in product(calculator_variants, prompt_variants)
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run currency conversion experiments."
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro-preview",
        help="Model name (default: gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=20,
        help="Runs per configuration (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for JSONL files (default: results/).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing results and re-run all indices.",
    )
    parser.add_argument(
        "--cross-model",
        nargs="+",
        metavar="MODEL",
        help=(
            "Run cross-model comparison with the given model names. "
            "Uses Original calculator and original prompt (Section 2.4 setup)."
        ),
    )
    parser.add_argument(
        "--prompt-variants",
        action="store_true",
        help=(
            "Run all prompt × calculator variants for --model. "
            "Produces the prompt-variant comparison."
        ),
    )
    args = parser.parse_args()

    if args.cross_model:
        configs = cross_model_configs(args.cross_model, n_runs=args.n_runs)
    elif args.prompt_variants:
        configs = prompt_variant_configs(args.model, n_runs=args.n_runs)
    else:
        # Default: single config, single model, Original calculator, original prompt
        configs = [
            ExperimentConfig(
                model_name=args.model,
                prompt_variant=PromptVariant.ORIGINAL,
                calculator_variant=CalculatorVariant.ORIGINAL,
                n_runs=args.n_runs,
            )
        ]

    asyncio.run(
        run_batch(
            configs,
            output_dir=args.output_dir,
            resume=not args.no_resume,
        )
    )
