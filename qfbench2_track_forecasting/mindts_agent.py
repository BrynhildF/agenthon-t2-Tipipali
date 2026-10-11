"""## Executive summary (read this first)

Produce the three Track 2 artifacts using joint numerical paths and cited house-LLM scenarios.
This implements a MindTS-inspired API adaptation, not the original neural anomaly detector.
The organizer supplies the model endpoint, model name and proxy environment at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import tomllib
from typing import Any

import jsonschema
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qfbench2_common.taskcard import schema_path

from .cli import _read_panels
from .cutoff import trusted_asof
from .limits import ParseLimits
from .mindts_core import NumericState, condense, prepare, sample
from .mindts_endpoint import ModelReply, build_prompt, call_model, read_corpus


def _safe_text(value: Any) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _rationale(
    state: NumericState,
    draws: NDArray[np.float64],
    meta: dict[str, Any],
    ledger: dict[str, Any],
) -> str:
    lines = [
        "## Executive summary (read this first)",
        "",
        "These are joint probability forecasts, not anomaly scores. The numerical engine is a",
        "stationary block bootstrap (resampling consecutive whole cross-asset change vectors),",
        "with cited external-text scenarios when a valid house-model answer is available.",
        "This is a MindTS-inspired API adaptation; no original MindTS neural weights are used.",
        "",
        f"Unit: {_safe_text(meta['unit_id'])}. As-of: {meta['asof']}.",
        f"Target: {meta['target']}. Draws: {meta['n_draws']}. Seed: {meta['seed']}.",
        "",
        "## Numeric calibration",
        "",
        "Only this unit's dated panel observations at or before the as-of are used. Level targets",
        "start from the last observed value; cumulative return targets start from zero and sum",
        "log(1+r) from decimal simple returns. Changes across large date gaps are removed.",
        "Scale blends recent and full-history variance. Lower-frequency series use observed-period",
        "steps; output horizons remain business-day offsets. Horizon values on each draw share",
        "one continuing path; all assets resample the same historical change vector together.",
        "Block restart probability is 0.2; restarts blend recent 60 rows (0.6) and all rows (0.4).",
        "Sparse/constant joint histories use a correlated Student-t variance floor instead.",
        "",
        f"Engine: {state.summary['engine']}. Joint rows: {state.summary['joint_history_rows']}.",
        "",
        "| Asset | Last date | Anchor | Step drift | Step SD | Valid changes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ai, asset in enumerate(state.assets):
        info = state.summary["assets"][asset]
        lines.append(
            f"| {_safe_text(asset)} | {info['last_date']} | {state.anchors[ai]:.8g} | "
            f"{state.drifts[ai]:.8g} | {state.scales[ai]:.8g} | {info['n_valid_steps']} |"
        )
    lines += [
        "",
        "## External evidence and adjustments",
        "",
        f"Reasoning applied: {meta['reasoning_applied']}.",
        f"Model status: {_safe_text(meta['model_status'])}.",
        f"Admitted documents: {meta['text_documents']}. Reader counts: {meta['text_filter']}.",
        "",
        "Quotes below are verified against supplied text; they are not model memories.",
        "Confidence and causal interpretations remain model hypotheses, not verified facts.",
        "Evidence is deduplicated and shrunk by relevance, confidence and age. Inferred",
        "evidence cannot narrow volatility; contradictory evidence widens it. A shared",
        "scenario quantile couples all assets, with ordering by directional shift.",
    ]
    for asset in state.assets:
        item = ledger.get(asset, {})
        lines += ["", f"### {_safe_text(asset)}", ""]
        if not item.get("applied"):
            lines.append(
                "No text adjustment: " + _safe_text(item.get("reason", "no accepted evidence"))
            )
            continue
        for evidence in item["evidence"]:
            lines.append(
                f"- Source {_safe_text(evidence['doc_id'])}, dated {evidence['timestamp']}; "
                f"patches {_safe_text(evidence['patch_ids'])}; kind {evidence['kind']}; "
                f"quote: {_safe_text(evidence['quote'])}. "
                f"Interpretation: {_safe_text(evidence.get('because', 'not supplied'))}."
            )
        lines += [
            "",
            "| Scenario | Probability | Effective shift (SD) | Vol multiplier | Sources |",
            "|---|---:|---:|---:|---|",
        ]
        for scenario in item["scenarios"]:
            lines.append(
                f"| {_safe_text(scenario['name'])} | {scenario['probability']:.6g} | "
                f"{scenario['shift_sd']:.6g} | {scenario['vol_scale']:.6g} | "
                f"{_safe_text(scenario['doc_ids'])} |"
            )
    lines += [
        "",
        "## Additive adjustment ledger and emitted distribution",
        "",
        "Text delta = effective scenario shift x step SD x sqrt(longest simulated period)",
        "x (period / longest period). Volatility multiplies noise about anchor + drift.",
        "The text-delta column is the scenario-weighted mean. Bootstrap noise need not have",
        "exactly zero sample mean, so the empirical median is also reported separately.",
        "",
        "| Asset | Business-day horizon | Simulated periods | Anchor | Drift | "
        "Mean text delta | P05 | P50 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ai, asset in enumerate(state.assets):
        maximum = int(state.periods[ai].max())
        scenarios = ledger.get(asset, {}).get("scenarios", [])
        mean_shift = sum(s["probability"] * s["shift_sd"] for s in scenarios)
        for hi, (horizon, period) in enumerate(zip(state.horizons, state.periods[ai], strict=True)):
            delta = mean_shift * state.scales[ai] * math.sqrt(maximum) * period / maximum
            q = np.quantile(draws[:, ai, hi], [0.05, 0.5, 0.95])
            lines.append(
                f"| {_safe_text(asset)} | {horizon} | {period} | {state.anchors[ai]:.8g} | "
                f"{state.drifts[ai] * period:.8g} | {delta:.8g} | "
                f"{q[0]:.8g} | {q[1]:.8g} | {q[2]:.8g} |"
            )
    lines += [
        "",
        "No realized outcomes, other units or external internet sources are used.",
        "Model training cutoff must be declared truthfully in submission.json; this agent",
        "does not invent that declaration. This rationale is required but is never scored.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MindTS-inspired Track 2 forecasting adapter")
    parser.add_argument("--panels", required=True, type=pathlib.Path)
    parser.add_argument("--text", required=True, type=pathlib.Path)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--card", type=pathlib.Path)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline", action="store_true", help="never call the model")
    modes.add_argument(
        "--require-model",
        action="store_true",
        help="fail instead of falling back if no cited model scenarios are accepted",
    )
    args = parser.parse_args(argv)
    candidates = [args.panels / "card.toml", args.panels.parent / "card.toml"]
    card_path = args.card or next((p for p in candidates if p.is_file()), None)
    if card_path is None:
        parser.error("cannot locate card.toml beside the panels or in their parent")
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    if args.asof != trusted_asof(card):
        parser.error("--asof must equal the trusted card's as-of")
    if args.out.name != "forecast.parquet":
        parser.error("--out must name forecast.parquet")
    try:
        seed = args.seed if args.seed is not None else int(os.environ.get("QFBENCH_SEED", "2026"))
        if seed < 0:
            raise ValueError
    except ValueError:
        parser.error("seed must be a nonnegative integer")
    targets = card["targets"]
    assets = list(targets["asset_ids"])
    horizons = [int(h) for h in targets["horizons"]]
    target_type = targets.get("target_type", "level")
    limits = ParseLimits()
    card_floor = int(card.get("scoring", {}).get("params", {}).get("n_draws_min", 0) or 0)
    n_draws = max(args.n_draws, card_floor, limits.min_draws)
    if n_draws > limits.max_draws or n_draws * len(assets) * len(horizons) > limits.max_rows:
        parser.error("requested draw grid exceeds the official output limits")
    spec_path = card_path.parent / "forecast_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else None
    if spec is not None and not isinstance(spec, dict):
        parser.error("forecast_spec.json must be an object")
    state = prepare(
        _read_panels(args.panels),
        assets,
        horizons,
        args.asof,
        target_type,
        card=card,
        forecast_spec=spec,
    )
    docs, counts = read_corpus(args.text, args.asof)
    if args.offline:
        reply = ModelReply(reason="--offline; no API call made")
    elif not docs:
        reply = ModelReply(reason="no admissible dated documents; no API call made")
    else:
        reply = call_model(build_prompt(state.summary, docs), seed=seed)
    signals, ledger = condense(reply.data or {}, state, docs)
    if args.require_model and not signals:
        parser.error(
            "--require-model: no accepted cited scenarios; " + (reply.reason or "invalid evidence")
        )
    draws = sample(state, n_draws, seed, signals)
    meta: dict[str, Any] = {
        "unit_id": card["task"]["id"],
        "asof": args.asof,
        "target": target_type,
        "representation": "samples",
        "asset_ids": assets,
        "horizons": horizons,
        "n_draws": n_draws,
        "seed": seed,
        "method_variant": "mindts-inspired-api-forecasting",
        "original_model_reproduced": False,
        "reasoning_applied": bool(signals),
        "model": reply.model or None,
        "model_usage": reply.usage,
        "model_status": reply.reason
        or ("accepted cited scenarios" if signals else "no valid scenarios"),
        "text_documents": len(docs),
        "text_filter": counts,
        "numeric_summary": state.summary,
        "evidence_ledger": ledger,
        "rationale": {"file": "forecast_rationale.md", "method": "cited joint path scenarios"},
    }
    schema = json.loads(schema_path("forecast.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(meta, schema)
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    rationale = _rationale(state, draws, meta, ledger)
    if len(meta_json.encode("utf-8")) > limits.max_meta_bytes:
        parser.error("metadata exceeds the official byte budget")
    if len(rationale.encode("utf-8")) > limits.max_rationale_bytes:
        parser.error("rationale exceeds the official byte budget")
    frame = pd.DataFrame(
        {
            "draw": np.repeat(np.arange(n_draws, dtype=np.int32), len(assets) * len(horizons)),
            "asset": np.tile(np.repeat(assets, len(horizons)), n_draws),
            "horizon": np.tile(np.asarray(horizons, dtype=np.int32), n_draws * len(assets)),
            "value": draws.reshape(-1),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    (args.out.parent / "forecast_meta.json").write_text(meta_json, encoding="utf-8")
    (args.out.parent / "forecast_rationale.md").write_text(rationale, encoding="utf-8")
    print(f"Wrote {n_draws} joint draws to {args.out}; reasoning_applied={bool(signals)}")
    if not signals:
        print("Numerical-only fallback: " + meta["model_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
