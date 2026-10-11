"""## Executive summary (read this first)

Combine numeric patch descriptions and cited text scenarios into joint forecast paths.
MindTS supplies the endogenous/exogenous alignment and condensation design motivation.
The numerical engine is a CPU block bootstrap, not MindTS neural weights or a paper reproduction.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .cli import _series
from .horizons import HorizonMetadataError, monthly_horizon_steps
from .targets import log_return_steps

RETURN_TARGETS = frozenset({"log_return"})


@dataclass
class NumericState:
    """Numerical calibration and dated patch descriptions from one unit's input only."""

    assets: list[str]
    horizons: list[int]
    periods: NDArray[np.int64]
    offsets: NDArray[np.int64]
    anchors: NDArray[np.float64]
    scales: NDArray[np.float64]
    drifts: NDArray[np.float64]
    innovations: NDArray[np.float64]
    contiguous: NDArray[np.bool_]
    summary: dict[str, Any]


def _valid_steps(series: pd.Series, returns: bool) -> pd.Series:
    dates = pd.DatetimeIndex(series.index)
    gap = pd.Series(dates, index=series.index).diff().dt.total_seconds() / 86400
    typical = float(gap.dropna().median()) if gap.notna().any() else 1.0
    values = pd.Series(log_return_steps(series), index=series.index) if returns else series.diff()
    # Missing years in a transfer series are never interpreted as one daily change.
    return values.where(gap.isna() | (gap <= max(typical * 3, 5)))


def _patches(asset: str, series: pd.Series, steps: pd.Series) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    recent = series.iloc[-120:]
    for start in range(0, len(recent), 20):
        patch = recent.iloc[start : start + 20]
        local = steps.reindex(patch.index).dropna()
        if patch.empty:
            continue
        segments.append(
            {
                "patch_id": f"{asset}:p{len(segments)}",
                "start": str(patch.index[0])[:10],
                "end": str(patch.index[-1])[:10],
                "n": len(patch),
                "min": float(patch.min()),
                "max": float(patch.max()),
                "median": float(patch.median()),
                "step_sum": float(local.sum()),
                "step_volatility": float(local.std(ddof=0)) if len(local) else 0.0,
            }
        )
    return segments


def prepare(
    panels: dict[str, pd.DataFrame],
    assets: list[str],
    horizons: list[int],
    asof: str,
    target_type: str,
    *,
    card: dict[str, Any] | None = None,
    forecast_spec: dict[str, Any] | None = None,
) -> NumericState:
    """Calibrate per-step changes and horizon spacing without reading other units or outcomes."""
    dt.date.fromisoformat(asof)
    if not assets or len(set(assets)) != len(assets):
        raise ValueError("asset roster must be nonempty and unique")
    if not horizons or len(set(horizons)) != len(horizons) or min(horizons) < 1:
        raise ValueError("horizons must be unique positive business-day offsets")
    if max(horizons) > 2600:
        raise ValueError("horizon exceeds the bounded path engine")
    if target_type not in {"level", "yield", *RETURN_TARGETS}:
        raise ValueError(f"unsupported target_type: {target_type}")
    returns = target_type in RETURN_TARGETS
    histories: dict[str, pd.Series] = {}
    steps: dict[str, pd.Series] = {}
    spacings: list[float] = []
    for asset in assets:
        s = _series(panels, asset, asof)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index, errors="raise"))
        s = s.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if s.index.has_duplicates:
            raise ValueError(f"{asset}: duplicated target observations")
        if len(s) < 3:
            raise ValueError(f"{asset}: need at least three finite dated observations")
        spacing = float(np.median(np.diff(s.index.to_numpy()) / np.timedelta64(1, "D")))
        spacings.append(spacing)
        histories[asset] = s
        steps[asset] = _valid_steps(s, returns)
    if max(spacings) > 7 and min(spacings) <= 7:
        raise ValueError("mixed daily and lower-frequency target panels need explicit resampling")
    monthly = min(spacings) > 7
    offsets = np.zeros(len(assets), dtype=np.int64)
    declared = (card or {}).get("targets", {}).get("target_frequency")
    explicit = any(
        "observation_periods" in source.get("targets", {})
        or any("observation_period" in q for q in source.get("questions", []))
        for source in (card or {}, forecast_spec or {})
    )
    if monthly:
        if target_type != "level":
            raise HorizonMetadataError(
                "Monthly targets require levels and explicit observation periods"
            )
        last_dates = {}
        anchor_months = []
        for asset, history in histories.items():
            months = history.index.to_period("M")
            if months.has_duplicates or np.median(np.diff(months.asi8)) != 1:
                raise HorizonMetadataError("Target series must have unique monthly observations")
            last_dates[asset] = str(history.index[-1])[:10]
            anchor_months.append(int(months[-1].ordinal))
            # A missing observation month never becomes one monthly innovation.
            steps[asset] = history.diff().where(np.r_[False, np.diff(months.asi8) == 1])
        periods = monthly_horizon_steps(
            assets,
            horizons,
            last_dates,
            asof=asof,
            card=card,
            forecast_spec=forecast_spec,
        ).astype(np.int64)
        offsets = np.asarray(anchor_months, dtype=np.int64) - min(anchor_months)
    else:
        if max(spacings) > 3:
            raise ValueError("unsupported target cadence; expected business-daily or monthly")
        if explicit:
            raise HorizonMetadataError("Monthly observation periods conflict with daily targets")
        if declared == "monthly":
            warnings.warn(
                "Monthly declaration on dense daily targets; retaining daily sampling",
                UserWarning,
                stacklevel=2,
            )
        periods = np.tile(horizons, (len(assets), 1)).astype(np.int64)
    if int((periods + offsets[:, None]).max()) > 2600:
        raise ValueError("explicit target grid exceeds the bounded path engine")
    anchors: list[float] = []
    scales: list[float] = []
    drifts: list[float] = []
    descriptions: dict[str, Any] = {}
    for ai, asset in enumerate(assets):
        series = histories[asset]
        step = steps[asset].dropna().iloc[-1024:]
        if len(step) < 2:
            raise ValueError(f"{asset}: insufficient contiguous changes for scale calibration")
        recent = step.iloc[-60:]
        floor = max(1e-6, abs(float(series.iloc[-1])) * 1e-5 if not returns else 1e-6)
        variance = 0.65 * float(recent.var(ddof=0)) + 0.35 * float(step.var(ddof=0))
        scale = max(math.sqrt(max(variance, 0)), floor)
        anchor = 0.0 if returns else float(series.iloc[-1])
        # Shrink finite-sample return drift; a constant per-period return keeps its known mean.
        drift = float(step.mean()) if returns else 0.0
        if returns and float(step.std(ddof=0)) > 1e-12:
            drift *= len(step) / (len(step) + 256)
            bound = 0.5 * scale / math.sqrt(int(periods[ai].max()))
            drift = float(np.clip(drift, -bound, bound))
        anchors.append(anchor)
        scales.append(scale)
        drifts.append(drift)
        descriptions[asset] = {
            "last_observed": float(series.iloc[-1]),
            "last_date": str(series.index[-1])[:10],
            "history_start": str(series.index[0])[:10],
            "history_end": str(series.index[-1])[:10],
            "n_observations": len(series),
            "n_valid_steps": len(step),
            "anchor": anchor,
            "step_mean": float(step.mean()),
            "step_drift": drift,
            "step_sd": scale,
            "horizon_sd": scale * math.sqrt(int(periods[ai].max())),
            "patches": _patches(asset, series, steps[asset]),
        }
    joint_steps = {}
    for asset, step_series in steps.items():
        aligned = step_series.copy()
        if monthly:
            aligned.index = aligned.index.to_period("M").to_timestamp()
        joint_steps[asset] = aligned
    joint = pd.DataFrame(joint_steps).dropna().iloc[-2048:]
    raw = joint.to_numpy(dtype=np.float64)
    if len(raw) >= 2:
        std = raw.std(axis=0)
        shocks = (raw - raw.mean(axis=0)) / np.maximum(std, 1e-12)
    else:
        std = np.zeros(len(assets))
        shocks = np.empty((0, len(assets)))
    if len(joint) >= 2:
        gaps = np.diff(joint.index.to_numpy()) / np.timedelta64(1, "D")
        limit = max(float(np.median(gaps)) * 3, 5)
        continues = np.diff(joint.index.to_period("M").asi8) == 1 if monthly else gaps <= limit
        contiguous = np.concatenate([continues, [False]])
    else:
        contiguous = np.zeros(len(joint), dtype=np.bool_)
    if len(shocks) < 20 or np.any(std <= 1e-12):
        # Conservative correlated Student-t fallback for constant or sparse joint history.
        shocks = np.empty((0, len(assets)))
        contiguous = np.zeros(0, dtype=np.bool_)
    summary = {
        "asof": asof,
        "target_type": target_type,
        "horizon_keys": horizons,
        "simulated_periods": {a: periods[i].tolist() for i, a in enumerate(assets)},
        "sampling_frequency": "calendar_month" if monthly else "business_daily",
        "return_conversion": "log1p(simple_return)" if returns else None,
        "engine": "joint stationary block bootstrap" if len(shocks) else "joint Student-t floor",
        "joint_history_rows": len(joint),
        "assets": descriptions,
    }
    return NumericState(
        assets,
        horizons,
        periods,
        offsets,
        np.array(anchors),
        np.array(scales),
        np.array(drifts),
        shocks,
        contiguous,
        summary,
    )


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, float | int) or not math.isfinite(value):
        raise ValueError("model field must be a finite JSON number")
    return float(value)


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def condense(
    reply: dict[str, Any],
    state: NumericState,
    docs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate quotes and patch ownership; deduplicate and shrink text adjustments."""
    documents = {str(d["doc_id"]): d for d in docs}
    parsed_assets = reply.get("assets", {})
    accepted: dict[str, Any] = {}
    ledger: dict[str, Any] = {}
    if not isinstance(parsed_assets, dict):
        parsed_assets = {}
    for asset in state.assets:
        entry = parsed_assets.get(asset)
        rejected = 0
        evidence: dict[str, Any] = {}
        valid_patches = {p["patch_id"] for p in state.summary["assets"][asset]["patches"]}
        if not isinstance(entry, dict):
            ledger[asset] = {"applied": False, "reason": "no asset evidence returned"}
            continue
        items = entry.get("evidence", [])
        if not isinstance(items, list):
            items = []
        for item in items[:4]:
            try:
                doc = documents[item["doc_id"]]
                quote = item["quote"]
                if not isinstance(quote, str) or len(quote.strip()) < 12 or len(quote) > 500:
                    raise ValueError("missing bounded verbatim citation")
                if _normalized_text(quote) not in _normalized_text(doc["text"]):
                    raise ValueError("quote not present in supplied source")
                ids = item["patch_ids"]
                if not isinstance(ids, list) or not ids or not set(ids).issubset(valid_patches):
                    raise ValueError("wrong asset or nonexistent patch")
                relevance = _number(item["relevance"])
                confidence = _number(item["confidence"])
                if not 0.2 <= relevance <= 1 or not 0 < confidence <= 1:
                    raise ValueError("weak or invalid relevance/confidence")
                if item["kind"] not in {"established", "inferred"}:
                    raise ValueError("unknown evidence kind")
                age = (
                    dt.date.fromisoformat(state.summary["asof"])
                    - dt.date.fromisoformat(doc["timestamp"])
                ).days
                strength = relevance * confidence * math.exp(-max(age, 0) / 365)
                kept = {**item, "strength": strength, "timestamp": doc["timestamp"]}
                previous = evidence.get(item["doc_id"])
                if previous is None or strength > previous["strength"]:
                    evidence[item["doc_id"]] = kept
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected += 1
        if not evidence:
            ledger[asset] = {
                "applied": False,
                "reason": "no valid cited patch-aligned evidence",
                "rejected_evidence": rejected,
            }
            continue
        strength = min(0.8, max(e["strength"] for e in evidence.values()))
        scenarios: list[dict[str, Any]] = []
        entries = entry.get("scenarios", [])
        if not isinstance(entries, list):
            entries = []
        for scenario in entries[:4]:
            try:
                probability = _number(scenario["probability"])
                shift = _number(scenario["shift_sd"])
                vol = _number(scenario["vol_scale"])
                citations = scenario["doc_ids"]
                if (
                    not isinstance(citations, list)
                    or not citations
                    or not set(citations) <= evidence.keys()
                ):
                    raise ValueError("scenario lacks retained evidence")
                if not 0 < probability <= 1 or not 0 < vol <= 3:
                    raise ValueError("invalid probability or volatility")
                established = all(evidence[d]["kind"] == "established" for d in citations)
                vol = float(np.clip(vol, 0.8 if established else 1.0, 2.0))
                if entry.get("contradiction") is True:
                    vol = max(vol, 1.15)
                scenarios.append(
                    {
                        "name": str(scenario.get("name", "scenario"))[:80],
                        "probability": probability,
                        "shift_sd": float(np.clip(shift, -1, 1)) * strength,
                        "vol_scale": 1 + strength * (vol - 1),
                        "doc_ids": citations,
                    }
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected += 1
        if not scenarios:
            ledger[asset] = {
                "applied": False,
                "reason": "no valid cited scenarios",
                "rejected_evidence": rejected,
            }
            continue
        total = sum(s["probability"] for s in scenarios)
        for s in scenarios:
            s["probability"] /= total
        # Sort consistently: the same scenario quantile drives all asset paths.
        scenarios.sort(key=lambda s: s["shift_sd"])
        accepted[asset] = {"scenarios": scenarios, "strength": strength}
        ledger[asset] = {
            "applied": True,
            "strength": strength,
            "evidence": list(evidence.values()),
            "scenarios": scenarios,
            "rejected_evidence": rejected,
            "contradiction": entry.get("contradiction") is True,
        }
    return accepted, ledger


def sample(
    state: NumericState,
    n_draws: int,
    seed: int,
    signals: dict[str, Any] | None = None,
) -> NDArray[np.float64]:
    """Simulate one continuing joint path per draw, preserving horizon and asset dependence."""
    if n_draws < 200:
        raise ValueError("at least 200 joint draws are required")
    rng = np.random.default_rng(seed)
    a_count = len(state.assets)
    running = np.zeros((n_draws, a_count), dtype=np.float64)
    result = np.empty((n_draws, a_count, len(state.horizons)), dtype=np.float64)
    endpoints = state.periods + state.offsets[:, None]
    n_history = len(state.innovations)
    positions = rng.integers(0, max(n_history, 1), size=n_draws)
    for period in range(1, int(endpoints.max()) + 1):
        if n_history:
            restart = rng.random(n_draws) < 0.2
            restart |= ~state.contiguous[positions]
            positions = (positions + 1) % n_history
            count = int(restart.sum())
            # Blend recent regimes and full history, always taking whole asset vectors.
            full = rng.integers(0, n_history, size=count)
            recent = rng.integers(max(0, n_history - 60), n_history, size=count)
            positions[restart] = np.where(rng.random(count) < 0.6, recent, full)
            innovation = state.innovations[positions]
        else:
            common = rng.standard_normal((n_draws, 1))
            individual = rng.standard_normal((n_draws, a_count))
            t_scale = np.sqrt(5 / rng.chisquare(7, size=(n_draws, 1)))
            innovation = (0.6 * common + 0.8 * individual) * t_scale
        running += (innovation * state.scales + state.drifts) * (period > state.offsets)
        for ai, hi in np.argwhere(endpoints == period):
            result[:, ai, hi] = state.anchors[ai] + running[:, ai]
    if signals:
        quantiles = rng.random(n_draws)
        for ai, asset in enumerate(state.assets):
            if asset not in signals:
                continue
            scenarios = signals[asset]["scenarios"]
            cumulative = np.cumsum([s["probability"] for s in scenarios])
            choices = np.minimum(np.searchsorted(cumulative, quantiles), len(scenarios) - 1)
            shifts = np.array([s["shift_sd"] for s in scenarios])[choices]
            scales = np.array([s["vol_scale"] for s in scenarios])[choices]
            maximum = int(state.periods[ai].max())
            for hi, p in enumerate(state.periods[ai]):
                centre = state.anchors[ai] + state.drifts[ai] * p
                noise = result[:, ai, hi] - centre
                delta = shifts * state.scales[ai] * math.sqrt(maximum) * (p / maximum)
                result[:, ai, hi] = centre + noise * scales + delta
    if not np.isfinite(result).all():
        raise ValueError("nonfinite forecast; refusing to emit invalid output")
    return result
