"""## Executive summary (read this first)

Exercise the MindTS-inspired submission using synthetic series and a mocked house endpoint.
No reference values from any competition unit are used.
"""

from __future__ import annotations

import json
import pathlib
import tomllib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import write_card

from qfbench2_track_forecasting import mindts_agent, mindts_core, mindts_endpoint


def panels(*, returns: bool = False, monthly: bool = False) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(41)
    dates = pd.date_range(end="2020-06-30", periods=160, freq="ME" if monthly else "B")
    steps = rng.normal(0, 0.01, len(dates))
    return {
        "synthetic": pd.DataFrame(
            {
                "date": list(dates) * 2,
                "asset": ["SYN-A"] * len(dates) + ["SYN-B"] * len(dates),
                "value": np.concatenate(
                    [
                        steps if returns else 2 + steps.cumsum(),
                        steps * 2 if returns else -1 + (steps * 2).cumsum(),
                    ]
                ),
            }
        )
    }


def evidence_reply(summary: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "assets": {
            asset: {
                "evidence": [
                    {
                        "doc_id": docs[0]["doc_id"],
                        "quote": "Policy uncertainty remains elevated",
                        "patch_ids": [summary["assets"][asset]["patches"][-1]["patch_id"]],
                        "relevance": 0.9,
                        "confidence": 0.8,
                        "kind": "inferred",
                        "because": "A dated policy statement supports a modest upside scenario.",
                    }
                ],
                "contradiction": False,
                "scenarios": [
                    {
                        "name": "neutral",
                        "probability": 0.5,
                        "shift_sd": 0.0,
                        "vol_scale": 1.0,
                        "doc_ids": [docs[0]["doc_id"]],
                    },
                    {
                        "name": "upside",
                        "probability": 0.5,
                        "shift_sd": 0.7,
                        "vol_scale": 1.2,
                        "doc_ids": [docs[0]["doc_id"]],
                    },
                ],
            }
            for asset in summary["assets"]
        }
    }


def test_numerical_paths_preserve_asset_and_horizon_dependence() -> None:
    state = mindts_core.prepare(panels(), ["SYN-B", "SYN-A"], [21, 1, 5], "2020-06-30", "level")
    a = mindts_core.sample(state, 2000, 7)
    b = mindts_core.sample(state, 2000, 7)
    assert a.shape == (2000, 2, 3)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
    assert np.corrcoef(a[:, 0, 0], a[:, 1, 0])[0, 1] > 0.95
    assert np.corrcoef(a[:, 1, 0], a[:, 1, 2])[0, 1] > 0.25


def test_return_target_sums_log1p_returns_instead_of_differencing_or_adding_spot() -> None:
    source = panels(returns=True)
    source["synthetic"]["value"] = 0.01
    state = mindts_core.prepare(source, ["SYN-A"], [1, 5], "2020-06-30", "log_return")
    draws = mindts_core.sample(state, 2000, 7)
    assert state.anchors[0] == 0
    assert state.summary["assets"]["SYN-A"]["step_mean"] == pytest.approx(np.log1p(0.01))
    assert draws[:, 0, 1].mean() == pytest.approx(5 * np.log1p(0.01), abs=1e-6)


def test_monthly_observations_are_not_treated_as_daily_steps() -> None:
    card = {
        "targets": {
            "target_frequency": "monthly",
            "horizons": [21, 42],
            "observation_periods": ["2020-09", "2020-10"],
        }
    }
    state = mindts_core.prepare(
        panels(monthly=True), ["SYN-A"], [21, 42], "2020-06-30", "level", card=card
    )
    np.testing.assert_array_equal(state.periods, [[3, 4]])


def test_cutoff_filter_makes_future_rows_irrelevant() -> None:
    source = panels()
    late = pd.DataFrame({"date": ["2020-12-31"], "asset": ["SYN-A"], "value": [1e9]})
    extended = {"synthetic": pd.concat([source["synthetic"], late], ignore_index=True)}
    a = mindts_core.prepare(source, ["SYN-A"], [5], "2020-06-30", "level")
    b = mindts_core.prepare(extended, ["SYN-A"], [5], "2020-06-30", "level")
    np.testing.assert_array_equal(mindts_core.sample(a, 500, 1), mindts_core.sample(b, 500, 1))


def test_transfer_gap_does_not_become_a_daily_shock() -> None:
    source = panels()
    frame = source["synthetic"]
    frame = frame[(frame["asset"] != "SYN-A") | (frame["date"] < "2020-03-01")].copy()
    source["synthetic"] = pd.concat(
        [frame, pd.DataFrame({"date": ["2020-06-30"], "asset": ["SYN-A"], "value": [100.0]})],
        ignore_index=True,
    )
    state = mindts_core.prepare(source, ["SYN-A"], [5], "2020-06-30", "level")
    assert state.scales[0] < 0.1
    assert all(p["step_volatility"] < 0.1 for p in state.summary["assets"]["SYN-A"]["patches"])


def test_corpus_rejects_future_malformed_and_escaping_paths(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "text"
    root.mkdir()
    (root / "ok.txt").write_text("Policy uncertainty remains elevated", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("must not read", encoding="utf-8")
    (root / "corpus_index.json").write_text(
        json.dumps(
            {
                "documents": [
                    {"doc_id": "ok", "timestamp": "2020-06-01", "file": "ok.txt"},
                    {"doc_id": "late", "timestamp": "2020-07-01", "file": "ok.txt"},
                    {"doc_id": "invalid", "timestamp": "2019-nonsense", "file": "ok.txt"},
                    {"doc_id": "escape", "timestamp": "2020-06-01", "file": "../outside.txt"},
                ]
            }
        ),
        encoding="utf-8",
    )
    docs, counts = mindts_endpoint.read_corpus(root, "2020-06-30")
    assert [d["doc_id"] for d in docs] == ["ok"]
    assert counts["excluded"] == 3


def test_condensation_requires_real_quote_and_same_asset_patch() -> None:
    state = mindts_core.prepare(panels(), ["SYN-A", "SYN-B"], [5], "2020-06-30", "level")
    docs = [
        {
            "doc_id": "policy",
            "timestamp": "2020-06-15",
            "text": "Policy uncertainty remains elevated",
        }
    ]
    reply = evidence_reply(state.summary, docs)
    accepted, ledger = mindts_core.condense(reply, state, docs)
    assert set(accepted) == {"SYN-A", "SYN-B"}
    assert ledger["SYN-A"]["strength"] > 0
    reply["assets"]["SYN-A"]["evidence"][0]["quote"] = "invented statement"
    reply["assets"]["SYN-B"]["evidence"][0]["patch_ids"] = ["SYN-A:p0"]
    accepted, _ = mindts_core.condense(reply, state, docs)
    assert not accepted


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "garbage"])
def test_nonfinite_or_nonnumeric_model_numbers_are_rejected(bad: Any) -> None:
    state = mindts_core.prepare(panels(), ["SYN-A"], [5], "2020-06-30", "level")
    docs = [
        {
            "doc_id": "policy",
            "timestamp": "2020-06-15",
            "text": "Policy uncertainty remains elevated",
        }
    ]
    reply = evidence_reply(state.summary, docs)
    for scenario in reply["assets"]["SYN-A"]["scenarios"]:
        scenario["shift_sd"] = bad
    accepted, _ = mindts_core.condense(reply, state, docs)
    assert not accepted


def test_valid_text_changes_draws_and_inferred_evidence_cannot_narrow() -> None:
    state = mindts_core.prepare(panels(), ["SYN-A"], [5, 21], "2020-06-30", "level")
    docs = [
        {
            "doc_id": "policy",
            "timestamp": "2020-06-15",
            "text": "Policy uncertainty remains elevated",
        }
    ]
    reply = evidence_reply(state.summary, docs)
    reply["assets"]["SYN-A"]["scenarios"][0]["vol_scale"] = 0.2
    accepted, _ = mindts_core.condense(reply, state, docs)
    assert min(s["vol_scale"] for s in accepted["SYN-A"]["scenarios"]) >= 1
    plain = mindts_core.sample(state, 2000, 9)
    text = mindts_core.sample(state, 2000, 9, accepted)
    assert text[:, 0, 1].mean() > plain[:, 0, 1].mean()


def test_offline_mode_never_calls_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QFBENCH_NETWORK", "none")
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "pinned-house-model")
    monkeypatch.setattr(
        mindts_endpoint.urllib.request,
        "build_opener",
        lambda *args: pytest.fail("offline run attempted HTTP"),
    )
    reply = mindts_endpoint.call_model("prompt", seed=4)
    assert reply.data is None
    assert "none" in reply.reason


def test_house_request_uses_injected_endpoint_and_disables_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, limit: int) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": '{"assets": {}}'}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 6},
                }
            ).encode()

    class Opener:
        def open(self, request: Any, timeout: int) -> Response:
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            captured["authorization"] = request.get_header("Authorization")
            return Response()

    monkeypatch.setenv("QFBENCH_NETWORK", "restricted")
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.invalid")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-unit-bearer")
    monkeypatch.setenv("MODEL_NAME", "pinned-house-model")
    monkeypatch.setattr(mindts_endpoint.urllib.request, "build_opener", lambda *args: Opener())
    reply = mindts_endpoint.call_model("prompt", seed=123)
    assert reply.data == {"assets": {}}
    assert captured["url"] == "https://house.invalid/v1/chat/completions"
    assert captured["body"]["model"] == "pinned-house-model"
    assert captured["body"]["tools"] == []
    assert captured["body"]["tool_choice"] == "none"
    assert captured["body"]["seed"] == 123
    assert captured["body"]["temperature"] == 0
    assert captured["authorization"] == "Bearer synthetic-unit-bearer"
    assert captured["body"]["max_tokens"] <= 4000


@pytest.mark.parametrize("with_model", [False, True])
def test_cli_writes_only_three_required_files_and_official_gates_pass(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, with_model: bool
) -> None:
    from qfbench2_track_forecasting.scoring import build_verifier

    unit = tmp_path / "unit"
    write_card(unit)
    panel_dir = unit / "panels"
    panel_dir.mkdir()
    panels()["synthetic"].to_parquet(panel_dir / "synthetic.parquet", index=False)
    text_dir = unit / "text"
    text_dir.mkdir()
    (text_dir / "doc.txt").write_text("Policy uncertainty remains elevated", encoding="utf-8")
    (text_dir / "corpus_index.json").write_text(
        json.dumps(
            {"documents": [{"doc_id": "policy", "timestamp": "2020-06-15", "file": "doc.txt"}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QFBENCH_NETWORK", "none")
    monkeypatch.setenv("QFBENCH_SEED", "91")
    out = tmp_path / "out"
    args = [
        "--panels",
        str(panel_dir),
        "--text",
        str(text_dir),
        "--asof",
        "2020-06-30",
        "--out",
        str(out / "forecast.parquet"),
        "--n-draws",
        "200",
    ]
    if with_model:

        def fake_call(prompt: str, *, seed: int) -> mindts_endpoint.ModelReply:
            summary_text, docs_text = prompt.split("NUMERICAL_SUMMARY:\n", 1)[1].split(
                "\nFROZEN_CORPUS:\n", 1
            )
            reply = evidence_reply(json.loads(summary_text), json.loads(docs_text))
            return mindts_endpoint.ModelReply(data=reply, model="pinned-house-model")

        monkeypatch.setattr(mindts_agent, "call_model", fake_call)
        args.append("--require-model")
    assert mindts_agent.main(args) == 0
    assert {f.name for f in out.iterdir()} == {
        "forecast.parquet",
        "forecast_meta.json",
        "forecast_rationale.md",
    }
    meta = json.loads((out / "forecast_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 91
    assert meta["reasoning_applied"] is with_model
    if with_model:
        assert meta["evidence_ledger"]["SYN-A"]["applied"]
        assert "Policy uncertainty remains elevated" in (out / "forecast_rationale.md").read_text(
            encoding="utf-8"
        )
    assert meta["method_variant"] == "mindts-inspired-api-forecasting"
    card = tomllib.loads((unit / "card.toml").read_text(encoding="utf-8"))
    ctx = {"card": card, "unit_dir": unit, "output_dir": out}
    verdict = build_verifier(ctx).run(ctx)
    assert verdict.admissible, verdict.detail


@pytest.mark.parametrize("asof,require_model", [("2020-06-29", False), ("2020-06-30", True)])
def test_cli_fails_before_writing_for_bad_cutoff_or_missing_required_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, asof: str, require_model: bool
) -> None:
    unit = tmp_path / "unit"
    write_card(unit)
    panel_dir = unit / "panels"
    panel_dir.mkdir()
    panels()["synthetic"].to_parquet(panel_dir / "synthetic.parquet", index=False)
    out = tmp_path / "out"
    args = [
        "--panels",
        str(panel_dir),
        "--text",
        str(unit / "text"),
        "--asof",
        asof,
        "--out",
        str(out / "forecast.parquet"),
    ]
    if require_model:
        args.append("--require-model")
    monkeypatch.setenv("QFBENCH_NETWORK", "none")
    with pytest.raises(SystemExit) as exc:
        mindts_agent.main(args)
    assert exc.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [False]},
        {"choices": [{"message": {"content": 123}}]},
        {"choices": [{"finish_reason": "length", "message": {"content": '{"assets": {}}'}}]},
        {"choices": [{"message": {"content": '{"assets": NaN}'}}]},
    ],
)
def test_bad_house_replies_fail_to_numerical_fallback(
    monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, limit: int) -> bytes:
            return json.dumps(payload).encode()

    class Opener:
        def open(self, request: Any, timeout: int) -> Response:
            return Response()

    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.invalid")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-unit-bearer")
    monkeypatch.setenv("MODEL_NAME", "pinned-house-model")
    monkeypatch.setenv("QFBENCH_NETWORK", "restricted")
    monkeypatch.setattr(mindts_endpoint.urllib.request, "build_opener", lambda *args: Opener())
    reply = mindts_endpoint.call_model("prompt", seed=1)
    assert reply.data is None
    assert reply.reason


@pytest.mark.parametrize("setting", ["missing-token", "endpoint-path", "token-budget"])
def test_invalid_house_settings_do_not_attempt_network(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    monkeypatch.setenv("QFBENCH_NETWORK", "restricted")
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.invalid")
    monkeypatch.setenv("MODEL_NAME", "house")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-unit-bearer")
    monkeypatch.setenv("MODEL_MAX_TOKENS", "4000")
    if setting == "missing-token":
        monkeypatch.delenv("MODEL_TOKEN")
    elif setting == "endpoint-path":
        monkeypatch.setenv("MODEL_ENDPOINT", "https://house.invalid/v1")
    else:
        monkeypatch.setenv("MODEL_MAX_TOKENS", "4001")
    monkeypatch.setattr(
        mindts_endpoint.urllib.request,
        "build_opener",
        lambda *args: pytest.fail("invalid configuration attempted HTTP"),
    )
    reply = mindts_endpoint.call_model("prompt", seed=1)
    assert reply.data is None
    assert "synthetic-unit-bearer" not in reply.reason


def test_monthly_release_lag_and_different_anchors_preserve_grid() -> None:
    source = panels(monthly=True)
    frame = source["synthetic"]
    source["synthetic"] = frame[(frame["asset"] != "SYN-B") | (frame["date"] < "2020-06-01")].copy()
    card = {
        "targets": {
            "target_frequency": "monthly",
            "horizons": [9, 37],
            "observation_periods": ["2020-09", "2020-10"],
        }
    }
    state = mindts_core.prepare(
        source, ["SYN-A", "SYN-B"], [9, 37], "2020-08-31", "level", card=card
    )
    np.testing.assert_array_equal(state.periods, [[3, 4], [4, 5]])
    np.testing.assert_array_equal(state.offsets, [1, 0])
    assert state.horizons == [9, 37]
    assert mindts_core.sample(state, 300, 2).shape == (300, 2, 2)


def test_monthly_joint_history_aligns_observation_month_not_date_style() -> None:
    source = panels(monthly=True)
    mask = source["synthetic"]["asset"] == "SYN-B"
    source["synthetic"].loc[mask, "date"] = (
        source["synthetic"].loc[mask, "date"].dt.to_period("M").dt.to_timestamp()
    )
    card = {
        "targets": {
            "target_frequency": "monthly",
            "horizons": [21],
            "observation_periods": ["2020-09"],
        }
    }
    state = mindts_core.prepare(source, ["SYN-A", "SYN-B"], [21], "2020-06-30", "level", card=card)
    assert state.summary["joint_history_rows"] == 159
    draws = mindts_core.sample(state, 2000, 2)
    assert np.corrcoef(draws[:, 0, 0], draws[:, 1, 0])[0, 1] > 0.95


@pytest.mark.parametrize("conflict", [False, True])
def test_missing_or_conflicting_monthly_mapping_is_rejected(conflict: bool) -> None:
    from qfbench2_track_forecasting.horizons import HorizonMetadataError

    card = (
        {
            "targets": {
                "target_frequency": "monthly",
                "horizons": [21],
                "observation_periods": ["2020-09"],
            }
        }
        if conflict
        else {}
    )
    spec = (
        {
            "targets": {
                "target_frequency": "monthly",
                "horizons": [21],
                "observation_periods": ["2020-10"],
            }
        }
        if conflict
        else {}
    )
    with pytest.raises(HorizonMetadataError):
        mindts_core.prepare(
            panels(monthly=True),
            ["SYN-A"],
            [21],
            "2020-06-30",
            "level",
            card=card,
            forecast_spec=spec,
        )


@pytest.mark.parametrize("bad_return", [-1.0, -1.1])
def test_impossible_simple_returns_are_rejected(bad_return: float) -> None:
    source = panels(returns=True)
    source["synthetic"].loc[0, "value"] = bad_return
    with pytest.raises(ValueError):
        mindts_core.prepare(source, ["SYN-A"], [5], "2020-06-30", "log_return")
