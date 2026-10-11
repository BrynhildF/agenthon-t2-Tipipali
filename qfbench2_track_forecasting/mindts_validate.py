"""## Executive summary (read this first)

Generate numerical-only outputs for every public card in a pinned official checkout.
Then invoke that checkout's official smoke command, including manifest checks.
Do not score against undisclosed outcomes or bypass unsupported filesystem checks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tomllib

from .cutoff import trusted_asof
from .mindts_agent import main as forecast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate every public task with the official scorer"
    )
    parser.add_argument("--official-track", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--gates-only",
        action="store_true",
        help="Windows diagnostic only; NOT a complete smoke validation",
    )
    args = parser.parse_args(argv)
    root = args.official_track.resolve()
    if not (root / "qfbench2_track_forecasting/scoring.py").is_file():
        parser.error("provide the official Track 2 checkout, not an arbitrary unit directory")
    cards = sorted((root / "units").glob("*/card.toml"))
    if not cards:
        parser.error("official checkout has no cards")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONUTF8": "1"}
    gate_code = (
        "import pathlib,tomllib,sys; "
        "from qfbench2_track_forecasting.scoring import build_verifier; "
        "u=pathlib.Path(sys.argv[1]); "
        "c={'card':tomllib.loads((u/'card.toml').read_text(encoding='utf-8')),"
        "'unit_dir':u,'output_dir':pathlib.Path(sys.argv[2])}; "
        "v=build_verifier(c).run(c); print(v.detail); sys.exit(not v.admissible)"
    )
    for path in cards:
        unit = path.parent
        out = args.out_dir.resolve() / unit.name
        try:
            card = tomllib.loads(path.read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                forecast(
                    [
                        "--panels",
                        str(unit / "panels"),
                        "--text",
                        str(unit / "text"),
                        "--asof",
                        trusted_asof(card),
                        "--out",
                        str(out / "forecast.parquet"),
                        "--offline",
                    ]
                )
            command = (
                [sys.executable, "-c", gate_code, str(unit), str(out)]
                if args.gates_only
                else [
                    sys.executable,
                    "-m",
                    "qfbench2_common.smoke",
                    str(unit),
                    str(out),
                    "--track",
                    "forecasting",
                ]
            )
            checked = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
                check=False,
            )
            if checked.returncode:
                raise ValueError((checked.stdout + checked.stderr)[-2000:])
        except (Exception, SystemExit) as exc:
            failed.append({"unit": unit.name, "error": str(exc)})
    result = {
        "scope": "gates-only" if args.gates_only else "full-official-smoke",
        "total": len(cards),
        "passed": len(cards) - len(failed),
        "failures": failed,
    }
    (args.out_dir / "validation-summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
