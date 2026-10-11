"""## Executive summary (read this first)

Export a minimal participant source ZIP for the team's new empty GitHub repository.
Use an explicit source allowlist; exclude task data, outputs, credentials and Git history.
This source ZIP is not a CodaBench submission and does not publish anything.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile


def export_source(target: pathlib.Path) -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    selected = [
        root / name
        for name in (
            "pyproject.toml",
            "LICENSE",
            "ARTIFACT_PROVENANCE.md",
            "README.mindts.md",
            "Dockerfile.mindts",
            "Dockerfile.mindts.dockerignore",
            ".gitignore",
            ".github/workflows/mindts-dev-image.yml",
            "tests/conftest.py",
            "tests/test_mindts_agent.py",
            "tests/test_mindts_submission.py",
        )
    ]
    selected.extend(sorted((root / "qfbench2_track_forecasting").glob("*.py")))
    if not all(path.is_file() and not path.is_symlink() for path in selected):
        raise ValueError("export must run from the complete participant source checkout")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any user-owned pre-existing archive.
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(root).as_posix())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export source, not a competition submission")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    export_source(args.out)
    print(f"Exported participant source to {args.out}; no publication or submission made")
    return 0


if __name__ == "__main__":
    sys.exit(main())
