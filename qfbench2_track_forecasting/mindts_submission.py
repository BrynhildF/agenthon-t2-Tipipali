"""## Executive summary (read this first)

Prepare team 188's Development descriptor and pack it with the official team-proof toolkit.
The image digest must come from an actual registry build. Never store or log a Team Key.
Packing requires Linux and a hidden terminal prompt; draft creation also works on Windows.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.metadata
import json
import os
import pathlib
import re
import sys
import zipfile
from typing import Any

import qfbench2_common
from qfbench2_common.contracts.descriptor import SubmissionDescriptor, seal_descriptor_digest
from qfbench2_common.team_claim import pack_submission

TEAM_NUMBER = 188
IMAGE_REPOSITORY = "brynhildf/agenthon-t2-tipipali"
HOUSE_MODEL = {
    "name": "nvidia/nemotron-3-super-120b-a12b",
    "version": "rl-030326-fp8",
    "revision": "rl-030326-fp8",
    "training_cutoff": "unpublished",
    "access": "api",
}


def create_draft(digest: str) -> dict[str, Any]:
    """Copy the released C5 fixture, replacing example identity and model information."""
    if importlib.metadata.version("qfbench2-common") != "2.6.0":
        raise ValueError("install the pinned official qfbench2-common==2.6.0 first")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest) or digest == "sha256:" + "0" * 64:
        raise ValueError("provide a real lowercase sha256 image digest from the build")
    root = pathlib.Path(qfbench2_common.__file__).parent
    fixture = root / "contracts/fixtures/c5/forecasting_dev.json"
    draft: dict[str, Any] = json.loads(fixture.read_text(encoding="utf-8"))
    draft.pop("team_id")
    draft.pop("descriptor_digest")
    draft["image"] = {"registry": "ghcr.io", "repository": IMAGE_REPOSITORY, "digest": digest}
    draft["models"] = [dict(HOUSE_MODEL)]
    draft["image_access"] = "public"
    draft["license"] = "MIT"
    # A disposable alias checks all other descriptor fields; it never enters a deliverable.
    validation = seal_descriptor_digest({**draft, "team_id": "validation-only-not-a-team"})
    SubmissionDescriptor.from_mapping(validation)
    return draft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare team 188's Development submission")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--digest", help="actual GHCR manifest digest, not an image tag")
    source.add_argument(
        "--image-metadata",
        type=pathlib.Path,
        help="image-metadata.json downloaded from the build workflow",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--pack", action="store_true", help="hidden-key prompt; Linux only")
    args = parser.parse_args(argv)
    digest = args.digest
    if args.image_metadata:
        metadata = json.loads(args.image_metadata.read_text(encoding="utf-8"))
        digest = metadata.get("containerimage.digest")
    if not isinstance(digest, str):
        parser.error("build metadata must contain containerimage.digest")
    try:
        draft = create_draft(digest)
    except ValueError as exc:
        parser.error(str(exc))
    if args.pack:
        if os.name != "posix" or not hasattr(os, "fchmod"):
            parser.error(
                "official secure packer requires Linux; run this command in Docker or Linux"
            )
        if not sys.stdin.isatty():
            parser.error("packing requires an interactive terminal for hidden Team Key input")
        if any((args.out_dir / name).exists() for name in ("submission.zip", "submission.json")):
            parser.error("submission files already exist; choose a fresh output directory")
        key = getpass.getpass("Team 188 Key (hidden, never saved): ")
        try:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            archive = args.out_dir / "submission.zip"
            alias = pack_submission(draft, TEAM_NUMBER, key, archive)
        finally:
            key = ""
        with zipfile.ZipFile(archive) as packed:
            if set(packed.namelist()) != {"submission.json", "team-claim.json"}:
                raise ValueError("unexpected submission ZIP contents")
            descriptor = packed.read("submission.json")
        (args.out_dir / "submission.json").write_bytes(descriptor)
        print(f"Prepared Development ZIP for team 188 ({alias}); no upload made")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / "submission.draft.json"
        if path.exists():
            parser.error("submission.draft.json already exists; choose a fresh output directory")
        path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        print("Prepared unsealed draft. This is NOT an uploadable ZIP; run --pack on Linux next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
