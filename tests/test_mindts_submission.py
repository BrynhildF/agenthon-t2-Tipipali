"""## Executive summary (read this first)

Check Development descriptors and official team-proof packing using synthetic keys only.
Windows must refuse secure packing rather than weakening the official filesystem controls.
"""

from __future__ import annotations

import json
import os
import pathlib
import zipfile
from typing import Any

import pytest
from qfbench2_common.contracts.descriptor import SubmissionDescriptor
from qfbench2_common.team_claim import seal_for_team

from qfbench2_track_forecasting import mindts_submission as submission

DIGEST = "sha256:" + "a" * 64  # Synthetic fixture, never a deliverable image digest.


def test_draft_matches_official_dev_contract_without_fake_team_identity() -> None:
    draft = submission.create_draft(DIGEST)
    assert "team_id" not in draft and "descriptor_digest" not in draft
    assert draft["phase"] == "dev"
    assert draft["category"] == "api"
    assert draft["competition_id"] == "agenthon2026-forecasting-dev"
    assert draft["image"]["repository"] == "brynhildf/agenthon-t2-tipipali"
    assert draft["models"] == [submission.HOUSE_MODEL]
    sealed = seal_for_team(draft, 188, "synthetic-test-key-not-a-real-credential")
    SubmissionDescriptor.from_mapping(sealed)
    assert sealed["team_id"] != "188"
    assert "synthetic-test-key" not in json.dumps(sealed)


@pytest.mark.parametrize("digest", ["latest", "sha256:" + "0" * 64, "sha256:" + "A" * 64])
def test_invalid_digest_is_rejected(digest: str) -> None:
    with pytest.raises(ValueError):
        submission.create_draft(digest)


def test_draft_command_reads_build_metadata_and_does_not_pack(tmp_path: pathlib.Path) -> None:
    metadata = tmp_path / "image-metadata.json"
    metadata.write_text(json.dumps({"containerimage.digest": DIGEST}), encoding="utf-8")
    out = tmp_path / "draft"
    assert submission.main(["--image-metadata", str(metadata), "--out-dir", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"submission.draft.json"}
    with pytest.raises(SystemExit):
        submission.main(["--digest", DIGEST, "--out-dir", str(out)])


def test_source_export_contains_workflow_but_no_data_outputs_or_git(tmp_path: pathlib.Path) -> None:
    from qfbench2_track_forecasting.mindts_export import export_source

    target = tmp_path / "source.zip"
    export_source(target)
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        assert ".github/workflows/mindts-dev-image.yml" in names
        assert "qfbench2_track_forecasting/mindts_agent.py" in names
        assert "tests/conftest.py" in names
        assert "ARTIFACT_PROVENANCE.md" in names
        assert not any(n.startswith(("out/", "units/", ".git/")) for n in names)
        assert ".github/workflows/ci.yml" not in names
        workflow = archive.read(".github/workflows/mindts-dev-image.yml").decode()
        assert "python -m pytest .official-toolkit/common" in workflow
        for flag in (
            "--user 65534:65534",
            "--cap-drop ALL",
            "--pids-limit 256",
            "noexec,nosuid,nodev,size=64m",
        ):
            assert flag in workflow
    with pytest.raises(FileExistsError):
        export_source(target)


def test_noninteractive_or_unsupported_packing_refuses_before_key_prompt(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(submission.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(submission.getpass, "getpass", lambda *args: pytest.fail("key requested"))
    with pytest.raises(SystemExit) as exc:
        submission.main(["--digest", DIGEST, "--out-dir", str(tmp_path / "pack"), "--pack"])
    assert exc.value.code == 2
    assert not (tmp_path / "pack").exists()


@pytest.mark.skipif(os.name != "posix", reason="official secure packer requires Unix permissions")
def test_hidden_prompt_packs_exactly_two_members_without_plaintext_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "synthetic-test-key-not-a-real-credential"
    monkeypatch.setattr(submission.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(submission.getpass, "getpass", lambda *args: key)
    assert submission.main(["--digest", DIGEST, "--out-dir", str(tmp_path), "--pack"]) == 0
    with zipfile.ZipFile(tmp_path / "submission.zip") as archive:
        assert set(archive.namelist()) == {"submission.json", "team-claim.json"}
        assert all(key.encode() not in archive.read(name) for name in archive.namelist())
        sealed: dict[str, Any] = json.loads(archive.read("submission.json"))
        SubmissionDescriptor.from_mapping(sealed)
    assert (tmp_path / "submission.zip").stat().st_mode & 0o777 == 0o600
    with pytest.raises(SystemExit):
        submission.main(["--digest", DIGEST, "--out-dir", str(tmp_path), "--pack"])
