from pathlib import Path
import json

import pytest

from consistent_dm_video.pipeline import (
    BatchPlan,
    CharacterProfile,
    VideoScene,
    stable_scene_seed,
    write_manifest,
)


def test_stable_scene_seed_is_deterministic():
    assert stable_scene_seed(7, "Ada", "walk", 1) == stable_scene_seed(7, "Ada", "walk", 1)
    assert stable_scene_seed(7, "Ada", "walk", 1) != stable_scene_seed(7, "Ada", "run", 1)


def test_write_manifest_repeats_identity_controls(tmp_path: Path):
    visual = tmp_path / "character.png"
    visual.write_bytes(b"fake image")
    plan = BatchPlan(
        character=CharacterProfile("Ada", visual, traits=("red coat", "silver glasses")),
        scenes=(VideoScene("explores a forest"), VideoScene("boards a train")),
    )

    paths = write_manifest(plan, tmp_path / "out")

    assert len(paths) == 2
    first = json.loads(paths[0].read_text())
    second = json.loads(paths[1].read_text())
    assert first["character"] == second["character"]
    assert "red coat" in first["video"]["prompt"]
    assert first["video"]["seed"] != second["video"]["seed"]


def test_voice_reference_requires_consent(tmp_path: Path):
    visual = tmp_path / "character.png"
    voice = tmp_path / "voice.wav"
    visual.write_bytes(b"fake image")
    voice.write_bytes(b"fake voice")
    plan = BatchPlan(
        character=CharacterProfile("Ada", visual, voice_reference=voice, consent_confirmed=False),
        scenes=(VideoScene("speaks to camera"),),
    )

    with pytest.raises(PermissionError):
        write_manifest(plan, tmp_path / "out")
