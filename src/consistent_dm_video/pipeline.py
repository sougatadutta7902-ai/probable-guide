"""Pipeline primitives for consistent-character diffusion-model video batches.

This module intentionally separates orchestration from heavyweight model
backends.  The default backend writes deterministic generation manifests that
can be consumed by real video diffusion, image-reference, and text-to-speech
systems.  That makes the safety checks and repeatability testable without
requiring a GPU in development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from typing import Iterable


@dataclass(frozen=True)
class CharacterProfile:
    """Identity anchors used to keep a character consistent across clips."""

    name: str
    visual_reference: Path
    voice_reference: Path | None = None
    consent_confirmed: bool = False
    traits: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("character name is required")
        if not self.visual_reference.exists():
            raise FileNotFoundError(f"visual reference not found: {self.visual_reference}")
        if self.voice_reference is not None:
            if not self.voice_reference.exists():
                raise FileNotFoundError(f"voice reference not found: {self.voice_reference}")
            if not self.consent_confirmed:
                raise PermissionError(
                    "voice reference was supplied, but consent was not confirmed; "
                    "rerun with --i-have-voice-consent only for your own voice or with explicit permission"
                )


@dataclass(frozen=True)
class VideoScene:
    """A single video request in a batch."""

    prompt: str
    duration_seconds: float = 4.0
    motion: str = "cinematic subtle motion"
    negative_prompt: str = "identity drift, different face, different voice, low quality"

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("scene prompt cannot be blank")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass(frozen=True)
class GenerationSettings:
    """Reproducibility and consistency controls for a batch."""

    seed: int = 1337
    width: int = 1024
    height: int = 576
    fps: int = 24
    guidance_scale: float = 7.5
    identity_strength: float = 0.85
    voice_similarity: float = 0.9
    backend: str = "manifest"

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= self.identity_strength <= 1:
            raise ValueError("identity_strength must be between 0 and 1")
        if not 0 <= self.voice_similarity <= 1:
            raise ValueError("voice_similarity must be between 0 and 1")


@dataclass(frozen=True)
class BatchPlan:
    """Complete plan for generating a consistent multi-video batch."""

    character: CharacterProfile
    scenes: tuple[VideoScene, ...]
    settings: GenerationSettings = field(default_factory=GenerationSettings)

    def validate(self) -> None:
        self.character.validate()
        self.settings.validate()
        if not self.scenes:
            raise ValueError("at least one scene is required")
        for scene in self.scenes:
            scene.validate()


def stable_scene_seed(base_seed: int, character_name: str, prompt: str, index: int) -> int:
    """Create a deterministic per-scene seed while preserving batch identity."""

    digest = hashlib.sha256(f"{base_seed}:{character_name}:{index}:{prompt}".encode()).hexdigest()
    return int(digest[:8], 16)


def build_identity_prompt(character: CharacterProfile, scene: VideoScene) -> str:
    """Compose a prompt that repeats stable identity anchors for each video."""

    traits = ", ".join(character.traits) if character.traits else "same face, same hairstyle, same outfit style"
    return (
        f"Character: {character.name}. Keep the exact same visual identity across every shot: {traits}. "
        f"Scene: {scene.prompt}. Motion: {scene.motion}."
    )


def write_manifest(plan: BatchPlan, output_dir: Path) -> list[Path]:
    """Write one JSON manifest per scene and return the created paths."""

    plan.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for index, scene in enumerate(plan.scenes, start=1):
        scene_seed = stable_scene_seed(plan.settings.seed, plan.character.name, scene.prompt, index)
        payload = {
            "backend": plan.settings.backend,
            "character": {
                "name": plan.character.name,
                "visual_reference": str(plan.character.visual_reference),
                "voice_reference": str(plan.character.voice_reference) if plan.character.voice_reference else None,
                "traits": list(plan.character.traits),
            },
            "video": {
                "prompt": build_identity_prompt(plan.character, scene),
                "negative_prompt": scene.negative_prompt,
                "duration_seconds": scene.duration_seconds,
                "width": plan.settings.width,
                "height": plan.settings.height,
                "fps": plan.settings.fps,
                "seed": scene_seed,
                "guidance_scale": plan.settings.guidance_scale,
                "identity_strength": plan.settings.identity_strength,
                "voice_similarity": plan.settings.voice_similarity if plan.character.voice_reference else None,
            },
            "notes": [
                "Use the same visual reference/identity adapter for all scenes.",
                "Use the voice reference only with explicit consent and local law compliance.",
                "Render each scene independently, then assemble clips in an editor if needed.",
            ],
        }
        path = output_dir / f"scene_{index:03d}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        created.append(path)
    return created


def scenes_from_lines(lines: Iterable[str], duration_seconds: float) -> tuple[VideoScene, ...]:
    """Convert non-empty text lines into scene definitions."""

    return tuple(VideoScene(prompt=line.strip(), duration_seconds=duration_seconds) for line in lines if line.strip())
