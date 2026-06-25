"""Command line interface for consistent DM video batch planning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import BatchPlan, CharacterProfile, GenerationSettings, scenes_from_lines, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic manifests for a consistent-character diffusion-model video batch."
    )
    parser.add_argument("--character-name", required=True, help="Name or label for the consistent character.")
    parser.add_argument("--visual-reference", required=True, type=Path, help="Image file anchoring the character look.")
    parser.add_argument("--voice-reference", type=Path, help="Optional voice sample for approved voice consistency.")
    parser.add_argument(
        "--i-have-voice-consent",
        action="store_true",
        help="Confirm the voice reference is yours or you have explicit permission to use it.",
    )
    parser.add_argument("--scene", action="append", default=[], help="Scene prompt. Repeat for multiple videos.")
    parser.add_argument("--scenes-file", type=Path, help="Text file with one scene prompt per line.")
    parser.add_argument("--trait", action="append", default=[], help="Stable character trait to repeat in every prompt.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated manifests.")
    parser.add_argument("--duration", type=float, default=4.0, help="Duration in seconds for every scene.")
    parser.add_argument("--seed", type=int, default=1337, help="Base seed for deterministic per-scene seeds.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--identity-strength", type=float, default=0.85)
    parser.add_argument("--voice-similarity", type=float, default=0.9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_lines = list(args.scene)
    if args.scenes_file:
        scene_lines.extend(args.scenes_file.read_text(encoding="utf-8").splitlines())

    plan = BatchPlan(
        character=CharacterProfile(
            name=args.character_name,
            visual_reference=args.visual_reference,
            voice_reference=args.voice_reference,
            consent_confirmed=args.i_have_voice_consent,
            traits=tuple(args.trait),
        ),
        scenes=scenes_from_lines(scene_lines, args.duration),
        settings=GenerationSettings(
            seed=args.seed,
            width=args.width,
            height=args.height,
            fps=args.fps,
            identity_strength=args.identity_strength,
            voice_similarity=args.voice_similarity,
        ),
    )
    paths = write_manifest(plan, args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
