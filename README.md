# Consistent DM Video

`consistent-dm-video` is a small Python CLI that prepares deterministic generation manifests for batches of diffusion-model videos. The manifests repeat the same visual reference, character traits, and random-seed strategy for every scene so a real video diffusion backend can keep the character's look stable across multiple clips.

The project also includes a consent gate for voice references. Only use a voice sample when it is your own voice or you have explicit permission from the speaker.

## Install

```bash
python -m pip install -e .
```

## Example

```bash
touch reference.png voice.wav
consistent-dm-video \
  --character-name "Mira" \
  --visual-reference reference.png \
  --voice-reference voice.wav \
  --i-have-voice-consent \
  --trait "green jacket" \
  --trait "short curly black hair" \
  --scene "walking through a neon market at night" \
  --scene "standing on a quiet rooftop at sunrise" \
  --output-dir outputs/mira
```

The command writes `scene_001.json`, `scene_002.json`, and so on. Each file contains:

- the same character visual and optional approved voice reference;
- a repeated identity prompt for the character;
- deterministic per-scene seeds derived from the base seed;
- settings such as FPS, resolution, identity strength, and voice similarity.

## Using a real diffusion/video backend

This repository does not download or run a heavyweight diffusion model by default. Use the generated JSON as a safe, reproducible handoff to your preferred stack, such as a video diffusion pipeline with an identity adapter plus a consented text-to-speech or voice-conversion system. Keep the same reference assets and identity controls for every manifest in the batch.
