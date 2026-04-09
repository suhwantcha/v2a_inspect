from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from v2a_inspect.workflows.state import InspectState
from v2a_inspect.clients.audio import (
    generate_dialogue_openai,
    generate_dummy_audio,
    generate_sfx_elevenlabs,
    generate_music_elevenlabs,
)
from ._shared import append_state_message

logger = logging.getLogger(__name__)


def generate_audio_tracks(state: InspectState) -> dict[str, object]:
    """
    Generate audio for each item in audio_plan (preferred) or raw_tracks (fallback).

    - Skips 'silence' items (no audio to generate).
    - Uses volume / intensity from plan to enrich the generation prompt.
    - Stores results as generated_audio[item_id] = wav_path.
    """
    audio_plan = state.get("audio_plan")

    if audio_plan is not None and audio_plan.items:
        return _generate_from_plan(state, audio_plan)
    else:
        return _generate_from_raw_tracks(state)


# ── Plan-based generation (new path) ──────────────────────────────────────────

def _generate_from_plan(state: InspectState, audio_plan) -> dict[str, object]:
    """Generate audio from AudioPlan items (intent-conditioned)."""
    audio_dir = Path(tempfile.mkdtemp(prefix="v2a_generated_audio_"))

    # Preserve already-generated audio from previous iterations (refinement support)
    generated_audio: dict[str, str] = dict(state.get("generated_audio") or {})

    # Determine which items need (re)generation:
    # - First run: all non-silence items
    # - Refinement: items in evaluation_score.weak_item_ids + new items
    score = state.get("evaluation_score")
    is_refinement = score is not None and state.get("refinement_iteration", 0) > 0
    if is_refinement and score:
        regenerate_ids = set(score.weak_item_ids)
        # Also include items not yet generated
        regenerate_ids.update(
            item.item_id for item in audio_plan.items
            if item.item_id not in generated_audio and item.type != "silence"
        )
        logger.info(
            "Refinement re-generation: %d items targeted.", len(regenerate_ids)
        )
    else:
        regenerate_ids = None  # None = generate all

    # Respect causal order if available from relation node (Phase 2)
    relation_graph = state.get("relation_graph")
    if relation_graph and relation_graph.causal_order:
        order_index = {item_id: i for i, item_id in enumerate(relation_graph.causal_order)}
        items = sorted(audio_plan.items, key=lambda x: order_index.get(x.item_id, 9999))
    else:
        items = sorted(audio_plan.items, key=lambda x: x.time[0])

    n_generated = 0
    n_skipped = 0
    for item in items:
        if item.type == "silence":
            continue

        # Skip items that don't need regeneration during refinement
        if regenerate_ids is not None and item.item_id not in regenerate_ids:
            n_skipped += 1
            continue

        out_path = str(audio_dir / f"{item.item_id}.wav")
        duration = float(item.time[1] - item.time[0])
        desc = _enrich_description(item)

        try:
            path = _call_generation_api(item.type, desc, out_path, duration)
            if path:
                generated_audio[item.item_id] = path
                n_generated += 1
        except Exception as e:
            logger.error("Failed to generate audio for item %s: %s", item.item_id, e)

    suffix = f" (refinement: {n_generated} re-generated, {n_skipped} kept)" if is_refinement else ""
    return {
        "generated_audio": generated_audio,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Generated {n_generated} audio tracks from audio_plan{suffix}.",
        ),
    }


def _enrich_description(item) -> str:
    """Add intensity context to the description for better conditioning."""
    desc = item.description.strip()
    intensity = item.intensity

    # For music/ambience, add intensity hint to prompt
    if item.type in ("music", "ambience") and intensity != 0.5:
        if intensity >= 0.8:
            desc = f"Intense and powerful: {desc}"
        elif intensity >= 0.6:
            desc = f"Moderate intensity: {desc}"
        elif intensity <= 0.3:
            desc = f"Subtle and quiet: {desc}"
    return desc


def _call_generation_api(
    kind: str,
    description: str,
    out_path: str,
    duration: float,
) -> str | None:
    """Route to the appropriate audio generation API."""
    if kind == "dialogue":
        # Quoted text → OpenAI TTS; otherwise vocal SFX → ElevenLabs
        if '"' in description or "'" in description:
            return generate_dialogue_openai(description, out_path, duration=duration)
        else:
            return generate_sfx_elevenlabs(description, out_path, duration=duration)
    elif kind in ("sfx", "ambience"):
        return generate_sfx_elevenlabs(description, out_path, duration=duration)
    elif kind == "music":
        return generate_music_elevenlabs(description, out_path, duration=duration)
    else:
        return generate_dummy_audio(duration, out_path)


# ── Raw-track fallback (original path, no audio_plan) ─────────────────────────

def _generate_from_raw_tracks(state: InspectState) -> dict[str, object]:
    """Original generation logic based on raw_tracks (used when no audio_plan)."""
    raw_tracks = state.get("raw_tracks", [])
    if not raw_tracks:
        return {
            "progress_messages": append_state_message(
                state, "progress_messages", "No tracks to generate audio for."
            )
        }

    audio_dir = Path(tempfile.mkdtemp(prefix="v2a_generated_audio_"))
    generated_audio: dict[str, str] = {}

    for track in raw_tracks:
        out_path = str(audio_dir / f"{track.track_id}.wav")
        duration = float(track.end - track.start)

        try:
            path = _call_generation_api(track.kind, track.description.strip(), out_path, duration)
            if path:
                generated_audio[track.track_id] = path
        except Exception as e:
            logger.error("Failed to generate audio for track %s: %s", track.track_id, e)

    return {
        "generated_audio": generated_audio,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Generated {len(generated_audio)} audio track files (from raw_tracks).",
        ),
    }
