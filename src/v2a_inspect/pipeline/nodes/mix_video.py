from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
from moviepy.audio.fx import MultiplyVolume

from v2a_inspect.workflows.state import InspectState
from ._shared import append_state_message

logger = logging.getLogger(__name__)

# dB reduction applied when a 'ducks' relation is active (Phase 2)
_DUCK_VOLUME_FACTOR = 0.25  # approx -12 dB


def mix_video_tracks(state: InspectState) -> dict[str, object]:
    """
    Mix generated audio tracks over the original video.

    If audio_plan is present (Phase 1+), uses plan items for timestamps,
    volume, and pan. Silence windows mute any overlapping audio.
    Falls back to raw_tracks if no audio_plan.
    """
    video_path = state.get("video_path")
    generated_audio = state.get("generated_audio", {})
    audio_plan = state.get("audio_plan")
    relation_graph = state.get("relation_graph")  # Phase 2, may be None

    if not video_path or not generated_audio:
        return {
            "progress_messages": append_state_message(
                state, "progress_messages", "No video or audio to mix."
            )
        }

    out_dir = Path(tempfile.mkdtemp(prefix="v2a_mixed_video_"))
    out_path = str(out_dir / "mixed_output.mp4")

    try:
        video = VideoFileClip(video_path)
        audio_clips = []

        # Keep original audio if present (optional: could remove it)
        if video.audio is not None:
            audio_clips.append(video.audio)

        # Compute silence windows from audio_plan
        silence_windows: list[tuple[float, float]] = []
        if audio_plan:
            silence_windows = [
                (item.time[0], item.time[1])
                for item in audio_plan.items
                if item.type == "silence"
            ]

        if audio_plan and audio_plan.items:
            audio_clips = _mix_from_plan(
                audio_plan, generated_audio, audio_clips, silence_windows, relation_graph
            )
        else:
            audio_clips = _mix_from_raw_tracks(state, generated_audio, audio_clips)

        if audio_clips:
            final_audio = CompositeAudioClip(audio_clips)
            video = video.with_audio(final_audio)

        logger.info("Writing mixed video: %s", out_path)
        video.write_videofile(out_path, audio_codec="aac", fps=video.fps, logger=None)

        video.close()
        for clip in audio_clips:
            if hasattr(clip, "close"):
                clip.close()

    except Exception as e:
        logger.error("Failed to mix video: %s", e)
        return {
            "progress_messages": append_state_message(
                state, "progress_messages", f"Failed mixing video: {e}"
            )
        }

    return {
        "mixed_video_path": out_path,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Mixed {len(generated_audio)} audio tracks into final video.",
        ),
    }


# ── Plan-based mixing (Phase 1+) ───────────────────────────────────────────────

def _mix_from_plan(
    audio_plan,
    generated_audio: dict[str, str],
    existing_clips: list,
    silence_windows: list[tuple[float, float]],
    relation_graph,
) -> list:
    """Build audio clip list from AudioPlan items."""
    # Build ducking map from relation_graph (Phase 2)
    ducks_map: dict[str, list[tuple[tuple[float, float], float]]] = {}
    if relation_graph:
        item_time: dict[str, tuple[float, float]] = {
            item.item_id: item.time for item in audio_plan.items
        }
        for rel in relation_graph.relations:
            if rel.relation == "ducks":
                duck_window = item_time.get(rel.from_item_id)
                if duck_window:
                    ducks_map.setdefault(rel.to_item_id, []).append(
                        (duck_window, rel.strength)
                    )

    clips = list(existing_clips)
    for item in audio_plan.items:
        if item.type == "silence":
            continue
        wav_path = generated_audio.get(item.item_id)
        if not wav_path:
            continue

        clip = AudioFileClip(wav_path).with_start(item.time[0])

        # 1. Volume from plan
        if item.volume != 1.0:
            clip = clip.with_effects([MultiplyVolume(item.volume)])

        # 2. Apply duck factor if this item is being ducked (Phase 2)
        if item.item_id in ducks_map:
            clip = _apply_duck_to_clip(clip, item, ducks_map[item.item_id])

        # 3. Simple stereo pan via channel weighting
        if item.pan != 0.0:
            clip = _apply_pan(clip, item.pan)

        # 4. Attenuate during silence windows (only for global tracks to preserve SFX/Dialogue)
        if item.type in ("music", "ambience"):
            clip = _attenuate_during_silence(clip, item, silence_windows)

        clips.append(clip)
    return clips


def _mix_from_raw_tracks(
    state: InspectState,
    generated_audio: dict[str, str],
    existing_clips: list,
) -> list:
    """Original raw_track based mixing (fallback)."""
    raw_tracks = state.get("raw_tracks", [])
    clips = list(existing_clips)
    for track in raw_tracks:
        wav_path = generated_audio.get(track.track_id)
        if not wav_path:
            continue
        clip = AudioFileClip(wav_path).with_start(track.start)
        clips.append(clip)
    return clips


# ── Audio effect helpers ──────────────────────────────────────────────────────

def _apply_pan(clip, pan: float):
    """
    Simple stereo pan: -1.0 = left only, 0.0 = center, 1.0 = right only.
    Uses channel volume weighting. Safe for both scalar and array t.
    """
    try:
        left_vol = max(0.0, 1.0 - pan)
        right_vol = max(0.0, 1.0 + pan)
        peak = max(left_vol, right_vol, 1e-6)
        left_vol /= peak
        right_vol /= peak

        def _pan_audio(get_frame, t):
            import numpy as np
            frame = get_frame(t)  # shape: (n_samples, channels) or (n_samples,)
            if frame.ndim == 1:
                frame = np.c_[frame, frame]  # mono → stereo
            frame = frame.copy().astype(float)
            frame[:, 0] *= left_vol
            frame[:, 1] *= right_vol
            return frame

        return clip.transform(_pan_audio, apply_to="audio")
    except Exception:
        return clip


def _apply_duck_to_clip(clip, item, duck_entries: list[tuple[tuple[float, float], float]]):
    """Reduce clip volume during periods when a 'ducks' source is playing.
    t may be a numpy array — we use vectorized operations throughout.
    """
    try:
        start_time = float(item.time[0])

        def _duck_audio(get_frame, t):
            import numpy as np
            frame = get_frame(t)  # (n_samples, channels)
            frame = np.array(frame, dtype=float)

            # t can be scalar float or 1-D array of sample times
            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            abs_t = t_arr + start_time  # absolute timeline times

            # Start with all-ones factor per sample
            factors = np.ones(len(t_arr))
            for (duck_start, duck_end), strength in duck_entries:
                in_duck = (abs_t >= duck_start) & (abs_t <= duck_end)
                duck_factor = 1.0 - strength * (1.0 - _DUCK_VOLUME_FACTOR)
                factors = np.where(in_duck, np.minimum(factors, duck_factor), factors)

            # Broadcast factors to frame shape
            if frame.ndim == 2:
                factors = factors[:, np.newaxis]  # (n, 1) → broadcasts across channels
            return frame * factors

        return clip.transform(_duck_audio, apply_to="audio")
    except Exception:
        return clip


def _attenuate_during_silence(
    clip,
    item,
    silence_windows: list[tuple[float, float]],
):
    """Zero out audio samples that fall inside any silence window.
    t may be a numpy array — fully vectorized.
    """
    if not silence_windows:
        return clip

    start_time = float(item.time[0])
    item_end = float(item.time[1])
    overlapping = [
        (sw_start, sw_end)
        for sw_start, sw_end in silence_windows
        if sw_start < item_end and sw_end > start_time
    ]
    if not overlapping:
        return clip

    try:
        def _silence_audio(get_frame, t):
            import numpy as np
            frame = get_frame(t)  # (n_samples, channels)
            frame = np.array(frame, dtype=float)

            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            abs_t = t_arr + start_time  # absolute timeline times

            # Build a mask: True where sample should be silent
            silent_mask = np.zeros(len(t_arr), dtype=bool)
            for sw_start, sw_end in overlapping:
                silent_mask |= (abs_t >= sw_start) & (abs_t <= sw_end)

            if silent_mask.any():
                if frame.ndim == 2:
                    frame[silent_mask, :] = 0.0
                else:
                    frame[silent_mask] = 0.0
            return frame

        return clip.transform(_silence_audio, apply_to="audio")
    except Exception:
        return clip
