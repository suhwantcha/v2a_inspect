"""
[4] Audio Plan node — Unified intent-conditioned audio timeline.

Merges local grouped tracks + global macro-segments into a single AudioPlan.
Silence nodes are automatically inserted before key_moment emotional beats.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from v2a_inspect.workflows.state import InspectState

from ..prompt_templates import resolve_prompt
from ..response_models import AudioPlan, AudioPlanItem, DirectorIntent, GroupedAnalysis
from ..response_models.scenes import MacroSegment, SceneObject
from ._shared import append_state_message, invoke_structured_text

logger = logging.getLogger(__name__)


def generate_audio_plan(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """
    Merge grouped local tracks + global macro-segments into a unified AudioPlan.
    Uses director_intent to set volume/intensity per item.
    Automatically inserts silence windows before key_moment beats.
    """
    options = state.get("options")
    if options is None:
        raise ValueError("generate_audio_plan requires 'options' in state.")

    grouped_analysis: GroupedAnalysis | None = state.get("grouped_analysis")
    scene_analysis = state.get("scene_analysis")
    if grouped_analysis is None or scene_analysis is None:
        raise ValueError("generate_audio_plan requires 'grouped_analysis' and 'scene_analysis' in state.")

    intent: DirectorIntent | None = state.get("director_intent")

    # ── Build prompt context ───────────────────────────────────────────────────
    local_tracks_text = _build_local_tracks_text(grouped_analysis)
    global_tracks_text = _build_global_tracks_text(scene_analysis.macro_segments)

    if intent:
        emotional_arc_text = "\n".join(
            f"  [{b.time[0]:.1f}s–{b.time[1]:.1f}s] {b.emotion} "
            f"(intensity={b.intensity:.1f}{'  ← KEY MOMENT' if b.key_moment else ''})"
            for b in intent.emotional_arc
        )
        genre = intent.genre
        overall_mood = intent.overall_mood
        audio_direction = intent.audio_direction
    else:
        emotional_arc_text = "(no director intent available)"
        genre = "unknown"
        overall_mood = "neutral"
        audio_direction = "Generate audio that matches the visual content."

    prompt = resolve_prompt("plan").render(
        genre=genre,
        overall_mood=overall_mood,
        audio_direction=audio_direction,
        emotional_arc_text=emotional_arc_text,
        local_tracks_text=local_tracks_text,
        global_tracks_text=global_tracks_text,
    )

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        plan_response = invoke_structured_text(
            llm,
            prompt=prompt,
            schema=AudioPlan,
            model=options.gemini_model,
            timeout_ms=options.text_timeout_ms,
            max_retries=options.max_retries,
            label="audio_plan",
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        # Fallback: build a simple plan directly from raw tracks without LLM
        logger.warning("Audio plan LLM call failed; building fallback plan. Reason: %s", exc)
        plan_response = _build_fallback_plan(grouped_analysis, scene_analysis.macro_segments, scene_analysis.total_duration)

    # ── Insert silence nodes before key_moment beats ───────────────────────────
    silence_pad = getattr(options, "silence_pre_key_moment_sec", 0.4)
    if intent and silence_pad > 0.0:
        silence_items = _build_silence_items(intent, silence_pad, len(plan_response.items))
        all_items = sorted(
            list(plan_response.items) + silence_items,
            key=lambda x: x.time[0],
        )
        plan_response = AudioPlan(
            items=all_items,
            total_duration=plan_response.total_duration or scene_analysis.total_duration,
        )
    else:
        plan_response = AudioPlan(
            items=plan_response.items,
            total_duration=plan_response.total_duration or scene_analysis.total_duration,
        )

    silence_count = sum(1 for i in plan_response.items if i.type == "silence")
    message = (
        f"Audio plan built: {len(plan_response.items)} items "
        f"({silence_count} silence windows)."
    )
    return {
        "audio_plan": plan_response,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_local_tracks_text(grouped_analysis: GroupedAnalysis) -> str:
    """Render grouped local tracks as a numbered text list for the LLM prompt."""
    lines: list[str] = []
    tracks_by_id = {t.track_id: t for t in grouped_analysis.raw_tracks}

    for group in grouped_analysis.groups:
        track = tracks_by_id.get(group.member_ids[0]) if group.member_ids else None
        if track is None or track.kind not in ("dialogue", "sfx"):
            continue
        # Collect all member timestamps
        member_times = [
            f"{tracks_by_id[mid].start:.1f}s–{tracks_by_id[mid].end:.1f}s"
            for mid in group.member_ids
            if mid in tracks_by_id
        ]
        # Use the representative track's pan for the group
        pan_val = track.pan if hasattr(track, "pan") else 0.0
        lines.append(
            f"[{group.group_id}] kind={track.kind} "
            f"times={', '.join(member_times)} "
            f"pan={pan_val:+.2f} "
            f'description="{group.canonical_description}"'
        )
    return "\n".join(lines) if lines else "(none)"


def _build_global_tracks_text(macro_segments: list[MacroSegment]) -> str:
    """Render global macro-segment tracks as text for the LLM prompt."""
    lines: list[str] = []
    for seg in macro_segments:
        for obj in seg.music:
            lines.append(
                f"[music] {seg.time_range.start:.1f}s–{seg.time_range.end:.1f}s "
                f'"{obj.description}"'
            )
        for obj in seg.ambience:
            lines.append(
                f"[ambience] {seg.time_range.start:.1f}s–{seg.time_range.end:.1f}s "
                f'"{obj.description}"'
            )
    return "\n".join(lines) if lines else "(none)"


def _build_silence_items(
    intent: DirectorIntent,
    pad_sec: float,
    existing_count: int,
) -> list[AudioPlanItem]:
    """Create silence AudioPlanItems before each key_moment beat."""
    items: list[AudioPlanItem] = []
    for beat in intent.emotional_arc:
        if not beat.key_moment:
            continue
        silence_end = beat.time[0]
        silence_start = max(0.0, silence_end - pad_sec)
        if silence_start >= silence_end:
            continue
        idx = existing_count + len(items)
        items.append(
            AudioPlanItem(
                item_id=f"plan_silence_{idx}",
                type="silence",
                time=(silence_start, silence_end),
                description=f"Intentional silence before key moment: {beat.emotion}",
                volume=0.0,
                intensity=0.0,
                pan=0.0,
                confidence=1.0,
            )
        )
    return items


def _build_fallback_plan(
    grouped_analysis: GroupedAnalysis,
    macro_segments: list[MacroSegment],
    total_duration: float,
) -> AudioPlan:
    """Build a minimal plan directly from tracks without LLM, used as fallback."""
    items: list[AudioPlanItem] = []
    tracks_by_id = {t.track_id: t for t in grouped_analysis.raw_tracks}

    for group in grouped_analysis.groups:
        track = tracks_by_id.get(group.member_ids[0]) if group.member_ids else None
        if track is None:
            continue
        items.append(
            AudioPlanItem(
                item_id=f"plan_{track.kind}_{len(items)}",
                type=track.kind,  # type: ignore[arg-type]
                time=(track.start, track.end),
                description=group.canonical_description,
                track_id=track.track_id,
            )
        )

    for seg in macro_segments:
        for obj in seg.music:
            items.append(
                AudioPlanItem(
                    item_id=f"plan_music_{len(items)}",
                    type="music",
                    time=(seg.time_range.start, seg.time_range.end),
                    description=obj.description,
                )
            )
        for obj in seg.ambience:
            items.append(
                AudioPlanItem(
                    item_id=f"plan_ambience_{len(items)}",
                    type="ambience",
                    time=(seg.time_range.start, seg.time_range.end),
                    description=obj.description,
                )
            )

    return AudioPlan(
        items=sorted(items, key=lambda x: x.time[0]),
        total_duration=total_duration,
    )
