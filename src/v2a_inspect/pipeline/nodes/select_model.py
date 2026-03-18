from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from ..prompt_templates import resolve_prompt
from ..response_models import (
    ModelSelectResponse,
    ModelSelection,
)
from v2a_inspect.workflows.state import InspectState

from ._shared import (
    append_state_message,
    build_model_select_segment_list,
    get_active_groups,
    invoke_structured_video,
)


def select_models(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Assign TTA or VTA preferences to track groups and member tracks."""

    options = state.get("options")
    if options is None:
        raise ValueError("select_models requires 'options' in state.")

    raw_tracks = state.get("raw_tracks")
    if raw_tracks is None:
        raise ValueError("select_models requires 'raw_tracks' in state.")

    groups = [group.model_copy(deep=True) for group in get_active_groups(state)]
    copied_tracks = [track.model_copy(deep=True) for track in raw_tracks]

    if not groups:
        return {
            "final_groups": [],
            "raw_tracks": copied_tracks,
            "progress_messages": append_state_message(
                state,
                "progress_messages",
                "Skipped model selection because there are no groups.",
            ),
        }

    gemini_file = state.get("gemini_file")
    if gemini_file is None:
        warnings = append_state_message(
            state,
            "warnings",
            "Skipped model selection because no Gemini file is available.",
        )
        return {
            "final_groups": groups,
            "raw_tracks": copied_tracks,
            "warnings": warnings,
        }

    tracks_by_id = {track.track_id: track for track in copied_tracks}
    warnings = list(state.get("warnings", []))

    for group in groups:
        member_tracks = [
            tracks_by_id[track_id]
            for track_id in group.member_ids
            if track_id in tracks_by_id
        ]
        if not member_tracks:
            continue

        if all(track.kind == "background" for track in member_tracks):
            background_selection = _background_model_selection()
            for track in member_tracks:
                track.model_selection = background_selection.model_copy(deep=True)
            group.model_selection = background_selection
            continue

        resolved_prompt = resolve_prompt("model_select").render(
            segment_list=build_model_select_segment_list(member_tracks)
        )

        try:
            response = invoke_structured_video(
                llm,
                file_obj=gemini_file,
                fps=options.fps,
                prompt=resolved_prompt,
                schema=ModelSelectResponse,
                model=options.gemini_model,
                timeout_ms=options.video_timeout_ms,
                max_retries=options.max_retries,
                label=f"model_select_{group.group_id}",
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"Model selection failed for {group.group_id}; leaving it unassigned. Reason: {exc}"
            )
            continue

        vta_scores: list[float] = []
        tta_scores: list[float] = []
        for segment in response.segments:
            segment_index = segment.segment_index
            if segment_index is None or segment_index >= len(member_tracks):
                continue

            track = member_tracks[segment_index]
            selection = _select_model_from_scores(
                motion=segment.motion_level,
                coupling=segment.event_coupling,
                source_div=segment.source_diversity,
                n_objects=track.n_scene_objects,
                duration=track.duration,
            )
            model_type: Literal["TTA", "VTA"] = selection[0]
            confidence = selection[1]
            vta_score = selection[2]
            tta_score = selection[3]
            track.model_selection = ModelSelection(
                model_type=model_type,
                confidence=confidence,
                vta_score=vta_score,
                tta_score=tta_score,
                reasoning=segment.reasoning,
                rule_based=False,
            )
            vta_scores.append(vta_score)
            tta_scores.append(tta_score)

        if not vta_scores:
            continue

        average_vta = sum(vta_scores) / len(vta_scores)
        average_tta = sum(tta_scores) / len(tta_scores)
        score_difference = average_vta - average_tta
        group_model: Literal["TTA", "VTA"]
        if score_difference >= 1.5:
            group_model = "VTA"
            group_confidence = min(0.5 + score_difference * 0.15, 0.95)
        elif score_difference <= -1.5:
            group_model = "TTA"
            group_confidence = min(0.5 + (-score_difference) * 0.15, 0.95)
        else:
            group_model = "TTA"
            group_confidence = 0.5

        member_models = {
            track.model_selection.model_type
            for track in member_tracks
            if track.model_selection is not None
        }
        if len(member_models) > 1:
            group_confidence = min(group_confidence, 0.55)

        reasoning = "; ".join(
            track.model_selection.reasoning
            for track in member_tracks
            if track.model_selection is not None and track.model_selection.reasoning
        )
        group.model_selection = ModelSelection(
            model_type=group_model,
            confidence=round(group_confidence, 3),
            vta_score=round(average_vta, 2),
            tta_score=round(average_tta, 2),
            reasoning=reasoning[:200],
            rule_based=False,
        )

    updates: dict[str, object] = {
        "final_groups": groups,
        "raw_tracks": copied_tracks,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            "Assigned model selections to track groups.",
        ),
    }
    if warnings != state.get("warnings", []):
        updates["warnings"] = warnings
    return updates


def _background_model_selection() -> ModelSelection:
    reasoning = (
        "Background track: TTA preferred to avoid foreground object sound bleed-through "
        "that VTA may introduce by attending to visible objects."
    )
    return ModelSelection(
        model_type="TTA",
        confidence=0.90,
        vta_score=1.0,
        tta_score=5.0,
        reasoning=reasoning,
        rule_based=True,
    )


def _select_model_from_scores(
    motion: float,
    coupling: float,
    source_div: float,
    n_objects: int = 0,
    duration: float = 0.0,
) -> tuple[Literal["TTA", "VTA"], float, float, float]:
    vta_raw = (motion + coupling) / 2.0

    tta_raw = source_div
    if n_objects >= 2:
        tta_raw += 1.0
    if n_objects >= 3:
        tta_raw += 0.5

    if 0.0 < duration < 1.0:
        vta_raw += 0.5

    difference = vta_raw - tta_raw
    if difference >= 1.5:
        confidence = min(0.5 + difference * 0.15, 0.95)
        return "VTA", round(confidence, 3), round(vta_raw, 2), round(tta_raw, 2)
    if difference <= -1.5:
        confidence = min(0.5 + (-difference) * 0.15, 0.95)
        return "TTA", round(confidence, 3), round(vta_raw, 2), round(tta_raw, 2)
    return "TTA", 0.5, round(vta_raw, 2), round(tta_raw, 2)
