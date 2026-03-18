from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from v2a_inspect.workflows.state import InspectState

from ..prompt_templates import resolve_prompt
from ..response_models import TrackGroup, VLMVerifyResponse
from ._shared import (
    append_state_message,
    build_verify_segment_list,
    get_active_groups,
    invoke_structured_video,
)


def verify_groups(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Use Gemini VLM to confirm or split multi-member track groups."""

    options = state.get("options")
    if options is None:
        raise ValueError("verify_groups requires 'options' in state.")

    raw_tracks = state.get("raw_tracks")
    if raw_tracks is None:
        raise ValueError("verify_groups requires 'raw_tracks' in state.")

    groups = [group.model_copy(deep=True) for group in get_active_groups(state)]
    if not groups:
        return {
            "verified_groups": [],
            "final_groups": [],
            "progress_messages": append_state_message(
                state,
                "progress_messages",
                "Skipped VLM verification because there are no groups.",
            ),
        }

    gemini_file = state.get("gemini_file")
    if gemini_file is None:
        warnings = append_state_message(
            state,
            "warnings",
            "Skipped VLM verification because no Gemini file is available.",
        )
        return {
            "verified_groups": groups,
            "final_groups": groups,
            "warnings": warnings,
        }

    tracks_by_id = {track.track_id: track for track in raw_tracks}
    warnings = list(state.get("warnings", []))
    updated_groups: list[TrackGroup] = []

    for group in groups:
        if len(group.member_ids) < 2:
            group.vlm_verified = False
            updated_groups.append(group)
            continue

        member_scenes = {
            tracks_by_id[track_id].scene_index
            for track_id in group.member_ids
            if track_id in tracks_by_id
        }
        if len(member_scenes) < 2:
            group.vlm_verified = False
            updated_groups.append(group)
            continue

        resolved_prompt = resolve_prompt("vlm_verify").render(
            canonical_description=group.canonical_description,
            segment_list=build_verify_segment_list(group, tracks_by_id),
        )

        try:
            response = invoke_structured_video(
                llm,
                file_obj=gemini_file,
                fps=options.fps,
                prompt=resolved_prompt,
                schema=VLMVerifyResponse,
                model=options.gemini_model,
                timeout_ms=options.video_timeout_ms,
                max_retries=options.max_retries,
                label=f"vlm_verify_{group.group_id}",
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"VLM verification failed for {group.group_id}; keeping the text group. Reason: {exc}"
            )
            group.vlm_verified = False
            updated_groups.append(group)
            continue

        if response.same_entity is True or response.same_entity == "uncertain":
            group.vlm_verified = response.same_entity is True
            updated_groups.append(group)
            continue

        confirmed_groups = _normalize_confirmed_groups(response, len(group.member_ids))
        for sub_index, sub_members in enumerate(confirmed_groups):
            sub_track_ids = [group.member_ids[index] for index in sub_members]
            sub_tracks = [
                tracks_by_id[track_id]
                for track_id in sub_track_ids
                if track_id in tracks_by_id
            ]
            canonical_description = (
                max(sub_tracks, key=lambda track: len(track.description)).description
                if sub_tracks
                else group.canonical_description
            )
            updated_groups.append(
                TrackGroup(
                    group_id=f"{group.group_id}_{chr(ord('a') + sub_index)}",
                    canonical_description=canonical_description,
                    member_ids=sub_track_ids,
                    vlm_verified=True,
                )
            )

        covered_ids = {
            group.member_ids[index]
            for sub_members in confirmed_groups
            for index in sub_members
        }
        uncovered_ids = [
            track_id for track_id in group.member_ids if track_id not in covered_ids
        ]
        for sub_index, track_id in enumerate(uncovered_ids):
            track = tracks_by_id.get(track_id)
            updated_groups.append(
                TrackGroup(
                    group_id=f"{group.group_id}_uc{sub_index}",
                    canonical_description=(
                        track.description
                        if track is not None
                        else group.canonical_description
                    ),
                    member_ids=[track_id],
                    vlm_verified=True,
                )
            )

    updates: dict[str, object] = {
        "verified_groups": updated_groups,
        "final_groups": updated_groups,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"VLM verification produced {len(updated_groups)} groups.",
        ),
    }
    if warnings != state.get("warnings", []):
        updates["warnings"] = warnings
    return updates


def _normalize_confirmed_groups(
    response: VLMVerifyResponse,
    group_size: int,
) -> list[list[int]]:
    confirmed = response.confirmed_groups
    if not confirmed:
        return [[index] for index in range(group_size)]

    normalized: list[list[int]] = []
    for sub_group in confirmed:
        valid_indices = [
            member_index for member_index in sub_group if 0 <= member_index < group_size
        ]
        if valid_indices:
            normalized.append(valid_indices)

    if normalized:
        return normalized
    return [[index] for index in range(group_size)]
