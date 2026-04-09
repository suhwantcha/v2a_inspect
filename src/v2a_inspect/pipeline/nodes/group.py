from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from ..prompt_templates import resolve_prompt
from ..response_models import (
    GroupingResponse,
    TrackGroup,
)
from v2a_inspect.workflows.state import InspectState

from ._shared import (
    append_state_message,
    build_grouping_numbered_list,
    invoke_structured_text,
)


def group_tracks(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Group Local raw tracks by sound similarity using Gemini text analysis. Pass Global tracks directly."""

    options = state.get("options")
    if options is None:
        raise ValueError("group_tracks requires 'options' in state.")

    raw_tracks = state.get("raw_tracks")
    if raw_tracks is None:
        raise ValueError("group_tracks requires 'raw_tracks' in state.")

    if not raw_tracks:
        return {
            "text_groups": [],
            "final_groups": [],
            "progress_messages": append_state_message(
                state,
                "progress_messages",
                "Skipped text grouping because there are no raw tracks.",
            ),
        }

    # Separate tracks into local (to group) and global (to bypass)
    local_indices = []
    global_indices = []
    
    for i, track in enumerate(raw_tracks):
        if track.kind in ("music", "ambience"):
            global_indices.append(i)
        else:
            local_indices.append(i)
            
    local_tracks = [raw_tracks[i] for i in local_indices]

    groups: list[TrackGroup] = []
    warnings = list(state.get("warnings", []))
    
    # 1. Bypass Global Tracks (singleton groups)
    for pos, g_idx in enumerate(global_indices):
        track = raw_tracks[g_idx]
        groups.append(
            TrackGroup(
                group_id=f"g_global_{pos}",
                canonical_description=track.description,
                member_ids=[track.track_id],
                vlm_verified=True, # Bypass VLM verify since they are singletons
            )
        )

    # 2. Group Local Tracks
    if not local_tracks:
         updates: dict[str, object] = {
            "text_groups": groups,
            "final_groups": groups,
            "progress_messages": append_state_message(
                state,
                "progress_messages",
                f"Skipped text grouping. Only {len(global_indices)} global tracks bypassed.",
            ),
        }
         return updates

    resolved_prompt = resolve_prompt("grouping").render(
        numbered_list=build_grouping_numbered_list(local_tracks)
    )

    try:
        response = invoke_structured_text(
            llm,
            prompt=resolved_prompt,
            schema=GroupingResponse,
            model=options.gemini_model,
            timeout_ms=options.text_timeout_ms,
            max_retries=options.max_retries,
            label="text_grouping",
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        response = GroupingResponse()
        warnings.append(
            f"Text grouping failed; falling back to singleton groups. Reason: {exc}"
        )

    index_groups = _parse_grouping_response(response, len(local_tracks))
    canonical_map = _extract_canonical_indices(response, index_groups)

    global_offset = len(groups)
    
    for position, member_local_indices in enumerate(index_groups):
        canonical_local_index = canonical_map.get(position, member_local_indices[0])
        canonical_description = local_tracks[canonical_local_index].description
        groups.append(
            TrackGroup(
                group_id=f"g{global_offset + position}",
                canonical_description=canonical_description,
                member_ids=[local_tracks[i].track_id for i in member_local_indices],
                vlm_verified=False,
            )
        )

    updates = {
        "text_groups": groups,
        "final_groups": groups,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Grouped {len(local_tracks)} local tracks into {len(groups) - global_offset} text groups. "
            f"Bypassed {len(global_indices)} global tracks.",
        ),
    }
    if warnings != state.get("warnings", []):
        updates["warnings"] = warnings
    return updates


def _parse_grouping_response(
    response: GroupingResponse,
    num_tracks: int,
) -> list[list[int]]:
    seen: set[int] = set()
    parsed: list[list[int]] = []

    for group in response.groups:
        valid_members = [
            member_index
            for member_index in group.member_indices
            if 0 <= member_index < num_tracks and member_index not in seen
        ]
        if valid_members:
            parsed.append(valid_members)
            seen.update(valid_members)

    for index in range(num_tracks):
        if index not in seen:
            parsed.append([index])

    return parsed


def _extract_canonical_indices(
    response: GroupingResponse,
    groups_by_members: list[list[int]],
) -> dict[int, int]:
    member_to_canonical: dict[frozenset[int], int] = {}
    for group in response.groups:
        if group.canonical_index is None:
            continue
        member_to_canonical[frozenset(group.member_indices)] = group.canonical_index

    canonical_map: dict[int, int] = {}
    for position, members in enumerate(groups_by_members):
        candidate = member_to_canonical.get(frozenset(members))
        if candidate in members:
            canonical_map[position] = candidate
        else:
            canonical_map[position] = members[0]
    return canonical_map
