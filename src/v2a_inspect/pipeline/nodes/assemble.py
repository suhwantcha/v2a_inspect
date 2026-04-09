from __future__ import annotations

from v2a_inspect.workflows.state import InspectState

from ..response_models import GroupedAnalysis
from ._shared import append_state_message, get_active_groups


def assemble_grouped_analysis(state: InspectState) -> dict[str, object]:
    """Build the final grouped analysis and annotate the copied scene analysis."""

    scene_analysis = state.get("scene_analysis")
    if scene_analysis is None:
        raise ValueError(
            "assemble_grouped_analysis requires 'scene_analysis' in state."
        )

    raw_tracks = list(state.get("raw_tracks", []))
    groups = [group.model_copy(deep=True) for group in get_active_groups(state)]

    track_to_group: dict[str, str] = {}
    track_to_canonical: dict[str, str] = {}
    for group in groups:
        for track_id in group.member_ids:
            track_to_group[track_id] = group.group_id
            track_to_canonical[track_id] = group.canonical_description

    annotated_scene_analysis = scene_analysis.model_copy(deep=True)
    
    # Annotate Local Scenes
    for scene in annotated_scene_analysis.scenes:
        scene_index = scene.scene_index
        for idx, obj in enumerate(scene.dialogues):
            track_id = f"s{scene_index}_dlg{idx}"
            obj.group_id = track_to_group.get(track_id)
            obj.canonical_description = track_to_canonical.get(track_id)
            
        for idx, obj in enumerate(scene.sfx):
            track_id = f"s{scene_index}_sfx{idx}"
            obj.group_id = track_to_group.get(track_id)
            obj.canonical_description = track_to_canonical.get(track_id)

    # Annotate Macro Segments
    for segment in annotated_scene_analysis.macro_segments:
        segment_index = segment.segment_index
        for idx, obj in enumerate(segment.music):
            track_id = f"s{segment_index}_mus{idx}"
            obj.group_id = track_to_group.get(track_id)
            obj.canonical_description = track_to_canonical.get(track_id)
            
        for idx, obj in enumerate(segment.ambience):
            track_id = f"s{segment_index}_amb{idx}"
            obj.group_id = track_to_group.get(track_id)
            obj.canonical_description = track_to_canonical.get(track_id)

    grouped_analysis = GroupedAnalysis(
        scene_analysis=annotated_scene_analysis,
        raw_tracks=raw_tracks,
        groups=groups,
        track_to_group=track_to_group,
    )
    return {
        "grouped_analysis": grouped_analysis,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Assembled grouped analysis with {len(groups)} groups.",
        ),
    }
