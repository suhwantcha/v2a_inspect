from __future__ import annotations

from v2a_inspect.workflows.state import InspectState

from ..response_models import RawTrack
from ._shared import append_state_message


def extract_raw_tracks(state: InspectState) -> dict[str, object]:
    """Flatten scene analysis output into ordered raw tracks (Local and Global)."""

    scene_analysis = state.get("scene_analysis")
    if scene_analysis is None:
        raise ValueError("extract_raw_tracks requires 'scene_analysis' in state.")

    tracks: list[RawTrack] = []

    # 1. Extract Local Tracks (Dialogue, SFX)
    for scene in scene_analysis.scenes:
        scene_index = scene.scene_index
        n_dialogues = len(scene.dialogues)
        n_sfx = len(scene.sfx)
        
        for idx, obj in enumerate(scene.dialogues):
            tracks.append(
                RawTrack(
                    track_id=f"s{scene_index}_dlg{idx}",
                    scene_index=scene_index,
                    kind="dialogue",
                    description=obj.description,
                    start=obj.time_range.start,
                    end=obj.time_range.end,
                    obj_index=idx,
                    n_scene_objects=n_dialogues,
                    pan=obj.pan,
                )
            )
            
        for idx, obj in enumerate(scene.sfx):
            tracks.append(
                RawTrack(
                    track_id=f"s{scene_index}_sfx{idx}",
                    scene_index=scene_index,
                    kind="sfx",
                    description=obj.description,
                    start=obj.time_range.start,
                    end=obj.time_range.end,
                    obj_index=idx,
                    n_scene_objects=n_sfx,
                    pan=obj.pan,
                )
            )

    # 2. Extract Global Tracks (Music, Ambience)
    for segment in scene_analysis.macro_segments:
        # Use segment_index as the scene_index for global tracks for now, to satisfy schema types
        segment_index = segment.segment_index
        n_music = len(segment.music)
        n_ambience = len(segment.ambience)
        
        for idx, obj in enumerate(segment.music):
             tracks.append(
                RawTrack(
                    track_id=f"s{segment_index}_mus{idx}",
                    scene_index=segment_index,
                    kind="music",
                    description=obj.description,
                    start=obj.time_range.start,
                    end=obj.time_range.end,
                    obj_index=idx,
                    n_scene_objects=n_music,
                    pan=obj.pan,
                )
            )
             
        for idx, obj in enumerate(segment.ambience):
             tracks.append(
                RawTrack(
                    track_id=f"s{segment_index}_amb{idx}",
                    scene_index=segment_index,
                    kind="ambience",
                    description=obj.description,
                    start=obj.time_range.start,
                    end=obj.time_range.end,
                    obj_index=idx,
                    n_scene_objects=n_ambience,
                    pan=obj.pan,
                )
            )

    return {
        "raw_tracks": tracks,
        "progress_messages": append_state_message(
            state,
            "progress_messages",
            f"Extracted {len(tracks)} raw tracks (Local & Global).",
        ),
    }
