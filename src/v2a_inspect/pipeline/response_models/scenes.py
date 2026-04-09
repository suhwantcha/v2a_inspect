from pydantic import BaseModel, Field
from typing import Optional, List


class TimeRange(BaseModel):
    start: float = Field(
        description="Start time in seconds with 0.1s precision (e.g., 1.3, 4.7, 12.1)"
    )
    end: float = Field(
        description="End time in seconds with 0.1s precision (e.g., 2.8, 6.4, 15.3)"
    )


class SceneObject(BaseModel):
    description: str = Field(
        description="Object or event description. Must include rich details and specific count if applicable."
    )
    time_range: TimeRange = Field(
        description="Time range when this sound appears. For Local tracks, must be within the scene's time range. For Global tracks, must be within the macro-segment's time range."
    )
    pan: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Estimated horizontal position of the sound source on screen. "
            "-1.0 = hard left edge, 0.0 = center, 1.0 = hard right edge. "
            "For Global tracks (music/ambience), always use 0.0."
        ),
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Track group ID assigned post-analysis for temporal consistency",
    )
    canonical_description: Optional[str] = Field(
        default=None, description="Unified canonical description for the group"
    )


class LocalScene(BaseModel):
    """A single visual scene snippet for Local Track extraction."""

    scene_index: int = Field(description="0-based scene index")
    time_range: TimeRange = Field(description="Time range of this scene")
    dialogues: List[SceneObject] = Field(
        description="Dialogue sounds based on moving lips, conversational gestures, or interactions in this scene. Emit empty list if none.",
        default_factory=list,
    )
    sfx: List[SceneObject] = Field(
        description="Sound effects (SFX) caused by visible physical actions or object interactions in this scene. Emit empty list if none.",
        default_factory=list,
    )


class MacroSegment(BaseModel):
    """A group of continuous scenes for Macro-Global Track extraction."""

    segment_index: int = Field(description="0-based segment index")
    time_range: TimeRange = Field(
        description="Time range encompassing multiple continuous scenes"
    )
    music: List[SceneObject] = Field(
        description="Music tracks visible or strongly implied (BGM) in this macro-segment. Emit empty list if none.",
        default_factory=list,
    )
    ambience: List[SceneObject] = Field(
        description="Environmental ambience or atmosphere spanning this macro-segment. Emit empty list if none.",
        default_factory=list,
    )


class VideoSceneAnalysis(BaseModel):
    total_duration: float = Field(description="Total video duration in seconds")
    scenes: List[LocalScene] = Field(
        description="List of raw visual scenes detected in the video for Local Tracks."
    )
    macro_segments: List[MacroSegment] = Field(
        description="List of macro-segments grouping continuous scenes for Global Tracks."
    )
