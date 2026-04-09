"""
Pydantic models for the Director Intent, Audio Plan, and Relation Graph stages.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── [2] Director Intent ────────────────────────────────────────────────────────


class EmotionalBeat(BaseModel):
    """A single emotional beat in the director's intended arc."""

    time: tuple[float, float] = Field(
        description="Time range [start, end] in seconds for this emotional beat."
    )
    emotion: str = Field(
        description=(
            "The intended emotional state the audience should feel. "
            "E.g. 'building dread', 'sudden shock', 'quiet relief'."
        )
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Emotional intensity 0.0 (subtle) to 1.0 (peak).",
    )
    key_moment: bool = Field(
        default=False,
        description=(
            "True if this beat is the climactic peak of the scene. "
            "A silence node will be automatically inserted just before this beat."
        ),
    )


class DirectorIntent(BaseModel):
    """Top-down emotional and narrative intent extracted from the full video."""

    genre: str = Field(
        description="Film genre. E.g. 'thriller', 'action', 'drama', 'horror', 'comedy'."
    )
    overall_mood: str = Field(
        description=(
            "2-5 adjectives describing the overall emotional atmosphere. "
            "E.g. 'tense, claustrophobic, hopeless'."
        )
    )
    emotional_arc: list[EmotionalBeat] = Field(
        description="Ordered list of emotional beats that define the intended audience experience.",
        default_factory=list,
    )
    audio_direction: str = Field(
        description=(
            "High-level guidance for how audio should serve the emotional intent. "
            "E.g. 'Use silence strategically. Swell into the climax. "
            "Keep music sparse until the reveal.'"
        )
    )


# ── [4] Audio Plan ─────────────────────────────────────────────────────────────


class AudioPlanItem(BaseModel):
    """A single element in the unified audio timeline."""

    item_id: str = Field(
        description=(
            "Unique identifier. Format: 'plan_{type}_{index}', e.g. 'plan_sfx_0', "
            "'plan_music_1', 'plan_silence_0'."
        )
    )
    type: Literal["sfx", "music", "ambience", "dialogue", "silence"] = Field(
        description="Audio type. 'silence' means no audio should play in this window."
    )
    time: tuple[float, float] = Field(
        description="[start, end] in seconds."
    )
    description: str = Field(
        description=(
            "The final, highly detailed text prompt for the audio generation API, naturally adapted to the track type:\n"
            "- Music: Genre, mood, tempo, and lead instruments.\n"
            "- SFX/Ambience: Physical and acoustic properties (source, texture, space, impact intensity). NO emotional words.\n"
            "- Dialogue: [Voice Profile] + exact script text with emotional tone.\n"
            "- Silence: Dramatic intent or cinematic purpose of the pause (e.g., 'abrupt silence to build tension')."
        )
    )
    volume: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Target volume level. 0.0 = silent, 1.0 = full.",
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Emotional intensity used to condition audio generation strength. "
            "Mapped from the director's emotional arc."
        ),
    )
    pan: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Stereo panning. -1.0 = hard left, 0.0 = center, 1.0 = hard right. "
            "Derived from the sound source's estimated on-screen position."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that this item is correctly specified. "
            "Items with confidence < 0.7 are candidates for refinement."
        ),
    )
    track_id: Optional[str] = Field(
        default=None,
        description="The RawTrack.track_id this item was derived from, if any.",
    )


class AudioPlan(BaseModel):
    """Unified audio timeline for the entire video, intent-conditioned."""

    items: list[AudioPlanItem] = Field(
        description="All audio plan items, including silence windows, ordered by start time.",
        default_factory=list,
    )
    total_duration: float = Field(
        description="Total video duration in seconds.",
        default=0.0,
    )


# ── [5] Lightweight Relation Graph ────────────────────────────────────────────


class AudioRelation(BaseModel):
    """A directed relationship between two AudioPlanItems."""

    from_item_id: str = Field(
        description="The source AudioPlanItem ID."
    )
    to_item_id: str = Field(
        description="The target AudioPlanItem ID."
    )
    relation: Literal["causes", "ducks"] = Field(
        description=(
            "'causes': from_item physically causes to_item — generate from_item first.\n"
            "'ducks': while from_item plays, lower to_item's volume automatically."
        )
    )
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Relation strength. For 'ducks': 1.0 = full duck (-12dB), 0.5 = gentle duck (-6dB)."
        ),
    )


class LLMRelationResponse(BaseModel):
    """Raw LLM output for the relation graph step."""

    relations: list[AudioRelation] = Field(
        description="All identified causes/ducks relationships between plan items.",
        default_factory=list,
    )


class RelationGraph(BaseModel):
    """Processed lightweight relation graph with topological sort info."""

    relations: list[AudioRelation] = Field(default_factory=list)
    causal_order: list[str] = Field(
        default_factory=list,
        description="item_ids sorted in dependency order (causes come before effects).",
    )


# ── [8] Evaluation ────────────────────────────────────────────────────────────


class EvaluationScore(BaseModel):
    """Mid-level evaluation scores for the generated audio."""

    temporal: float = Field(
        default=0.0,
        description="Temporal alignment score (0.0~1.0): how well audio timestamps match video events.",
    )
    semantic: float = Field(
        default=0.0,
        description="Semantic consistency score (0.0~1.0): how well audio descriptions match generated output.",
    )
    global_coherence: float = Field(
        default=0.0,
        description="Global coherence score (0.0~1.0): how well the overall audio matches the director intent.",
    )
    total: float = Field(
        default=0.0,
        description="Weighted composite: α·temporal + β·semantic + γ·global_coherence.",
    )
    iteration: int = Field(
        default=0,
        description="Which refinement iteration produced this score.",
    )
    passed: bool = Field(
        default=True,
        description="True if total >= eval_score_threshold.",
    )
    weak_item_ids: list[str] = Field(
        default_factory=list,
        description="item_ids with confidence < threshold — candidates for refinement.",
    )
    feedback: str = Field(
        default="",
        description="LLM-generated feedback describing specific problems found.",
    )
