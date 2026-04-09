from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from v2a_inspect.clients import DEFAULT_GEMINI_MODEL
from v2a_inspect.pipeline.response_models import (
    AudioPlan,
    DirectorIntent,
    EvaluationScore,
    GroupedAnalysis,
    RawTrack,
    RelationGraph,
    TrackGroup,
    VideoSceneAnalysis,
)


class InspectOptions(BaseModel):
    """User-configurable options for the inspection workflow."""

    fps: float = Field(default=8.0, gt=0.0)
    scene_analysis_mode: Literal["default", "extended"] = "default"
    enable_vlm_verify: bool = True
    enable_model_select: bool = False
    gemini_model: str = DEFAULT_GEMINI_MODEL
    upload_timeout_seconds: int = Field(default=300, ge=1)
    text_timeout_ms: int = Field(default=120_000, ge=1)
    video_timeout_ms: int = Field(default=180_000, ge=1)
    max_retries: int = Field(default=3, ge=0)
    poll_interval_seconds: float = Field(default=2.0, gt=0.0)

    # ── Phase 1: Director Intent + Audio Plan ─────────────────────────────────
    enable_director_intent: bool = True
    enable_audio_plan: bool = True
    silence_pre_key_moment_sec: float = Field(default=0.4, ge=0.0)

    # ── Phase 2: Relation Graph ───────────────────────────────────────────────
    enable_relation_graph: bool = True

    # ── Phase 3: Evaluation + Refinement (off by default) ────────────────────
    enable_evaluation: bool = False
    max_refinement_iter: int = Field(default=2, ge=0)
    eval_score_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    eval_alpha: float = Field(default=0.4, ge=0.0, le=1.0)   # temporal weight
    eval_beta: float = Field(default=0.4, ge=0.0, le=1.0)    # semantic weight
    eval_gamma: float = Field(default=0.2, ge=0.0, le=1.0)   # global coherence weight


class InspectState(TypedDict, total=False):
    """Shared LangGraph state for the inspection workflow."""

    # ── Existing fields (unchanged) ──────────────────────────────────────────
    video_path: str
    options: InspectOptions
    gemini_file: Any
    scene_analysis: VideoSceneAnalysis
    raw_tracks: list[RawTrack]
    text_groups: list[TrackGroup]
    verified_groups: list[TrackGroup]
    final_groups: list[TrackGroup]
    grouped_analysis: GroupedAnalysis
    generated_audio: dict[str, str]
    mixed_video_path: str
    trace_id: str
    root_observation_id: str
    errors: list[str]
    warnings: list[str]
    progress_messages: list[str]

    # ── Phase 1: Director Intent + Audio Plan ─────────────────────────────────
    director_intent: DirectorIntent           # [2] intent node output
    audio_plan: AudioPlan                     # [4] plan node output

    # ── Phase 2: Relation Graph ───────────────────────────────────────────────
    relation_graph: RelationGraph             # [5] relation node output

    # ── Phase 3: Evaluation + Refinement ─────────────────────────────────────
    evaluation_score: EvaluationScore         # [8] evaluate node output
    refinement_iteration: int                 # [8b] loop counter
