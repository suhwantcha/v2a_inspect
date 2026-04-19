from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import List

from v2a_inspect.workflows.state import InspectState

from ..response_models import VideoSceneAnalysis
from ..prompt_templates import resolve_prompt
from ..prompt_templates.provider import ResolvedPrompt
from ..response_models.scenes import LocalScene, MacroSegment
from ._shared import (
    append_state_message,
    invoke_structured_video,
)


class _LocalAnalysisResult(BaseModel):
    total_duration: float = Field(description="Total video duration in seconds")
    scenes: List[LocalScene] = Field(description="List of scenes detected in the video")


class _GlobalAnalysisResult(BaseModel):
    macro_segments: List[MacroSegment] = Field(description="List of macro-segments")


def _build_intent_prefix(state: InspectState) -> str:
    """Build an intent-context prefix to prepend to analysis prompts."""
    intent = state.get("director_intent")
    if intent is None:
        return ""
    beats_text = "\n".join(
        f"  [{b.time[0]:.1f}s-{b.time[1]:.1f}s] {b.emotion} (intensity={b.intensity:.1f}"
        + (", KEY MOMENT" if b.key_moment else "") + ")"
        for b in intent.emotional_arc
    )
    return (
        "=== DIRECTOR'S INTENT (use this to guide your analysis) ===\n"
        f"Genre: {intent.genre}\n"
        f"Overall mood: {intent.overall_mood}\n"
        f"Audio direction: {intent.audio_direction}\n"
        f"Emotional arc:\n{beats_text}\n"
        "Extract sounds that SERVE this emotional direction, "
        "not just sounds that literally match the visuals.\n"
        "=== END OF DIRECTOR'S INTENT ===\n\n"
    )


def analyze_scenes(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Run Gemini scene analysis for Local (Dialogue/SFX) and Global (Music/Ambience) tracks."""

    options = state.get("options")
    if options is None:
        raise ValueError("analyze_scenes requires 'options' in state.")

    gemini_file = state.get("gemini_file")
    if gemini_file is None:
        raise ValueError("analyze_scenes requires 'gemini_file' in state.")

    intent_prefix = _build_intent_prefix(state)

    # 1. Build prompts (with optional intent prefix)
    prompt_local = resolve_prompt("analyze_local")
    prompt_global = resolve_prompt("analyze_global")
    if intent_prefix:
        prompt_local = ResolvedPrompt(
            name=prompt_local.name,
            system_text=intent_prefix + prompt_local.system_text,
            user_text=prompt_local.user_text,
            source=prompt_local.source,
            langfuse_prompt=prompt_local.langfuse_prompt,
        )
        prompt_global = ResolvedPrompt(
            name=prompt_global.name,
            system_text=intent_prefix + prompt_global.system_text,
            user_text=prompt_global.user_text,
            source=prompt_global.source,
            langfuse_prompt=prompt_global.langfuse_prompt,
        )

    # 2. Run Local & Global analysis in parallel (each is a blocking Gemini API call)
    def _run_local() -> _LocalAnalysisResult:
        return invoke_structured_video(
            llm,
            file_obj=gemini_file,
            fps=options.fps,
            prompt=prompt_local,
            schema=_LocalAnalysisResult,
            model=options.gemini_model,
            timeout_ms=options.video_timeout_ms,
            max_retries=options.max_retries,
            label="analyze_local_tracks",
            config=config,
        )

    def _run_global() -> _GlobalAnalysisResult:
        return invoke_structured_video(
            llm,
            file_obj=gemini_file,
            fps=options.fps,
            prompt=prompt_global,
            schema=_GlobalAnalysisResult,
            model=options.gemini_model,
            timeout_ms=options.video_timeout_ms,
            max_retries=options.max_retries,
            label="analyze_global_tracks",
            config=config,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_local = executor.submit(_run_local)
        future_global = executor.submit(_run_global)
        local_analysis = future_local.result()
        global_analysis = future_global.result()

    # Merging the two analysis branches into the final schema
    scene_analysis = VideoSceneAnalysis(
        total_duration=local_analysis.total_duration,
        scenes=local_analysis.scenes,
        macro_segments=global_analysis.macro_segments,
    )

    intent_note = " (intent-conditioned)" if intent_prefix else ""
    message = (
        f"Analyzed {len(scene_analysis.scenes)} local scenes and "
        f"{len(scene_analysis.macro_segments)} macro-segments{intent_note}."
    )
    return {
        "scene_analysis": scene_analysis,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }
