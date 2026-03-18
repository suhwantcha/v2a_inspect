from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from v2a_inspect.workflows.state import InspectState

from ..response_models import VideoSceneAnalysis
from ._shared import (
    append_state_message,
    get_scene_analysis_prompt,
    invoke_structured_video,
)


def analyze_scenes(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Run Gemini scene analysis for the uploaded video."""

    options = state.get("options")
    if options is None:
        raise ValueError("analyze_scenes requires 'options' in state.")

    gemini_file = state.get("gemini_file")
    if gemini_file is None:
        raise ValueError("analyze_scenes requires 'gemini_file' in state.")

    resolved_prompt = get_scene_analysis_prompt(options)
    scene_analysis = invoke_structured_video(
        llm,
        file_obj=gemini_file,
        fps=options.fps,
        prompt=resolved_prompt,
        schema=VideoSceneAnalysis,
        model=options.gemini_model,
        timeout_ms=options.video_timeout_ms,
        max_retries=options.max_retries,
        label=f"scene_analysis_{options.scene_analysis_mode}",
        config=config,
    )
    message = f"Analyzed {len(scene_analysis.scenes)} scenes with Gemini."
    return {
        "scene_analysis": scene_analysis,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }
