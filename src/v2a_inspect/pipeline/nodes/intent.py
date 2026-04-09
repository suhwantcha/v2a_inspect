"""
[2] Director Intent node — Top-Down Anchor.

Extracts the director's intended emotional arc from the full video.
This result is injected into the analyze node to condition local/global analysis.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from v2a_inspect.workflows.state import InspectState

from ..prompt_templates import resolve_prompt
from ..response_models import DirectorIntent
from ._shared import append_state_message, invoke_structured_video

logger = logging.getLogger(__name__)


def extract_director_intent(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """
    Watch the full video and extract the director's intended emotional arc.
    The resulting DirectorIntent becomes the 'north star' for all downstream nodes.
    """
    options = state.get("options")
    if options is None:
        raise ValueError("extract_director_intent requires 'options' in state.")

    gemini_file = state.get("gemini_file")
    if gemini_file is None:
        raise ValueError("extract_director_intent requires 'gemini_file' in state.")

    prompt = resolve_prompt("intent")

    try:
        intent = invoke_structured_video(
            llm,
            file_obj=gemini_file,
            fps=options.fps,
            prompt=prompt,
            schema=DirectorIntent,
            model=options.gemini_model,
            timeout_ms=options.video_timeout_ms,
            max_retries=options.max_retries,
            label="director_intent",
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: if intent extraction fails, analysis continues without conditioning.
        logger.warning("Director intent extraction failed; proceeding without it. Reason: %s", exc)
        return {
            "warnings": [*state.get("warnings", []), f"Director intent extraction failed: {exc}"],
            "progress_messages": append_state_message(
                state,
                "progress_messages",
                "Director intent extraction failed — analysis will proceed unconditioned.",
            ),
        }

    beat_summary = ", ".join(
        f"{b.emotion}@{b.time[0]:.1f}s" for b in intent.emotional_arc[:4]
    )
    message = (
        f"Director intent extracted: genre={intent.genre!r}, "
        f"mood={intent.overall_mood!r}, "
        f"{len(intent.emotional_arc)} emotional beats ({beat_summary}{'...' if len(intent.emotional_arc) > 4 else ''})."
    )
    return {
        "director_intent": intent,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }
