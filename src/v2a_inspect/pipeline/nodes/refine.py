"""
[8b] Refinement node — Uncertainty-aware partial re-generation.

When evaluation score fails, this node:
1. Identifies weak AudioPlanItems (from EvaluationScore.weak_item_ids + low confidence)
2. Asks LLM to produce improved descriptions for those items only
3. Updates audio_plan with the improved items
4. Increments refinement_iteration counter

The updated plan then flows back to generate_audio for partial re-generation.
Only the weak items are regenerated — the rest remain unchanged.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from v2a_inspect.workflows.state import InspectState

from ..prompt_templates.provider import ResolvedPrompt
from ..response_models import AudioPlan, AudioPlanItem, DirectorIntent, EvaluationScore
from ._shared import append_state_message, invoke_structured_text

logger = logging.getLogger(__name__)

_REFINE_SYSTEM_PROMPT = """
You are a senior sound designer performing targeted revisions to an audio plan.

You will receive:
1. The director's intent (genre, mood, audio direction)
2. The evaluation feedback explaining what went wrong
3. A list of weak AudioPlanItems that need improvement

Your task: rewrite the `description` field for each weak item to be:
- More specific and evocative (avoid generic terms)
- Better aligned with the emotional context of its time region
- More likely to produce a high-quality audio generation result

Also adjust `volume`, `intensity`, and `confidence` if appropriate.
DO NOT change `item_id`, `type`, `time`, `pan`, or `track_id`.
Return exactly the same number of items as you received.
"""


class _RefinedItem(BaseModel):
    item_id: str
    description: str
    volume: float = Field(ge=0.0, le=1.0)
    intensity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)


class _RefineResponse(BaseModel):
    refined_items: list[_RefinedItem] = Field(default_factory=list)
    refinement_notes: str = Field(default="")


def refine_audio_plan(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """
    Selectively improve weak AudioPlanItems and prepare for partial re-generation.
    Returns updated audio_plan + incremented refinement_iteration.
    """
    options = state.get("options")
    if options is None:
        raise ValueError("refine_audio_plan requires 'options' in state.")

    audio_plan: AudioPlan | None = state.get("audio_plan")
    score: EvaluationScore | None = state.get("evaluation_score")
    intent: DirectorIntent | None = state.get("director_intent")
    iteration = state.get("refinement_iteration", 0)

    if audio_plan is None:
        return {
            "refinement_iteration": iteration + 1,
            "progress_messages": append_state_message(
                state, "progress_messages", "Refinement skipped — no audio plan."
            ),
        }

    # Determine which items to refine
    weak_ids: set[str] = set()
    if score:
        weak_ids.update(score.weak_item_ids)
    # Also include any items with low confidence (< 0.65)
    for item in audio_plan.items:
        if item.confidence < 0.65 and item.type != "silence":
            weak_ids.add(item.item_id)

    weak_items = [i for i in audio_plan.items if i.item_id in weak_ids]

    if not weak_items:
        logger.info("No weak items found; refinement iteration %d is a no-op.", iteration)
        return {
            "refinement_iteration": iteration + 1,
            "progress_messages": append_state_message(
                state, "progress_messages",
                f"Refinement iter {iteration + 1}: no weak items found, skipping."
            ),
        }

    # Build LLM prompt
    intent_context = _format_intent(intent)
    feedback = score.feedback if score else "General quality improvement needed."
    items_text = "\n".join(
        f"  [{item.item_id}] type={item.type} time={item.time[0]:.1f}s–{item.time[1]:.1f}s "
        f'vol={item.volume:.1f} intensity={item.intensity:.1f} confidence={item.confidence:.1f}\n'
        f'  current_description: "{item.description}"'
        for item in weak_items
    )

    user_text = (
        f"== DIRECTOR'S INTENT ==\n{intent_context}\n\n"
        f"== EVALUATION FEEDBACK ==\n{feedback}\n\n"
        f"== WEAK ITEMS TO IMPROVE ==\n{items_text}\n\n"
        "Rewrite descriptions for each item. Make them more specific and emotionally aligned."
    )

    prompt = ResolvedPrompt(
        name="intent",  # reuse known name to satisfy Literal type
        system_text=_REFINE_SYSTEM_PROMPT,
        user_text=user_text,
        source="local",
    )

    try:
        response = invoke_structured_text(
            llm,
            prompt=prompt,
            schema=_RefineResponse,
            model=options.gemini_model,
            timeout_ms=options.text_timeout_ms,
            max_retries=options.max_retries,
            label=f"refine_iter_{iteration + 1}",
            config=config,
        )
        refined_map = {r.item_id: r for r in response.refined_items}
    except Exception as exc:
        logger.warning("Refine LLM call failed; using original items. Reason: %s", exc)
        refined_map = {}

    # Merge refined items back into the plan (only update weak items)
    updated_items: list[AudioPlanItem] = []
    n_refined = 0
    for item in audio_plan.items:
        refined = refined_map.get(item.item_id)
        if refined and item.item_id in weak_ids:
            updated_items.append(item.model_copy(update={
                "description": refined.description,
                "volume": refined.volume,
                "intensity": refined.intensity,
                "confidence": refined.confidence,
            }))
            n_refined += 1
        else:
            updated_items.append(item)

    new_plan = AudioPlan(
        items=updated_items,
        total_duration=audio_plan.total_duration,
    )

    message = (
        f"Refinement iter {iteration + 1}: "
        f"improved {n_refined}/{len(weak_items)} weak items. "
        f"Notes: {response.refinement_notes[:120] if refined_map else 'LLM call failed.'}"
    )
    return {
        "audio_plan": new_plan,
        "refinement_iteration": iteration + 1,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }


def _format_intent(intent: DirectorIntent | None) -> str:
    if intent is None:
        return "(no director intent)"
    beats = ", ".join(
        f"{b.emotion}@{b.time[0]:.1f}s" for b in intent.emotional_arc[:5]
    )
    return (
        f"Genre: {intent.genre}\n"
        f"Mood: {intent.overall_mood}\n"
        f"Direction: {intent.audio_direction}\n"
        f"Key beats: {beats}"
    )
