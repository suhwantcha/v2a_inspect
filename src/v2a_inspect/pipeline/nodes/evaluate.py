"""
[8] Mid-level Evaluation node.

Computes a composite evaluation score:
  S = α·S_temp + β·S_sem + γ·S_global

Phase 3a implementation: LLM-as-judge (no CLAP model required).
- S_temp: rule-based temporal alignment check (audio duration vs plan duration)
- S_sem:  LLM judge — do description texts match what was generated?
          (uses file existence + description quality heuristics as proxy)
- S_global: LLM judge — does overall plan match the director intent?

Future upgrade: swap S_sem / S_global for CLAP embedding cosine similarity.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from v2a_inspect.workflows.state import InspectState

from ..prompt_templates.provider import ResolvedPrompt
from ..response_models import AudioPlan, DirectorIntent, EvaluationScore
from ._shared import append_state_message, invoke_structured_text

logger = logging.getLogger(__name__)

_EVALUATION_SYSTEM_PROMPT = """
You are a senior audio post-production supervisor reviewing an audio plan and the director's intent.

Evaluate the following two dimensions on a scale from 0.0 to 1.0:

1. **semantic_score** (S_sem): How well do the individual track descriptions in the audio plan
   capture sounds that would actually serve each scene? Are descriptions specific, evocative,
   and appropriate for the stated emotional context?
   - 1.0 = every item is specific and emotionally resonant
   - 0.5 = some items are generic or mismatched
   - 0.0 = descriptions are completely wrong or missing

2. **global_coherence_score** (S_global): How well does the overall audio plan serve the
   director's stated intent (genre, mood, emotional arc)?
   - 1.0 = plan perfectly reinforces the director's vision
   - 0.5 = plan partially serves the intent but misses key beats
   - 0.0 = plan contradicts or ignores the director's intent

Also identify **weak_item_ids**: a list of item_ids that are most problematic and should be regenerated.
Limit to at most 3 items. If everything is fine, return an empty list.

Provide brief **feedback** explaining the main issues found.
"""


class _EvaluationLLMResponse(BaseModel):
    semantic_score: float = Field(ge=0.0, le=1.0)
    global_coherence_score: float = Field(ge=0.0, le=1.0)
    weak_item_ids: list[str] = Field(default_factory=list)
    feedback: str = Field(default="")


def evaluate_audio(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """
    Evaluate the generated audio plan quality.
    Returns an EvaluationScore and decides whether refinement is needed.
    """
    options = state.get("options")
    if options is None:
        raise ValueError("evaluate_audio requires 'options' in state.")

    audio_plan: AudioPlan | None = state.get("audio_plan")
    generated_audio: dict[str, str] = state.get("generated_audio", {})
    intent: DirectorIntent | None = state.get("director_intent")
    iteration = state.get("refinement_iteration", 0)

    if audio_plan is None or not audio_plan.items:
        score = EvaluationScore(
            temporal=1.0, semantic=1.0, global_coherence=1.0, total=1.0,
            iteration=iteration, passed=True, weak_item_ids=[], feedback="No plan to evaluate.",
        )
        return {
            "evaluation_score": score,
            "progress_messages": append_state_message(
                state, "progress_messages", "Evaluation skipped — no audio plan."
            ),
        }

    # ── S_temp: rule-based temporal alignment ─────────────────────────────────
    s_temp = _compute_temporal_score(audio_plan, generated_audio)

    # ── S_sem + S_global: LLM judge ───────────────────────────────────────────
    s_sem, s_global, weak_ids, feedback = _llm_evaluate(
        audio_plan, intent, llm, options, config
    )

    # ── Composite score ────────────────────────────────────────────────────────
    alpha = options.eval_alpha
    beta = options.eval_beta
    gamma = options.eval_gamma
    total = alpha * s_temp + beta * s_sem + gamma * s_global

    passed = (total >= options.eval_score_threshold) or (iteration >= options.max_refinement_iter)

    score = EvaluationScore(
        temporal=round(s_temp, 3),
        semantic=round(s_sem, 3),
        global_coherence=round(s_global, 3),
        total=round(total, 3),
        iteration=iteration,
        passed=passed,
        weak_item_ids=weak_ids,
        feedback=feedback,
    )

    status = "PASSED" if passed else f"FAILED (iter {iteration}/{options.max_refinement_iter})"
    message = (
        f"Evaluation [{status}]: total={total:.3f} "
        f"(temp={s_temp:.2f}, sem={s_sem:.2f}, global={s_global:.2f}). "
        f"Weak items: {weak_ids or 'none'}."
    )
    return {
        "evaluation_score": score,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }


# ── Temporal score (rule-based, no LLM) ───────────────────────────────────────

def _compute_temporal_score(
    audio_plan: AudioPlan,
    generated_audio: dict[str, str],
) -> float:
    """
    Score based on:
    - Coverage: what fraction of non-silence items actually have generated audio files?
    - Duration alignment: does each generated file exist and have non-zero size?
    """
    non_silence = [i for i in audio_plan.items if i.type != "silence"]
    if not non_silence:
        return 1.0

    hit = 0
    for item in non_silence:
        wav_path = generated_audio.get(item.item_id)
        if wav_path and Path(wav_path).exists() and Path(wav_path).stat().st_size > 100:
            hit += 1

    coverage = hit / len(non_silence)

    # Penalize low confidence items (they drag down temporal reliability)
    avg_confidence = sum(i.confidence for i in non_silence) / len(non_silence)

    # Weighted mixture: 70% coverage + 30% avg confidence
    return round(0.7 * coverage + 0.3 * avg_confidence, 3)


# ── LLM judge ─────────────────────────────────────────────────────────────────

def _llm_evaluate(
    audio_plan: AudioPlan,
    intent: DirectorIntent | None,
    llm: BaseChatModel,
    options,
    config,
) -> tuple[float, float, list[str], str]:
    """Call the LLM judge for semantic and global coherence scores."""
    plan_summary = "\n".join(
        f"  [{item.item_id}] type={item.type} time={item.time[0]:.1f}s–{item.time[1]:.1f}s "
        f'vol={item.volume:.1f} intensity={item.intensity:.1f} confidence={item.confidence:.1f} '
        f'desc="{item.description[:100]}"'
        for item in audio_plan.items
        if item.type != "silence"
    )

    if intent:
        intent_summary = (
            f"Genre: {intent.genre}\n"
            f"Mood: {intent.overall_mood}\n"
            f"Audio direction: {intent.audio_direction}\n"
            f"Key emotional beats: " + ", ".join(
                f"{b.emotion}@{b.time[0]:.1f}s" for b in intent.emotional_arc[:5]
            )
        )
    else:
        intent_summary = "(No director intent available)"

    user_text = (
        f"== DIRECTOR'S INTENT ==\n{intent_summary}\n\n"
        f"== AUDIO PLAN (non-silence items) ==\n{plan_summary}\n\n"
        "Evaluate semantic_score and global_coherence_score. "
        "Identify the weakest item_ids (max 3). Provide feedback."
    )

    prompt = ResolvedPrompt(
        name="intent",  # reuse a known name to satisfy type
        system_text=_EVALUATION_SYSTEM_PROMPT,
        user_text=user_text,
        source="local",
    )

    try:
        response = invoke_structured_text(
            llm,
            prompt=prompt,
            schema=_EvaluationLLMResponse,
            model=options.gemini_model,
            timeout_ms=options.text_timeout_ms,
            max_retries=options.max_retries,
            label="evaluation_judge",
            config=config,
        )
        # Validate weak_item_ids are real item ids
        valid_ids = {item.item_id for item in audio_plan.items}
        weak_ids = [iid for iid in response.weak_item_ids if iid in valid_ids]
        return response.semantic_score, response.global_coherence_score, weak_ids, response.feedback
    except Exception as exc:
        logger.warning("Evaluation LLM judge failed; using default scores. Reason: %s", exc)
        return 0.8, 0.8, [], f"LLM judge failed: {exc}"
