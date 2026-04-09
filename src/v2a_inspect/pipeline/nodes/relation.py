"""
[5] Lightweight Relation Graph node.

Extracts 'causes' and 'ducks' relationships between AudioPlanItems.
Then applies topological sort to determine the causal generation order.
"""
from __future__ import annotations

import logging
from graphlib import TopologicalSorter, CycleError

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from v2a_inspect.workflows.state import InspectState

from ..prompt_templates import resolve_prompt
from ..response_models import AudioPlan, LLMRelationResponse, RelationGraph
from ._shared import append_state_message, invoke_structured_text

logger = logging.getLogger(__name__)


def build_relation_graph(
    state: InspectState,
    *,
    llm: BaseChatModel,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """
    Identify 'causes' and 'ducks' relationships between AudioPlanItems via LLM.
    Then compute a topological generation order based on 'causes' edges.
    """
    options = state.get("options")
    if options is None:
        raise ValueError("build_relation_graph requires 'options' in state.")

    audio_plan: AudioPlan | None = state.get("audio_plan")
    if audio_plan is None or not audio_plan.items:
        return {
            "relation_graph": RelationGraph(relations=[], causal_order=[]),
            "progress_messages": append_state_message(
                state, "progress_messages",
                "Skipped relation graph — no audio_plan items.",
            ),
        }

    # Build prompt context
    plan_items_text = _format_plan_for_prompt(audio_plan)
    prompt = resolve_prompt("relation").render(plan_items_text=plan_items_text)

    # LLM call for relation extraction
    relations = []
    warnings = list(state.get("warnings", []))
    try:
        response = invoke_structured_text(
            llm,
            prompt=prompt,
            schema=LLMRelationResponse,
            model=options.gemini_model,
            timeout_ms=options.text_timeout_ms,
            max_retries=options.max_retries,
            label="relation_graph",
            config=config,
        )
        # Validate: only keep relations referencing known item_ids
        valid_ids = {item.item_id for item in audio_plan.items}
        relations = [
            rel for rel in response.relations
            if rel.from_item_id in valid_ids and rel.to_item_id in valid_ids
        ]
        if len(relations) < len(response.relations):
            n_dropped = len(response.relations) - len(relations)
            warnings.append(
                f"Dropped {n_dropped} relation(s) referencing unknown item_ids."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Relation graph LLM call failed; using empty graph. Reason: %s", exc)
        warnings.append(f"Relation graph extraction failed: {exc}")

    # Topological sort of causal_order
    causal_order = _topo_sort(audio_plan, relations)

    graph = RelationGraph(relations=relations, causal_order=causal_order)

    causes_count = sum(1 for r in relations if r.relation == "causes")
    ducks_count = sum(1 for r in relations if r.relation == "ducks")
    message = (
        f"Relation graph built: {causes_count} 'causes' + {ducks_count} 'ducks' edges. "
        f"Causal generation order: {len(causal_order)} items."
    )
    updates: dict[str, object] = {
        "relation_graph": graph,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }
    if warnings != state.get("warnings", []):
        updates["warnings"] = warnings
    return updates


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_plan_for_prompt(audio_plan: AudioPlan) -> str:
    """Render audio plan items as a compact numbered list for the LLM."""
    lines: list[str] = []
    for item in sorted(audio_plan.items, key=lambda x: x.time[0]):
        lines.append(
            f"  [{item.item_id}] type={item.type} "
            f"time={item.time[0]:.1f}s–{item.time[1]:.1f}s "
            f'desc="{item.description[:80]}"'
        )
    return "\n".join(lines) if lines else "(empty)"


def _topo_sort(audio_plan: AudioPlan, relations: list) -> list[str]:
    """
    Topological sort based on 'causes' edges.
    - 'causes' means: from_item must be generated BEFORE to_item.
    - Items with no causal dependencies are sorted by time order.
    - Cycles are detected and handled gracefully (fallback to time order).
    """
    item_ids = [item.item_id for item in audio_plan.items]
    time_index = {item.item_id: item.time[0] for item in audio_plan.items}

    # Build dependency map: deps[X] = set of items that must come BEFORE X
    deps: dict[str, set[str]] = {iid: set() for iid in item_ids}
    for rel in relations:
        if rel.relation == "causes":
            if rel.to_item_id in deps and rel.from_item_id in deps:
                deps[rel.to_item_id].add(rel.from_item_id)

    try:
        ts = TopologicalSorter(deps)
        # static_order() returns a flat, resolved order
        order = list(ts.static_order())
        # Within the same dependency level, sort by time
        return sorted(order, key=lambda iid: (
            sum(1 for dep in deps.get(iid, set()) if dep in order), 
            time_index.get(iid, 0.0)
        ))
    except CycleError as exc:
        logger.warning("Cycle detected in relation graph — falling back to time order. %s", exc)
        return sorted(item_ids, key=lambda iid: time_index.get(iid, 0.0))
