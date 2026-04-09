from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import google.genai as genai
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from v2a_inspect.observability import start_observation

from v2a_inspect.pipeline.nodes import (
    analyze_scenes,
    assemble_grouped_analysis,
    build_relation_graph,
    evaluate_audio,
    extract_director_intent,
    extract_raw_tracks,
    generate_audio_plan,
    generate_audio_tracks,
    group_tracks,
    mix_video_tracks,
    refine_audio_plan,
    upload_video,
)
from v2a_inspect.pipeline.response_models import VideoSceneAnalysis

from .state import InspectOptions, InspectState


@dataclass(frozen=True)
class InspectRuntime:
    """Runtime dependencies for the inspect workflow."""

    llm: BaseChatModel
    genai_client: genai.Client


def build_initial_inspect_state(
    video_path: str,
    *,
    options: InspectOptions | None = None,
) -> InspectState:
    """Build a fresh initial state for full video inspection."""

    return InspectState(
        video_path=video_path,
        options=options or InspectOptions(),
        errors=[],
        warnings=[],
        progress_messages=[],
    )


def build_state_from_scene_analysis(
    scene_analysis: VideoSceneAnalysis,
    *,
    options: InspectOptions | None = None,
    video_path: str = "",
    gemini_file: Any | None = None,
) -> InspectState:
    """Build state seeded from scene analysis for future grouping-only flows."""

    state = InspectState(
        scene_analysis=scene_analysis,
        options=options or InspectOptions(),
        errors=[],
        warnings=[],
        progress_messages=[],
    )
    if video_path:
        state["video_path"] = video_path
    if gemini_file is not None:
        state["gemini_file"] = gemini_file
    return state


def build_inspect_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: Sequence[str] | None = None,
    interrupt_after: Sequence[str] | None = None,
    debug: bool = False,
) -> CompiledStateGraph:
    """Build the compiled LangGraph workflow for the inspect pipeline."""

    graph = cast(Any, StateGraph)(InspectState, context_schema=InspectRuntime)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("bootstrap",      _bootstrap_node)
    graph.add_node("upload",         _upload_node)
    graph.add_node("intent",         _intent_node)         # [2] Director Intent
    graph.add_node("analyze",        _analyze_node)
    graph.add_node("extract",        _extract_node)
    graph.add_node("group",          _group_node)
    graph.add_node("assemble",       _assemble_node)
    graph.add_node("plan",           _plan_node)           # [4] Audio Plan
    graph.add_node("relation",       _relation_node)       # [5] Relation Graph
    graph.add_node("generate_audio", _generate_audio_node)
    graph.add_node("evaluate",       _evaluate_node)       # [8] Evaluation
    graph.add_node("refine",         _refine_node)         # [8b] Refinement
    graph.add_node("mix_video",      _mix_video_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_edge(START, "bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        _route_after_bootstrap,
        {
            "upload":         "upload",
            "intent":         "intent",
            "analyze":        "analyze",
            "extract":        "extract",
            "generate_audio": "generate_audio",
        },
    )
    graph.add_conditional_edges(
        "upload",
        _route_after_upload,
        {
            "intent":   "intent",
            "analyze":  "analyze",
            "extract":  "extract",
        },
    )
    # intent → analyze (unconditional: intent feeds into analyze via state)
    graph.add_edge("intent",   "analyze")
    graph.add_edge("analyze",  "extract")
    graph.add_edge("extract",  "group")
    graph.add_edge("group",    "assemble")
    # assemble → plan (new): build unified Audio Plan from grouped analysis
    graph.add_conditional_edges(
        "assemble",
        _route_after_assemble,
        {
            "plan":           "plan",
            "generate_audio": "generate_audio",
        },
    )
    # plan → relation (new): build relation graph, then generate
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "relation":       "relation",
            "generate_audio": "generate_audio",
        },
    )
    graph.add_edge("relation",       "generate_audio")
    # generate_audio → evaluate (if enabled) or mix_video (skip)
    graph.add_conditional_edges(
        "generate_audio",
        _route_after_generate_audio,
        {
            "evaluate": "evaluate",
            "mix_video": "mix_video",
        },
    )
    # evaluate → refine (if failed) or mix_video (if passed)
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "refine":    "refine",
            "mix_video": "mix_video",
        },
    )
    # refine → generate_audio (loop back for partial regeneration)
    graph.add_edge("refine",    "generate_audio")
    graph.add_edge("mix_video", END)

    return cast(
        CompiledStateGraph,
        graph.compile(
            checkpointer=checkpointer,
            interrupt_before=list(interrupt_before) if interrupt_before is not None else None,
            interrupt_after=list(interrupt_after) if interrupt_after is not None else None,
            debug=debug,
            name="v2a_inspect_workflow",
        ),
    )


# ── Node wrappers ──────────────────────────────────────────────────────────────

def _bootstrap_node(
    state: InspectState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node("bootstrap", state, lambda: cast(dict[str, object], state))


def _upload_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "upload",
        state,
        lambda: upload_video(state, genai_client=runtime.context.genai_client),
    )


def _intent_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "intent",
        state,
        lambda: extract_director_intent(state, llm=runtime.context.llm, config=config),
    )


def _analyze_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "analyze",
        state,
        lambda: analyze_scenes(state, llm=runtime.context.llm, config=config),
    )


def _extract_node(
    state: InspectState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node("extract", state, lambda: extract_raw_tracks(state))


def _group_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "group",
        state,
        lambda: group_tracks(state, llm=runtime.context.llm, config=config),
    )


def _assemble_node(
    state: InspectState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node("assemble", state, lambda: assemble_grouped_analysis(state))


def _plan_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "plan",
        state,
        lambda: generate_audio_plan(state, llm=runtime.context.llm, config=config),
    )


def _relation_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "relation",
        state,
        lambda: build_relation_graph(state, llm=runtime.context.llm, config=config),
    )


def _generate_audio_node(
    state: InspectState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node("generate_audio", state, lambda: generate_audio_tracks(state))


def _evaluate_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "evaluate",
        state,
        lambda: evaluate_audio(state, llm=runtime.context.llm, config=config),
    )


def _refine_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "refine",
        state,
        lambda: refine_audio_plan(state, llm=runtime.context.llm, config=config),
    )


def _mix_video_node(
    state: InspectState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node("mix_video", state, lambda: mix_video_tracks(state))


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_bootstrap(
    state: InspectState,
) -> Literal["upload", "intent", "analyze", "extract", "generate_audio"]:
    if state.get("grouped_analysis") is not None:
        return "generate_audio"

    if state.get("scene_analysis") is not None:
        if state.get("gemini_file") is not None:
            return "extract"
        if _requires_video_context(state) and state.get("video_path"):
            return "upload"
        return "extract"

    if state.get("gemini_file") is not None:
        # Check if intent is needed
        options = _get_options(state)
        if options.enable_director_intent:
            return "intent"
        return "analyze"

    return "upload"


def _route_after_upload(
    state: InspectState,
) -> Literal["intent", "analyze", "extract"]:
    if state.get("scene_analysis") is not None:
        return "extract"
    options = _get_options(state)
    if options.enable_director_intent:
        return "intent"
    return "analyze"


def _route_after_assemble(
    state: InspectState,
) -> Literal["plan", "generate_audio"]:
    """Go to plan node if audio plan is enabled, otherwise skip to generation."""
    options = _get_options(state)
    if options.enable_audio_plan:
        return "plan"
    return "generate_audio"


def _route_after_plan(
    state: InspectState,
) -> Literal["relation", "generate_audio"]:
    """Go to relation node if enabled, otherwise skip to generation."""
    options = _get_options(state)
    if options.enable_relation_graph:
        return "relation"
    return "generate_audio"


def _route_after_generate_audio(
    state: InspectState,
) -> Literal["evaluate", "mix_video"]:
    """Go to evaluate node if evaluation is enabled, otherwise skip to mixing."""
    options = _get_options(state)
    if options.enable_evaluation:
        return "evaluate"
    return "mix_video"


def _route_after_evaluate(
    state: InspectState,
) -> Literal["refine", "mix_video"]:
    """
    If evaluation passed (or max iterations reached), go to mix.
    Otherwise, go to refine for targeted improvement.
    """
    score = state.get("evaluation_score")
    if score is None or score.passed:
        return "mix_video"
    return "refine"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_options(state: InspectState) -> InspectOptions:
    options = state.get("options")
    if options is None:
        raise ValueError("inspect graph state is missing 'options'.")
    return options


def _requires_video_context(state: InspectState) -> bool:
    options = _get_options(state)
    return options.enable_vlm_verify or options.enable_model_select or options.enable_director_intent


def _run_node(
    node_name: str,
    state: InspectState,
    action: Callable[[], dict[str, object]],
) -> dict[str, object]:
    with start_observation(
        name=f"graph.{node_name}",
        as_type="span",
        input=_summarize_node_input(node_name, state),
        metadata={"node": node_name},
    ) as node_observation:
        try:
            result = action()
        except Exception as exc:  # noqa: BLE001
            if node_observation is not None:
                node_observation.update(
                    output={"error": str(exc)},
                    level="ERROR",
                    status_message=str(exc),
                )
            raise RuntimeError(f"inspect graph failed in '{node_name}': {exc}") from exc

        if node_observation is not None:
            node_observation.update(output=_summarize_node_output(node_name, result))
        return result


def _summarize_node_input(node_name: str, state: InspectState) -> dict[str, object]:
    options = state.get("options")
    video_path = state.get("video_path")
    base: dict[str, object] = {
        "node": node_name,
        "video_name": Path(video_path).name if video_path else None,
    }
    if node_name == "upload":
        if options is not None:
            base.update(
                {
                    "upload_timeout_seconds": options.upload_timeout_seconds,
                    "poll_interval_seconds": options.poll_interval_seconds,
                }
            )
    elif node_name == "intent":
        base["has_gemini_file"] = state.get("gemini_file") is not None
    elif node_name == "analyze":
        if options is not None:
            base.update(
                {
                    "fps": options.fps,
                    "scene_analysis_mode": options.scene_analysis_mode,
                    "gemini_model": options.gemini_model,
                    "has_intent": state.get("director_intent") is not None,
                }
            )
    elif node_name == "extract":
        scene_analysis = state.get("scene_analysis")
        base["scene_count"] = len(scene_analysis.scenes) if scene_analysis else 0
    elif node_name in {"group", "assemble"}:
        base["raw_track_count"] = len(state.get("raw_tracks", []))
    elif node_name == "plan":
        base["grouped_analysis_present"] = state.get("grouped_analysis") is not None
        base["has_intent"] = state.get("director_intent") is not None
    elif node_name in {"verify", "select_model"}:
        base["candidate_group_count"] = _count_active_groups(state)
        base["has_gemini_file"] = state.get("gemini_file") is not None
    else:
        base["state_keys"] = sorted(state.keys())
    return base


def _summarize_node_output(
    node_name: str,
    result: dict[str, object],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "node": node_name,
        "updated_keys": sorted(result.keys()),
    }
    if "director_intent" in result:
        intent = result["director_intent"]
        summary["intent_genre"] = getattr(intent, "genre", None)
        summary["n_emotional_beats"] = len(getattr(intent, "emotional_arc", []))
    if "scene_analysis" in result:
        scene_analysis = cast(VideoSceneAnalysis, result["scene_analysis"])
        summary["scene_count"] = len(scene_analysis.scenes)
    if "raw_tracks" in result:
        summary["raw_track_count"] = len(cast(list[object], result["raw_tracks"]))
    if "text_groups" in result:
        summary["text_group_count"] = len(cast(list[object], result["text_groups"]))
    if "verified_groups" in result:
        summary["verified_group_count"] = len(cast(list[object], result["verified_groups"]))
    if "final_groups" in result:
        summary["final_group_count"] = len(cast(list[object], result["final_groups"]))
    if "grouped_analysis" in result:
        summary["grouped_analysis_ready"] = result.get("grouped_analysis") is not None
    if "audio_plan" in result:
        plan = result["audio_plan"]
        summary["audio_plan_item_count"] = len(getattr(plan, "items", []))
        summary["silence_count"] = sum(
            1 for i in getattr(plan, "items", []) if getattr(i, "type", "") == "silence"
        )
    if "relation_graph" in result:
        rg = result["relation_graph"]
        relations = getattr(rg, "relations", [])
        summary["causes_count"] = sum(1 for r in relations if r.relation == "causes")
        summary["ducks_count"] = sum(1 for r in relations if r.relation == "ducks")
        summary["causal_order_length"] = len(getattr(rg, "causal_order", []))
    if "evaluation_score" in result:
        s = result["evaluation_score"]
        summary["eval_total"] = getattr(s, "total", None)
        summary["eval_passed"] = getattr(s, "passed", None)
        summary["eval_iteration"] = getattr(s, "iteration", None)
        summary["eval_weak_count"] = len(getattr(s, "weak_item_ids", []))
    return summary


def _count_active_groups(state: InspectState) -> int:
    for key in ("final_groups", "verified_groups", "text_groups"):
        groups = state.get(key)
        if groups is not None:
            return len(groups)
    return 0
