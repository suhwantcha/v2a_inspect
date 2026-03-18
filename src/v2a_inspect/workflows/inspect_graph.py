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
    extract_raw_tracks,
    group_tracks,
    select_models,
    upload_video,
    verify_groups,
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

    graph.add_node("bootstrap", _bootstrap_node)
    graph.add_node("upload", _upload_node)
    graph.add_node("analyze", _analyze_node)
    graph.add_node("extract", _extract_node)
    graph.add_node("group", _group_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("select_model", _select_model_node)
    graph.add_node("assemble", _assemble_node)

    graph.add_edge(START, "bootstrap")
    graph.add_conditional_edges(
        "bootstrap",
        _route_after_bootstrap,
        {
            "upload": "upload",
            "analyze": "analyze",
            "extract": "extract",
        },
    )
    graph.add_conditional_edges(
        "upload",
        _route_after_upload,
        {
            "analyze": "analyze",
            "extract": "extract",
        },
    )
    graph.add_edge("analyze", "extract")
    graph.add_edge("extract", "group")
    graph.add_conditional_edges(
        "group",
        _route_after_group,
        {
            "verify": "verify",
            "select_model": "select_model",
            "assemble": "assemble",
        },
    )
    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "select_model": "select_model",
            "assemble": "assemble",
        },
    )
    graph.add_edge("select_model", "assemble")
    graph.add_edge("assemble", END)

    return cast(
        CompiledStateGraph,
        graph.compile(
            checkpointer=checkpointer,
            interrupt_before=list(interrupt_before)
            if interrupt_before is not None
            else None,
            interrupt_after=list(interrupt_after)
            if interrupt_after is not None
            else None,
            debug=debug,
            name="v2a_inspect_workflow",
        ),
    )


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


def _bootstrap_node(
    state: InspectState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node("bootstrap", state, lambda: cast(dict[str, object], state))


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


def _verify_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "verify",
        state,
        lambda: verify_groups(state, llm=runtime.context.llm, config=config),
    )


def _select_model_node(
    state: InspectState,
    runtime: Runtime[InspectRuntime],
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node(
        "select_model",
        state,
        lambda: select_models(state, llm=runtime.context.llm, config=config),
    )


def _assemble_node(
    state: InspectState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    return _run_node("assemble", state, lambda: assemble_grouped_analysis(state))


def _route_after_group(
    state: InspectState,
) -> Literal["verify", "select_model", "assemble"]:
    options = _get_options(state)
    if options.enable_vlm_verify:
        return "verify"
    if options.enable_model_select:
        return "select_model"
    return "assemble"


def _route_after_bootstrap(
    state: InspectState,
) -> Literal["upload", "analyze", "extract"]:
    if state.get("scene_analysis") is not None:
        if state.get("gemini_file") is not None:
            return "extract"
        if _requires_video_context(state) and state.get("video_path"):
            return "upload"
        return "extract"

    if state.get("gemini_file") is not None:
        return "analyze"
    return "upload"


def _route_after_upload(state: InspectState) -> Literal["analyze", "extract"]:
    if state.get("scene_analysis") is not None:
        return "extract"
    return "analyze"


def _route_after_verify(
    state: InspectState,
) -> Literal["select_model", "assemble"]:
    if _get_options(state).enable_model_select:
        return "select_model"
    return "assemble"


def _get_options(state: InspectState) -> InspectOptions:
    options = state.get("options")
    if options is None:
        raise ValueError("inspect graph state is missing 'options'.")
    return options


def _requires_video_context(state: InspectState) -> bool:
    options = _get_options(state)
    return options.enable_vlm_verify or options.enable_model_select


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
    elif node_name == "analyze":
        if options is not None:
            base.update(
                {
                    "fps": options.fps,
                    "scene_analysis_mode": options.scene_analysis_mode,
                    "gemini_model": options.gemini_model,
                }
            )
    elif node_name == "extract":
        scene_analysis = state.get("scene_analysis")
        base["scene_count"] = len(scene_analysis.scenes) if scene_analysis else 0
    elif node_name in {"group", "assemble"}:
        base["raw_track_count"] = len(state.get("raw_tracks", []))
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
    if "scene_analysis" in result:
        scene_analysis = cast(VideoSceneAnalysis, result["scene_analysis"])
        summary["scene_count"] = len(scene_analysis.scenes)
    if "raw_tracks" in result:
        summary["raw_track_count"] = len(cast(list[object], result["raw_tracks"]))
    if "text_groups" in result:
        summary["text_group_count"] = len(cast(list[object], result["text_groups"]))
    if "verified_groups" in result:
        summary["verified_group_count"] = len(
            cast(list[object], result["verified_groups"])
        )
    if "final_groups" in result:
        summary["final_group_count"] = len(cast(list[object], result["final_groups"]))
    if "grouped_analysis" in result:
        summary["grouped_analysis_ready"] = result.get("grouped_analysis") is not None
    return summary


def _count_active_groups(state: InspectState) -> int:
    for key in ("final_groups", "verified_groups", "text_groups"):
        groups = state.get(key)
        if groups is not None:
            return len(groups)
    return 0
