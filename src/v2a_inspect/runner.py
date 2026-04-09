from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

from langfuse import propagate_attributes

from v2a_inspect.observability import (
    WorkflowTraceContext,
    build_langgraph_runnable_config,
    create_langfuse_handler,
    start_observation,
)
from v2a_inspect.pipeline.response_models import GroupedAnalysis, VideoSceneAnalysis
from v2a_inspect.runtime import build_inspect_runtime
from v2a_inspect.workflows import (
    InspectOptions,
    InspectRuntime,
    InspectState,
    build_initial_inspect_state,
    build_inspect_graph,
    build_state_from_scene_analysis,
)
from v2a_inspect.workflows.inspect_graph import CompiledStateGraph

ProgressCallback = Callable[[str], None]


def run_inspect(
    video_path: str,
    *,
    options: InspectOptions | None = None,
    runtime: InspectRuntime | None = None,
    graph: CompiledStateGraph | None = None,
    progress_callback: ProgressCallback | None = None,
    warning_callback: ProgressCallback | None = None,
    trace_context: WorkflowTraceContext | None = None,
    interrupt_after: Sequence[str] | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> InspectState:
    """Run the inspect workflow for a video path with optional early interruption."""

    resolved_options = options or InspectOptions()
    initial_state = build_initial_inspect_state(video_path, options=resolved_options)
    return _run_workflow(
        initial_state,
        runtime=runtime,
        graph=graph,
        options=resolved_options,
        progress_callback=progress_callback,
        warning_callback=warning_callback,
        trace_context=trace_context,
        interrupt_after=interrupt_after,
        interrupt_before=interrupt_before,
    )


def run_synthesis(
    state: InspectState,
    *,
    options: InspectOptions | None = None,
    runtime: InspectRuntime | None = None,
    graph: CompiledStateGraph | None = None,
    progress_callback: ProgressCallback | None = None,
    warning_callback: ProgressCallback | None = None,
    trace_context: WorkflowTraceContext | None = None,
) -> InspectState:
    """Run the audio generation and video mixing stage from an existing state."""

    resolved_options = options or InspectOptions()
    # Resume from current state
    return _run_workflow(
        state,
        runtime=runtime,
        graph=graph,
        options=resolved_options,
        progress_callback=progress_callback,
        warning_callback=warning_callback,
        trace_context=trace_context,
    )


def run_group_from_scene_analysis(
    scene_analysis: VideoSceneAnalysis,
    *,
    options: InspectOptions | None = None,
    runtime: InspectRuntime | None = None,
    graph: CompiledStateGraph | None = None,
    video_path: str = "",
    gemini_file: object | None = None,
    progress_callback: ProgressCallback | None = None,
    warning_callback: ProgressCallback | None = None,
    trace_context: WorkflowTraceContext | None = None,
) -> InspectState:
    """Run grouping and optional verification/model selection from scene JSON."""

    resolved_options = options or InspectOptions()
    initial_state = build_state_from_scene_analysis(
        scene_analysis,
        options=resolved_options,
        video_path=video_path,
        gemini_file=gemini_file,
    )
    return _run_workflow(
        initial_state,
        runtime=runtime,
        graph=graph,
        options=resolved_options,
        progress_callback=progress_callback,
        warning_callback=warning_callback,
        trace_context=trace_context,
    )


def get_grouped_analysis(state: InspectState) -> GroupedAnalysis:
    """Extract the final grouped analysis from workflow state."""

    grouped_analysis = state.get("grouped_analysis")
    if grouped_analysis is None:
        raise ValueError("Inspect workflow did not produce 'grouped_analysis'.")
    return grouped_analysis


def _run_workflow(
    initial_state: InspectState,
    *,
    options: InspectOptions,
    runtime: InspectRuntime | None,
    graph: CompiledStateGraph | None,
    progress_callback: ProgressCallback | None,
    warning_callback: ProgressCallback | None,
    trace_context: WorkflowTraceContext | None,
    interrupt_after: Sequence[str] | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> InspectState:
    resolved_graph = graph or build_inspect_graph(
        interrupt_after=interrupt_after,
        interrupt_before=interrupt_before,
    )
    graph_runner = cast(Any, resolved_graph)
    resolved_runtime = runtime or build_inspect_runtime(
        model=options.gemini_model,
        max_retries=options.max_retries,
    )
    resolved_trace_context = trace_context or WorkflowTraceContext(
        source="runtime",
        operation=_detect_operation(initial_state),
    )
    trace_name = f"v2a-inspect.{resolved_trace_context.operation}"
    trace_metadata = {
        "source": resolved_trace_context.source,
        "operation": resolved_trace_context.operation,
        "scene_analysis_mode": options.scene_analysis_mode,
        "fps": options.fps,
        "enable_vlm_verify": options.enable_vlm_verify,
        "enable_model_select": options.enable_model_select,
        "gemini_model": options.gemini_model,
        **resolved_trace_context.metadata,
    }

    last_state: dict[str, object] | None = None
    emitted_progress = 0
    emitted_warnings = 0
    with start_observation(
        name=trace_name,
        as_type="chain",
        input=_summarize_workflow_input(initial_state, resolved_trace_context, options),
        metadata=trace_metadata,
    ) as root_observation:
        trace_attributes_context = (
            propagate_attributes(
                user_id=resolved_trace_context.user_id,
                session_id=resolved_trace_context.session_id,
                tags=[
                    "v2a-inspect",
                    resolved_trace_context.source,
                    resolved_trace_context.operation,
                    *resolved_trace_context.tags,
                ],
                trace_name=trace_name,
            )
            if root_observation is not None
            else nullcontext(None)
        )

        with trace_attributes_context:
            runnable_config = build_langgraph_runnable_config(
                handler=(
                    create_langfuse_handler(
                        trace_id=root_observation.trace_id,
                        parent_observation_id=root_observation.id,
                    )
                    if root_observation is not None
                    else None
                ),
                trace_context=resolved_trace_context,
                run_name="v2a_inspect_workflow",
                metadata={
                    "initial_state_keys": sorted(initial_state.keys()),
                },
            )

            try:
                for state_update in graph_runner.stream(
                    initial_state,
                    config=runnable_config,
                    context=resolved_runtime,
                    stream_mode="values",
                ):
                    if not isinstance(state_update, dict):
                        continue

                    last_state = state_update

                    progress_messages = list(state_update.get("progress_messages", []))
                    if progress_callback is not None:
                        while emitted_progress < len(progress_messages):
                            progress_callback(progress_messages[emitted_progress])
                            emitted_progress += 1
                    else:
                        emitted_progress = len(progress_messages)

                    warnings = list(state_update.get("warnings", []))
                    if warning_callback is not None:
                        while emitted_warnings < len(warnings):
                            warning_callback(warnings[emitted_warnings])
                            emitted_warnings += 1
                    else:
                        emitted_warnings = len(warnings)

                if last_state is None:
                    last_state = cast(
                        dict[str, object],
                        graph_runner.invoke(
                            initial_state,
                            config=runnable_config,
                            context=resolved_runtime,
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                if root_observation is not None:
                    root_observation.update(
                        output={"error": str(exc)},
                        level="ERROR",
                        status_message=str(exc),
                    )
                raise

        final_state = cast(InspectState, last_state or initial_state.copy())
        if root_observation is not None:
            final_state["trace_id"] = root_observation.trace_id
            final_state["root_observation_id"] = root_observation.id
            root_observation.update(output=_summarize_workflow_output(final_state))

    return final_state


def _detect_operation(initial_state: InspectState) -> Literal["analyze", "group"]:
    if initial_state.get("scene_analysis") is not None:
        return "group"
    return "analyze"


def _summarize_workflow_input(
    initial_state: InspectState,
    trace_context: WorkflowTraceContext,
    options: InspectOptions,
) -> dict[str, object]:
    video_path = initial_state.get("video_path")
    return {
        "source": trace_context.source,
        "operation": trace_context.operation,
        "video_name": Path(video_path).name if video_path else None,
        "has_scene_analysis": initial_state.get("scene_analysis") is not None,
        "fps": options.fps,
        "scene_analysis_mode": options.scene_analysis_mode,
        "enable_vlm_verify": options.enable_vlm_verify,
        "enable_model_select": options.enable_model_select,
        "gemini_model": options.gemini_model,
    }


def _summarize_workflow_output(state: InspectState) -> dict[str, object]:
    scene_analysis = state.get("scene_analysis")
    grouped_analysis = state.get("grouped_analysis")
    return {
        "scene_count": len(scene_analysis.scenes) if scene_analysis is not None else 0,
        "raw_track_count": len(state.get("raw_tracks", [])),
        "text_group_count": len(state.get("text_groups", [])),
        "verified_group_count": len(state.get("verified_groups", [])),
        "final_group_count": len(state.get("final_groups", [])),
        "grouped_analysis_group_count": (
            len(grouped_analysis.groups) if grouped_analysis is not None else 0
        ),
        "warning_count": len(state.get("warnings", [])),
        "error_count": len(state.get("errors", [])),
    }
