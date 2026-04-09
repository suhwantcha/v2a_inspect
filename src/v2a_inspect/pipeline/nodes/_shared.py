from __future__ import annotations

from typing import Any, Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from v2a_inspect.observability import start_observation
from v2a_inspect.clients import build_uploaded_video_content_block

from ..prompt_templates import ResolvedPrompt, resolve_prompt
from ..response_models import RawTrack, TrackGroup
from v2a_inspect.workflows.state import InspectOptions, InspectState

T = TypeVar("T", bound=BaseModel)


def append_state_message(
    state: InspectState,
    key: Literal["errors", "warnings", "progress_messages"],
    message: str,
) -> list[str]:
    return [*state.get(key, []), message]





def build_text_messages(prompt: ResolvedPrompt) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if prompt.system_text.strip():
        messages.append(SystemMessage(content=prompt.system_text))
    messages.append(HumanMessage(content=[{"type": "text", "text": prompt.user_text}]))
    return messages


def build_video_messages(
    file_obj: Any,
    *,
    fps: float,
    prompt: ResolvedPrompt,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    if prompt.system_text.strip():
        messages.append(SystemMessage(content=prompt.system_text))
    messages.append(
        HumanMessage(
            content=[
                {"type": "text", "text": prompt.user_text},
                build_uploaded_video_content_block(file_obj, fps=fps),
            ]
        )
    )
    return messages


def build_invoke_kwargs(
    llm: BaseChatModel,
    *,
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    invoke_kwargs: dict[str, Any] = {}

    # Avoid model_copy()-style overrides here: ChatGoogleGenerativeAI copies share
    # the same underlying google-genai client, and copied instances close that client
    # in __del__, which can break later nodes in the same workflow.
    current_model = getattr(llm, "model", None)
    if model is not None and current_model not in (None, model):
        raise ValueError(
            "Per-call model overrides are not supported for the shared Gemini chat "
            f"client. Runtime model is {current_model!r}, requested model is {model!r}."
        )
    if timeout_ms is not None:
        invoke_kwargs["timeout"] = timeout_ms / 1000
    if max_retries is not None:
        invoke_kwargs["max_retries"] = max(1, max_retries)

    return invoke_kwargs


def invoke_structured_text(
    llm: BaseChatModel,
    *,
    prompt: ResolvedPrompt,
    schema: type[T],
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    label: str = "",
    config: RunnableConfig | None = None,
) -> T:
    return _invoke_structured(
        llm=llm,
        messages=build_text_messages(prompt),
        prompt=prompt,
        schema=schema,
        model=model,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        label=label,
        config=config,
    )


def invoke_structured_video(
    llm: BaseChatModel,
    *,
    file_obj: Any,
    fps: float,
    prompt: ResolvedPrompt,
    schema: type[T],
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    label: str = "",
    config: RunnableConfig | None = None,
) -> T:
    return _invoke_structured(
        llm=llm,
        messages=build_video_messages(file_obj, fps=fps, prompt=prompt),
        prompt=prompt,
        schema=schema,
        model=model,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        label=label,
        config=config,
    )


def _invoke_structured(
    llm: BaseChatModel,
    *,
    messages: list[BaseMessage],
    prompt: ResolvedPrompt,
    schema: type[T],
    model: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    label: str = "",
    config: RunnableConfig | None = None,
) -> T:
    invoke_kwargs = build_invoke_kwargs(
        llm,
        model=model,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
    )
    structured_llm = llm.with_structured_output(
        schema,
        method="json_schema",
    )
    span_name = f"llm.{label}" if label else f"llm.{schema.__name__.lower()}"

    with start_observation(
        name=span_name,
        as_type="chain",
        input={
            "label": label or schema.__name__,
            "schema": schema.__name__,
            "prompt_name": prompt.name,
        },
        metadata={
            "schema": schema.__name__,
            "prompt_name": prompt.name,
        },
        prompt=prompt.langfuse_prompt,
        model=getattr(llm, "model", None),
    ) as request_observation:
        try:
            result = structured_llm.invoke(messages, config=config, **invoke_kwargs)
        except Exception as exc:  # noqa: BLE001
            if request_observation is not None:
                request_observation.update(
                    output={"error": str(exc)},
                    level="ERROR",
                    status_message=str(exc),
                )
            error_label = f" for {label}" if label else ""
            raise RuntimeError(f"LLM request failed{error_label}: {exc}") from exc

        if request_observation is not None:
            request_observation.update(
                output={
                    "schema": schema.__name__,
                    "label": label or schema.__name__,
                    "result_type": type(result).__name__,
                }
            )

    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def get_active_groups(state: InspectState) -> list[TrackGroup]:
    final_groups = state.get("final_groups")
    if final_groups is not None:
        return list(final_groups)

    verified_groups = state.get("verified_groups")
    if verified_groups is not None:
        return list(verified_groups)

    text_groups = state.get("text_groups")
    if text_groups is not None:
        return list(text_groups)

    return []


def build_grouping_numbered_list(tracks: list[RawTrack]) -> str:
    return "\n".join(
        f"[{index}] {track.track_id} ({track.kind}, scene {track.scene_index}, "
        f"{track.start:.1f}s-{track.end:.1f}s): {track.description}"
        for index, track in enumerate(tracks)
    )


def build_verify_segment_list(
    group: TrackGroup,
    tracks_by_id: dict[str, RawTrack],
) -> str:
    lines: list[str] = []
    for index, track_id in enumerate(group.member_ids):
        track = tracks_by_id[track_id]
        lines.append(
            f"  Segment {index}: scene {track.scene_index}, "
            f'{track.start:.1f}s-{track.end:.1f}s | "{track.description}"'
        )
    return "\n".join(lines)


def build_model_select_segment_list(member_tracks: list[RawTrack]) -> str:
    return "\n".join(
        f"  Segment {index}: scene {track.scene_index}, {track.start:.1f}s-{track.end:.1f}s"
        f" | kind={track.kind} | n_objects_in_scene={track.n_scene_objects}"
        f' | "{track.description}"'
        for index, track in enumerate(member_tracks)
    )
