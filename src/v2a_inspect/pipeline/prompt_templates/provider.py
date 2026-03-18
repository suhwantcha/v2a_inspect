from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from langfuse.model import ChatMessageDict

from v2a_inspect.observability.langfuse import (
    LangfusePromptClient,
    fetch_chat_prompt,
    get_langfuse_client,
    sync_chat_prompt,
)
from v2a_inspect.settings import settings

from .utils import _get_prompt_parts

PromptName = Literal[
    "grouping",
    "model_select",
    "scene_analysis_default",
    "scene_analysis_extended",
    "vlm_verify",
]

PROMPT_NAMES: tuple[PromptName, ...] = (
    "grouping",
    "model_select",
    "scene_analysis_default",
    "scene_analysis_extended",
    "vlm_verify",
)


@dataclass(frozen=True)
class ResolvedPrompt:
    name: PromptName
    system_text: str
    user_text: str
    source: Literal["local", "langfuse"]
    langfuse_prompt: LangfusePromptClient | None = None

    def render(self, **kwargs: Any) -> "ResolvedPrompt":
        return ResolvedPrompt(
            name=self.name,
            system_text=self.system_text.format(**kwargs),
            user_text=self.user_text.format(**kwargs),
            source=self.source,
            langfuse_prompt=self.langfuse_prompt,
        )


def get_local_prompt(prompt_name: PromptName) -> ResolvedPrompt:
    system_text, user_text = _get_prompt_parts(prompt_name)
    return ResolvedPrompt(
        name=prompt_name,
        system_text=system_text,
        user_text=user_text,
        source="local",
    )


def iter_local_prompts() -> list[ResolvedPrompt]:
    return [get_local_prompt(prompt_name) for prompt_name in PROMPT_NAMES]


def resolve_prompt(prompt_name: PromptName) -> ResolvedPrompt:
    local_prompt = get_local_prompt(prompt_name)
    backend = settings.prompt_backend

    if backend == "local":
        return local_prompt

    prompt_client = fetch_chat_prompt(
        prompt_name,
        fallback=(
            _build_langfuse_chat_messages(local_prompt) if backend == "auto" else None
        ),
    )
    if prompt_client is None:
        if backend == "auto" and get_langfuse_client() is None:
            return local_prompt
        raise ValueError(
            f"Prompt backend is '{backend}' but Langfuse prompt '{prompt_name}' is unavailable."
        )

    system_text, user_text = _extract_langfuse_chat_parts(prompt_name, prompt_client)

    return ResolvedPrompt(
        name=prompt_name,
        system_text=system_text,
        user_text=user_text,
        source="langfuse",
        langfuse_prompt=prompt_client,
    )


def sync_prompts(*, label: str | None = None) -> list[ResolvedPrompt]:
    synced: list[ResolvedPrompt] = []
    for resolved_prompt in iter_local_prompts():
        prompt_client = sync_chat_prompt(
            name=resolved_prompt.name,
            system_prompt=resolved_prompt.system_text,
            user_prompt=resolved_prompt.user_text,
            label=label,
        )
        system_text, user_text = _extract_langfuse_chat_parts(
            resolved_prompt.name,
            prompt_client,
        )
        synced.append(
            ResolvedPrompt(
                name=resolved_prompt.name,
                system_text=system_text,
                user_text=user_text,
                source="langfuse",
                langfuse_prompt=prompt_client,
            )
        )
    return synced


def _build_langfuse_chat_messages(prompt: ResolvedPrompt) -> list[ChatMessageDict]:
    messages: list[ChatMessageDict] = []
    if prompt.system_text.strip():
        messages.append(ChatMessageDict(role="system", content=prompt.system_text))
    messages.append(ChatMessageDict(role="user", content=prompt.user_text))
    return messages


def _extract_langfuse_chat_parts(
    prompt_name: PromptName,
    prompt_client: LangfusePromptClient,
) -> tuple[str, str]:
    prompt_body = prompt_client.prompt
    if not isinstance(prompt_body, list):
        raise TypeError(
            f"Prompt '{prompt_name}' must be a chat prompt, but Langfuse returned a non-chat payload."
        )

    messages = cast(list[dict[str, Any]], prompt_body)
    if len(messages) == 1:
        message = messages[0]
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            raise TypeError(
                f"Prompt '{prompt_name}' must contain exactly one user message or one system message followed by one user message."
            )
        return "", cast(str, message["content"])

    if len(messages) == 2:
        system_message, user_message = messages
        if (
            system_message.get("role") == "system"
            and isinstance(system_message.get("content"), str)
            and user_message.get("role") == "user"
            and isinstance(user_message.get("content"), str)
        ):
            return cast(str, system_message["content"]), cast(
                str, user_message["content"]
            )

    raise TypeError(
        f"Prompt '{prompt_name}' must contain exactly one user message or one system message followed by one user message."
    )
