from importlib import resources
from typing import Literal

assert isinstance(__package__, str), "Expected __package__ to be a string"
BASE_PATH = f"{__package__}.prompts"


def _normalize_prompt_name(prompt_name: str) -> str:
    return prompt_name.lower().replace(" ", "_").replace("-", "_").strip()


def _get_prompt_text(prompt_name: str, *, role: Literal["system", "user"]) -> str:
    """Load a prompt template part from package resources."""

    normalized_name = _normalize_prompt_name(prompt_name)
    prompt_dir = resources.files(f"{BASE_PATH}.{role}")
    return prompt_dir.joinpath(f"{normalized_name}.txt").read_text(encoding="utf-8")


def _get_prompt_parts(prompt_name: str) -> tuple[str, str]:
    """Load the system and user prompt texts for a prompt name."""

    return (
        _get_prompt_text(prompt_name, role="system"),
        _get_prompt_text(prompt_name, role="user"),
    )
