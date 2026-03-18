from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def save_uploaded_file(uploaded_file: Any) -> str:
    temp_dir = tempfile.mkdtemp(prefix="v2a_inspect_upload_")
    safe_name = "".join(
        character
        for character in Path(uploaded_file.name).name
        if character.isalnum() or character in "._-"
    )
    if not safe_name:
        safe_name = "video.mp4"

    file_path = os.path.join(temp_dir, safe_name)
    with open(file_path, "wb") as file_obj:
        file_obj.write(uploaded_file.getbuffer())
    return file_path


def extract_clip(
    video_path: str, start: float, end: float, clip_dir: str
) -> str | None:
    out_path = os.path.join(clip_dir, f"clip_{start:.3f}_{end:.3f}.mp4")
    if os.path.exists(out_path):
        return out_path

    try:
        from moviepy import VideoFileClip

        with VideoFileClip(video_path) as source:
            duration = source.duration
            actual_start = max(0.0, start)
            actual_end = min(end, duration)
            if actual_start >= actual_end:
                return None

            clip = source.subclipped(actual_start, actual_end)
            clip.write_videofile(out_path, logger=None, audio=False)
        return out_path
    except Exception:
        return None


def validate_video_file(path: str) -> bool:
    try:
        with open(path, "rb") as file_obj:
            header = file_obj.read(12)
        return (
            b"ftyp" in header
            or header[:4] == b"RIFF"
            or header[:4] == b"\x1a\x45\xdf\xa3"
        )
    except OSError:
        return False


def get_video_duration(video_path: str) -> float | None:
    try:
        from moviepy import VideoFileClip

        with VideoFileClip(video_path) as clip:
            return clip.duration
    except Exception:
        return None
