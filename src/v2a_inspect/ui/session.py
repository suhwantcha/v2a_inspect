from __future__ import annotations

import glob
import os
import shutil
import tempfile
import threading
import time

import streamlit as st

from v2a_inspect.settings import settings

SESSION_DEFAULTS: tuple[str, ...] = (
    "video_path",
    "scene_analysis",
    "grouped",
    "inspect_state",
    "clip_dir",
)


def initialize_session_state() -> None:
    for key in SESSION_DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = None
    if "model_overrides" not in st.session_state:
        st.session_state["model_overrides"] = {}


def reset_state() -> None:
    clip_dir = st.session_state.get("clip_dir")
    if clip_dir and os.path.isdir(clip_dir):
        shutil.rmtree(clip_dir, ignore_errors=True)

    upload_path = st.session_state.get("video_path")
    if upload_path:
        upload_dir = os.path.dirname(upload_path)
        if "v2a_inspect_upload_" in upload_dir:
            shutil.rmtree(upload_dir, ignore_errors=True)

    for key in SESSION_DEFAULTS:
        st.session_state[key] = None
    st.session_state["model_overrides"] = {}


def ensure_process_resources() -> None:
    cleanup_stale_temp()
    start_cleanup_thread()


@st.cache_resource
def get_analysis_semaphore() -> threading.Semaphore:
    return threading.Semaphore(settings.ui_analysis_concurrency_limit)


def cleanup_stale_temp(
    max_age_seconds: int | None = None,
) -> None:
    resolved_max_age = max_age_seconds or settings.ui_temp_cleanup_max_age_seconds
    now = time.time()
    tmp_base = tempfile.gettempdir()

    for prefix in ("v2a_inspect_upload_", "v2a_inspect_clips_"):
        for directory in glob.glob(os.path.join(tmp_base, prefix + "*")):
            try:
                if (
                    os.path.isdir(directory)
                    and (now - os.path.getmtime(directory)) > resolved_max_age
                ):
                    shutil.rmtree(directory, ignore_errors=True)
            except OSError:
                continue


@st.cache_resource
def start_cleanup_thread() -> threading.Thread:
    def loop() -> None:
        while True:
            time.sleep(settings.ui_cleanup_interval_seconds)
            cleanup_stale_temp()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread
