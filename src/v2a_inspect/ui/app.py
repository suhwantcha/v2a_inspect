from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import streamlit as st

from v2a_inspect.observability import WorkflowTraceContext
from v2a_inspect.runner import get_grouped_analysis, run_inspect
from v2a_inspect.settings import settings
from v2a_inspect.ui.auth import require_authentication
from v2a_inspect.ui.render import (
    render_footer,
    render_page_header,
    render_results,
    render_sidebar,
)
from v2a_inspect.ui.session import (
    ensure_process_resources,
    get_analysis_semaphore,
    get_langfuse_session_id,
    initialize_session_state,
    reset_state,
)
from v2a_inspect.ui.video import (
    get_video_duration,
    save_uploaded_file,
    validate_video_file,
)
from v2a_inspect.workflows import InspectOptions


def main() -> None:
    st.set_page_config(
        page_title="V2A Inspect",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    authenticator = require_authentication()
    ensure_process_resources()
    initialize_session_state()

    render_page_header()
    options = render_sidebar(authenticator)
    render_upload_step(options)

    grouped = st.session_state.get("grouped")
    scene_analysis = st.session_state.get("scene_analysis")
    if grouped is not None and scene_analysis is not None:
        render_results(
            grouped,
            scene_analysis,
            video_path=st.session_state.get("video_path") or "",
            clip_dir=st.session_state.get("clip_dir") or "",
            inspect_state=st.session_state.get("inspect_state"),
        )

    render_footer()


def render_upload_step(options: InspectOptions) -> None:
    st.header("Step 1: 영상 업로드 및 분석")

    uploaded_file = st.file_uploader(
        "영상 파일 선택",
        type=["mp4", "mov", "avi", "mkv"],
        help="MP4, MOV, AVI, MKV 형식 지원 (최대 60초)",
    )

    if uploaded_file is None:
        return

    is_new_video = (
        st.session_state.video_path is None
        or not Path(st.session_state.video_path).exists()
        or Path(st.session_state.video_path).name != uploaded_file.name
    )
    if is_new_video:
        reset_state()
        st.session_state.video_path = save_uploaded_file(uploaded_file)

        if not validate_video_file(st.session_state.video_path):
            st.error("유효한 영상 파일이 아닙니다.")
            reset_state()
            st.stop()

        duration = get_video_duration(st.session_state.video_path)
        if duration is not None and duration > 60.0:
            st.error(f"영상 길이가 {duration:.1f}초입니다. 최대 60초까지 허용됩니다.")
            reset_state()
            st.stop()

    st.video(uploaded_file)

    analyze_disabled = st.session_state.grouped is not None
    if st.button(
        "🎬 Director-First 분석 시작",
        type="primary",
        disabled=analyze_disabled,
        help="이미 분석된 경우 Reset 후 재실행하세요",
    ):
        run_analysis(st.session_state.video_path, options)

    if analyze_disabled:
        state = st.session_state.get("inspect_state")
        active_phases = []
        is_mixed = False
        if state:
            if state.get("director_intent"):  active_phases.append("🎯 Intent")
            if state.get("audio_plan"):        active_phases.append("📋 Plan")
            if state.get("relation_graph"):    active_phases.append("🕸️ Relation")
            if state.get("evaluation_score"):  active_phases.append("🔬 Eval")
            if state.get("mixed_video_path"):  
                active_phases.append("🎬 Mixed")
                is_mixed = True
        phase_str = " · ".join(active_phases) if active_phases else ""
        st.success(f"✅ 분석 완료. {phase_str}  *(재분석 시 Reset 클릭)*")

        if not is_mixed:
            st.warning("분석만 완료되었습니다. '오디오 생성 및 영상 합성' 버튼을 눌러 작업을 완료하세요.")
            if st.button("🎵 오디오 생성 및 영상 합성 (Final Mixing)", type="primary", width="stretch"):
                run_synthesis_flow(state, options)



def run_analysis(video_path: str, options: InspectOptions) -> None:
    clip_dir = tempfile.mkdtemp(prefix="v2a_inspect_clips_")
    st.session_state.clip_dir = clip_dir

    semaphore = get_analysis_semaphore()
    acquired = semaphore.acquire(timeout=settings.ui_analysis_acquire_timeout_seconds)
    if not acquired:
        st.error("서버가 바쁩니다. 잠시 후 다시 시도해주세요.")
        st.stop()

    try:
        with st.status("분석 진행 중...", expanded=True) as status:
            try:
                state = run_inspect(
                    video_path,
                    options=options,
                    progress_callback=status.write,
                    warning_callback=lambda msg: status.write(f"⚠️ {msg}"),
                    trace_context=_build_ui_trace_context(options),
                    interrupt_before=["generate_audio"],
                )

                scene_analysis = state.get("scene_analysis")
                if scene_analysis is None:
                    raise ValueError(
                        "Inspect workflow completed without scene analysis output."
                    )

                grouped = get_grouped_analysis(state)
                st.session_state.inspect_state = state
                st.session_state.scene_analysis = scene_analysis
                st.session_state.grouped = grouped

                # Compose summary message
                msg = (
                    f"✅ 그루핑 완료: {len(grouped.raw_tracks)}개 raw 트랙 → "
                    f"{len(grouped.groups)}개 그룹"
                )
                n_model = sum(1 for t in grouped.raw_tracks if t.model_selection)
                if n_model:
                    msg += f" | 모델 판정 {n_model}개"
                if state.get("director_intent"):
                    intent = state["director_intent"]
                    msg += f" | 🎯 {intent.genre}/{intent.overall_mood}"
                if state.get("audio_plan"):
                    plan = state["audio_plan"]
                    n_sil = sum(1 for i in plan.items if i.type == "silence")
                    msg += f" | 📋 {len(plan.items)}항목(silence {n_sil})"
                if state.get("relation_graph"):
                    rg = state["relation_graph"]
                    msg += f" | 🕸️ {len(rg.relations)}관계"
                if state.get("evaluation_score"):
                    score = state["evaluation_score"]
                    msg += f" | 🔬 {score.total:.2f}({'PASS' if score.passed else 'FAIL'})"

                status.write(msg)
                status.update(label="분석 완료!", state="complete")
                st.rerun()

            except TimeoutError:
                status.update(label="Timeout", state="error")
                st.error("영상 처리 시간이 초과되었습니다. 더 짧은 영상을 사용해주세요.")
            except Exception as exc:  # noqa: BLE001
                status.update(label="분석 실패", state="error")
                st.error(f"오류: {exc}")
                st.code(traceback.format_exc())
    finally:
        semaphore.release()


def run_synthesis_flow(state, options: InspectOptions) -> None:
    from v2a_inspect.runner import run_synthesis

    semaphore = get_analysis_semaphore()
    acquired = semaphore.acquire(timeout=settings.ui_analysis_acquire_timeout_seconds)
    if not acquired:
        st.error("서버가 바쁩니다. 잠시 후 다시 시도해주세요.")
        st.stop()

    try:
        with st.status("오디오 생성 및 영상 합성 진행 중...", expanded=True) as status:
            try:
                new_state = run_synthesis(
                    state,
                    options=options,
                    progress_callback=status.write,
                    warning_callback=lambda msg: status.write(f"⚠️ {msg}"),
                    trace_context=_build_ui_trace_context(options),
                )
                
                st.session_state.inspect_state = new_state
                
                msg = "✅ 오디오 생성 및 분석 파이프라인 합성이 완료되었습니다."
                if new_state.get("evaluation_score"):
                    score = new_state["evaluation_score"]
                    msg += f" | 🔬 Eval: {score.total:.2f} ({'PASS' if score.passed else 'FAIL'})"

                status.write(msg)
                status.update(label="합성 완료!", state="complete")
                st.rerun()

            except TimeoutError:
                status.update(label="Timeout", state="error")
                st.error("처리 시간이 초과되었습니다.")
            except Exception as exc:  # noqa: BLE001
                status.update(label="합성 실패", state="error")
                st.error(f"오류: {exc}")
                st.code(traceback.format_exc())
    finally:
        semaphore.release()


def _build_ui_trace_context(options: InspectOptions) -> WorkflowTraceContext:
    username = st.session_state.get("username")
    tags: list[str] = []
    if options.enable_vlm_verify:      tags.append("vlm-verify")
    if options.enable_model_select:    tags.append("model-select")
    if options.enable_director_intent: tags.append("director-intent")
    if options.enable_audio_plan:      tags.append("audio-plan")
    if options.enable_relation_graph:  tags.append("relation-graph")
    if options.enable_evaluation:      tags.append("evaluation")

    return WorkflowTraceContext(
        source="ui",
        operation="analyze",
        user_id=str(username) if username else None,
        session_id=get_langfuse_session_id(),
        tags=tuple(tags),
        metadata={
            "scene_analysis_mode": options.scene_analysis_mode,
            "fps": options.fps,
            "auth_mode": settings.auth_mode,
            "enable_director_intent": options.enable_director_intent,
            "enable_audio_plan": options.enable_audio_plan,
            "enable_relation_graph": options.enable_relation_graph,
            "enable_evaluation": options.enable_evaluation,
        },
    )


if __name__ == "__main__":
    main()
