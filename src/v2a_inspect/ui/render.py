from __future__ import annotations

from typing import Any, Literal, cast

import streamlit as st

from v2a_inspect.observability import build_score_id, create_trace_score
from v2a_inspect.pipeline.response_models import (
    GroupedAnalysis,
    RawTrack,
    TrackGroup,
    VideoSceneAnalysis,
)
from v2a_inspect.workflows import InspectOptions, InspectState

from .session import reset_state
from .video import extract_clip


def render_page_header() -> None:
    st.title("🔍 V2A Inspect — 트랙 그루핑 검증 시스템")
    st.markdown(
        "Gemini 장면 분석과 크로스씬 트랙 그루핑 결과를 시각화하여 "
        "**사람이 직접 검증**할 수 있는 검사 도구입니다.  \n"
        "오디오 생성 없음 — 분석과 그루핑 단계만 실행합니다."
    )
    st.divider()


def render_sidebar(authenticator: Any) -> InspectOptions:
    with st.sidebar:
        st.header("⚙️ 분석 옵션")

        fps = st.slider(
            "Analysis FPS", min_value=1.0, max_value=5.0, value=2.0, step=0.5
        )
        st.caption("초당 분석 프레임 수. 높을수록 정밀하지만 느림")

        prompt_type = cast(
            Literal["default", "extended"],
            st.selectbox("Prompt Type", ["default", "extended"], index=0),
        )
        st.caption("`default`: 간결 | `extended`: Foley 상세")

        enable_vlm_verify = st.checkbox("VLM 그룹 검증 사용", value=True)
        st.caption("Gemini VLM이 실제 영상 프레임으로 그룹핑 결과를 시각적으로 확인")

        enable_model_select = st.checkbox("TTA/VTA 모델 자동 선정", value=False)
        st.caption(
            "Gemini VLM이 각 씬의 동적 특성(싱크 중요도 vs 트랙 분리 중요도)을 분석하여 "
            "TTA 또는 VTA 모델을 자동 판정"
        )

        st.divider()

        with st.expander("🔄 파이프라인 구조", expanded=True):
            st.markdown(
                """
```
📹 Video Upload
      │
      ▼
🤖 Gemini Scene Analysis
   FPS · Prompt Type
      │
      ▼
 VideoSceneAnalysis
  ├─ Scene 0
  │   ├─ background_sound
  │   └─ objects (≤2)
  └─ Scene N ...
      │
      ▼
🔗 Cross-Scene Text Grouping
   (Gemini batch call)
      │
      ▼  (VLM verify ON)
👁️ VLM Group Verification
   (Gemini + video frames)
      │
      ▼
📦 GroupedAnalysis
  ├─ groups (canonical desc)
  └─ track_assignments
```
"""
            )

        st.divider()

        if st.button("🔄 Reset", use_container_width=True, type="secondary"):
            reset_state()
            st.rerun()

        authenticator.logout("Logout", "sidebar")

    return InspectOptions(
        fps=fps,
        scene_analysis_mode=prompt_type,
        enable_vlm_verify=enable_vlm_verify,
        enable_model_select=enable_model_select,
    )


def render_results(
    grouped: GroupedAnalysis,
    scene_analysis: VideoSceneAnalysis,
    *,
    video_path: str,
    clip_dir: str,
    inspect_state: InspectState | None,
) -> None:
    st.divider()
    st.header("Step 2: 분석 결과 요약")

    _render_state_messages(inspect_state)
    _render_langfuse_summary(inspect_state)

    n_scenes = len(scene_analysis.scenes)
    n_backgrounds = n_scenes
    n_objects = sum(len(scene.objects) for scene in scene_analysis.scenes)
    n_raw = len(grouped.raw_tracks)
    n_groups = len(grouped.groups)
    n_multi = sum(1 for group in grouped.groups if len(group.member_ids) > 1)
    n_verified = sum(1 for group in grouped.groups if group.vlm_verified)
    n_merged = n_raw - n_groups
    n_model_vta = sum(
        1
        for track in grouped.raw_tracks
        if track.model_selection and track.model_selection.model_type == "VTA"
    )
    n_model_tta = sum(
        1
        for track in grouped.raw_tracks
        if track.model_selection and track.model_selection.model_type == "TTA"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🎬 씬 수", n_scenes)
    c2.metric("🌲 배경 트랙", n_backgrounds)
    c3.metric("🎯 객체 트랙", n_objects)
    c4.metric("📦 Raw 트랙 수", n_raw)
    c5.metric(
        "🔗 최종 그룹 수",
        n_groups,
        delta=f"-{n_merged} 병합" if n_merged > 0 else "병합 없음",
        delta_color="normal" if n_merged > 0 else "off",
    )
    c6.metric("✅ VLM 검증 그룹", n_verified)

    if n_model_vta + n_model_tta > 0:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🟢 TTA 트랙", n_model_tta)
        mc2.metric("🔵 VTA 트랙", n_model_vta)
        mc3.metric(
            "⚠️ 그룹 내 이견",
            sum(
                1
                for group in grouped.groups
                if group.model_selection and group.model_selection.confidence < 0.6
            ),
        )

    st.caption(
        f"멀티멤버 그룹 {n_multi}개 (같은 개체로 판단된 트랙들이 하나의 그룹으로 묶임)"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("📋 씬 분석 JSON (raw)", expanded=False):
            st.json(scene_analysis.model_dump())

    with col_right:
        with st.expander("📋 트랙 그룹 JSON", expanded=False):
            st.json(
                {
                    "groups": {
                        group.group_id: {
                            "canonical_description": group.canonical_description,
                            "member_ids": group.member_ids,
                            "vlm_verified": group.vlm_verified,
                        }
                        for group in grouped.groups
                    },
                    "track_assignments": grouped.track_to_group,
                }
            )

    st.divider()
    st.header("Step 3: 그루핑 검증")
    st.caption(
        "각 그룹의 **canonical description**과 멤버별 원본 description·영상 클립을 나란히 배치합니다.  \n"
        "멀티멤버 그룹에서 같은 개체인지 직접 확인하세요."
    )

    tracks_by_id = {track.track_id: track for track in grouped.raw_tracks}
    for group in grouped.groups:
        members = [
            tracks_by_id[member_id]
            for member_id in group.member_ids
            if member_id in tracks_by_id
        ]
        _render_group_expander(
            group=group,
            members=members,
            video_path=video_path,
            clip_dir=clip_dir,
            trace_id=inspect_state.get("trace_id") if inspect_state else None,
        )


def render_footer() -> None:
    st.divider()
    st.caption(
        "V2A Inspect | Gemini Scene Analysis + Cross-Scene Track Grouping | No Audio Generation"
    )


def _render_state_messages(inspect_state: InspectState | None) -> None:
    if inspect_state is None:
        return

    warnings = inspect_state.get("warnings", [])
    progress_messages = inspect_state.get("progress_messages", [])

    if warnings:
        with st.expander("⚠️ 워크플로우 경고", expanded=True):
            for message in warnings:
                st.warning(message)

    if progress_messages:
        with st.expander("🧭 워크플로우 로그", expanded=False):
            for message in progress_messages:
                st.write(f"- {message}")


def _render_group_expander(
    *,
    group: TrackGroup,
    members: list[RawTrack],
    video_path: str,
    clip_dir: str,
    trace_id: str | None,
) -> None:
    if not members:
        st.warning("이 그룹에 표시할 멤버 트랙이 없습니다.")
        return

    is_multi = len(members) > 1

    if group.vlm_verified:
        badge = "✅ VLM 검증됨"
    elif is_multi:
        badge = "🔗 텍스트 그루핑"
    else:
        badge = "⬜ 싱글턴"

    short_desc = (
        group.canonical_description[:60] + "..."
        if len(group.canonical_description) > 60
        else group.canonical_description
    )

    with st.expander(f"{badge}  `{group.group_id}` — {short_desc}", expanded=is_multi):
        hcol_a, hcol_b = st.columns([4, 1])
        with hcol_a:
            st.markdown(
                f"**Canonical description:**  \n> {group.canonical_description}"
            )
        with hcol_b:
            st.markdown(f"**{badge}**")
            st.caption(f"멤버 {len(members)}개")
            if group.model_selection:
                selection = group.model_selection
                model_icon = "🔵" if selection.model_type == "VTA" else "🟢"
                conflict_flag = " ⚠️" if selection.confidence < 0.6 else ""
                rule_tag = " ⚡규칙" if selection.rule_based else ""
                st.markdown(
                    f"{model_icon} **{selection.model_type}**{conflict_flag}{rule_tag}  \n"
                    f"conf: {selection.confidence:.0%}  \n"
                    f"vta={selection.vta_score:.1f} / tta={selection.tta_score:.1f}"
                )
                if selection.confidence < 0.6:
                    st.caption("⚠️ 그룹 내 멤버 간 모델 이견 있음")

            current_override = st.session_state.model_overrides.get(
                group.group_id, "(자동)"
            )
            override = st.selectbox(
                "모델 오버라이드",
                ["(자동)", "TTA", "VTA"],
                index=["(자동)", "TTA", "VTA"].index(current_override),
                key=f"model_override_{group.group_id}",
            )
            if override != current_override:
                st.session_state.model_overrides[group.group_id] = override

            _render_group_review_controls(
                trace_id=trace_id,
                group=group,
                override=override,
            )

        st.markdown("---")

        if not is_multi and members:
            _render_singleton_member(
                members[0], video_path=video_path, clip_dir=clip_dir
            )
            return

        max_cols = min(len(members), 4)
        columns = st.columns(max_cols)
        for index, track in enumerate(members):
            with columns[index % max_cols]:
                _render_member(
                    track, video_path=video_path, clip_dir=clip_dir, heading_level=4
                )


def _render_singleton_member(
    track: RawTrack, *, video_path: str, clip_dir: str
) -> None:
    kind_icon = "🌲" if track.kind == "background" else "🎯"
    st.markdown(
        f"{kind_icon} `{track.track_id}` | "
        f"Scene {track.scene_index} | "
        f"{track.start:.1f}s – {track.end:.1f}s | "
        f"*{track.kind}*"
    )
    st.info(track.description)
    _render_track_model_selection(track)
    _render_track_clip(track, video_path=video_path, clip_dir=clip_dir)


def _render_member(
    track: RawTrack,
    *,
    video_path: str,
    clip_dir: str,
    heading_level: int,
) -> None:
    kind_icon = "🌲" if track.kind == "background" else "🎯"
    st.markdown(f"{'#' * heading_level} {kind_icon} `{track.track_id}`")
    st.caption(
        f"Scene {track.scene_index} | {track.start:.1f}s – {track.end:.1f}s | *{track.kind}*"
    )
    st.info(track.description)
    _render_track_model_selection(track)
    _render_track_clip(track, video_path=video_path, clip_dir=clip_dir)


def _render_track_model_selection(track: RawTrack) -> None:
    if not track.model_selection:
        return

    selection = track.model_selection
    model_icon = "🔵" if selection.model_type == "VTA" else "🟢"
    rule_tag = " ⚡규칙" if selection.rule_based else ""
    st.caption(
        f"{model_icon} **{selection.model_type}**{rule_tag} ({selection.confidence:.0%})  \n"
        f"vta={selection.vta_score:.1f} / tta={selection.tta_score:.1f}  \n"
        f"{selection.reasoning}"
    )


def _render_track_clip(track: RawTrack, *, video_path: str, clip_dir: str) -> None:
    if not video_path or not clip_dir:
        return

    clip_path = extract_clip(video_path, track.start, track.end, clip_dir)
    if clip_path:
        st.video(clip_path)
    else:
        st.warning("영상 클립 추출 실패")


def _render_langfuse_summary(inspect_state: InspectState | None) -> None:
    if inspect_state is None:
        return

    trace_id = inspect_state.get("trace_id")
    if not trace_id:
        return

    st.caption(f"Langfuse trace id: `{trace_id}`")
    with st.expander("🧪 Langfuse Review", expanded=False):
        quality_key = "langfuse_overall_grouping_quality"
        approval_key = "langfuse_approved_for_export"
        quality_score = st.slider(
            "Overall grouping quality",
            min_value=1,
            max_value=5,
            value=int(st.session_state.get(quality_key, 3)),
            key=quality_key,
        )
        approved = st.checkbox(
            "Approved for export",
            value=bool(st.session_state.get(approval_key, False)),
            key=approval_key,
        )

        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Save overall score", key="langfuse_save_overall_score"):
                success = create_trace_score(
                    trace_id=trace_id,
                    name="overall_grouping_quality",
                    value=float(quality_score),
                    data_type="NUMERIC",
                    score_id=build_score_id(trace_id, "overall_grouping_quality"),
                    metadata={"scale": "1-5"},
                    flush=True,
                )
                if success:
                    st.success("Saved overall grouping score to Langfuse.")
                else:
                    st.warning("Langfuse is not configured, so the score was not sent.")

        with col_right:
            if st.button("Save approval", key="langfuse_save_approval"):
                success = create_trace_score(
                    trace_id=trace_id,
                    name="approved_for_export",
                    value=1.0 if approved else 0.0,
                    data_type="BOOLEAN",
                    score_id=build_score_id(trace_id, "approved_for_export"),
                    metadata={"approved": approved},
                    flush=True,
                )
                if success:
                    st.success("Saved export approval to Langfuse.")
                else:
                    st.warning("Langfuse is not configured, so the score was not sent.")


def _render_group_review_controls(
    *,
    trace_id: str | None,
    group: TrackGroup,
    override: str,
) -> None:
    if not trace_id:
        return

    review_value = st.selectbox(
        "그룹 리뷰",
        ["(미기록)", "correct", "overmerged", "oversplit", "unclear"],
        key=f"langfuse_group_review_{group.group_id}",
    )

    if st.button(
        "Langfuse에 그룹 리뷰 기록",
        key=f"langfuse_save_group_review_{group.group_id}",
    ):
        if review_value == "(미기록)":
            st.warning("기록할 그룹 리뷰 값을 먼저 선택해주세요.")
        else:
            success = create_trace_score(
                trace_id=trace_id,
                name="group_review",
                value=review_value,
                data_type="CATEGORICAL",
                score_id=build_score_id(trace_id, "group_review", group.group_id),
                metadata={
                    "group_id": group.group_id,
                    "member_ids": group.member_ids,
                },
                flush=True,
            )
            if success:
                st.success(f"Saved group review for {group.group_id}.")
            else:
                st.warning("Langfuse is not configured, so the score was not sent.")

    if override != "(자동)" and st.button(
        "Langfuse에 오버라이드 기록",
        key=f"langfuse_save_model_override_{group.group_id}",
    ):
        success = create_trace_score(
            trace_id=trace_id,
            name="model_override",
            value=override,
            data_type="CATEGORICAL",
            score_id=build_score_id(trace_id, "model_override", group.group_id),
            metadata={
                "group_id": group.group_id,
                "auto_model": (
                    group.model_selection.model_type
                    if group.model_selection is not None
                    else None
                ),
                "override": override,
            },
            flush=True,
        )
        if success:
            st.success(f"Saved model override for {group.group_id}.")
        else:
            st.warning("Langfuse is not configured, so the score was not sent.")
