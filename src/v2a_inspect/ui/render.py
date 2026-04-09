"""
V2A Inspect — Streamlit UI (render.py)
Updated for the 10-step Director-First pipeline.
"""
from __future__ import annotations

import math
from pathlib import Path
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


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

def render_page_header() -> None:
    st.title("🎬 V2A Inspect — Director-First 오디오 합성 파이프라인")
    st.markdown(
        "Gemini 기반 **Director Intent → Audio Plan → Relation Graph → Evaluation** "
        "10-step 파이프라인으로 영상에 최적화된 멀티트랙 오디오를 자동 생성합니다."
    )
    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(authenticator: Any) -> InspectOptions:
    with st.sidebar:
        st.header("⚙️ 파이프라인 옵션")

        fps = st.slider("Analysis FPS", min_value=1.0, max_value=16.0, value=8.0, step=1.0)
        st.caption("초당 분석 프레임 수. 높을수록 정밀하지만 느림.")

        prompt_type = cast(
            Literal["default", "extended"],
            st.selectbox("Prompt Type", ["default", "extended"], index=0),
        )
        st.caption("`default`: 간결 | `extended`: Foley 상세")

        st.divider()
        st.subheader("Phase 1 — Intent + Plan")

        enable_director_intent = st.checkbox("🎯 Director Intent 추출", value=True)
        st.caption("감정 아크/장르/오디오 방향을 먼저 추출해 모든 분석에 주입")

        enable_audio_plan = st.checkbox("📋 Audio Plan 생성", value=True)
        st.caption("로컬/글로벌 분석을 통합한 타임라인 기반 오디오 플랜")

        silence_pad = st.slider(
            "Silence pad (key moment 전)", min_value=0.0, max_value=2.0, value=0.4, step=0.1
        )
        st.caption("Key emotional moment 직전에 삽입할 침묵(초)")

        st.divider()
        st.subheader("Phase 2 — Relation Graph")

        enable_relation_graph = st.checkbox("🕸️ Relation Graph 활성화", value=True)
        st.caption("causes/ducks 관계 추출 + 위상 정렬 생성 순서")

        st.divider()
        st.subheader("Phase 3 — Evaluation (실험적)")

        enable_evaluation = st.checkbox("🔬 Evaluation + Refinement", value=False)
        max_refinement_iter = st.slider("최대 Refinement 반복", 1, 5, 2)
        eval_threshold = st.slider("Evaluation 통과 점수", 0.5, 1.0, 0.75, 0.05)

        st.divider()
        st.subheader("기타")

        enable_vlm_verify = st.checkbox("VLM 그룹 검증", value=True)
        enable_model_select = st.checkbox("TTA/VTA 모델 자동 선정", value=False)

        st.divider()
        with st.expander("🗺️ 파이프라인 구조", expanded=False):
            st.markdown(
                """
```
[1] Video Input
      ↓
[2] Director Intent  ★ NEW
      ↓
[3A] Local Pass  ┬  [3B] Global Pass
   (SFX/Dialogue)   (Music/Ambience)
      └──────────────┘
[4] Audio Plan (Silence 자동삽입)  ★ NEW
      ↓
[5] Relation Graph (causes/ducks)  ★ NEW
      ↓
[6] Causal Generation Order (topo sort)
      ↓
[7] Audio Generation
      ↓
[8] Evaluation (LLM-as-judge)  ★ NEW
   (if fail) → [8b] Refinement Loop
      ↓
[9] Final Mixing (vol/pan/duck/silence)
```
"""
            )

        if st.button("🔄 Reset", width="stretch", type="secondary"):
            reset_state()
            st.rerun()

        authenticator.logout("Logout", "sidebar")

    return InspectOptions(
        fps=fps,
        scene_analysis_mode=prompt_type,
        enable_director_intent=enable_director_intent,
        enable_audio_plan=enable_audio_plan,
        silence_pre_key_moment_sec=silence_pad,
        enable_relation_graph=enable_relation_graph,
        enable_evaluation=enable_evaluation,
        max_refinement_iter=max_refinement_iter,
        eval_score_threshold=eval_threshold,
        enable_vlm_verify=enable_vlm_verify,
        enable_model_select=enable_model_select,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level results renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_results(
    grouped: GroupedAnalysis,
    scene_analysis: VideoSceneAnalysis,
    *,
    video_path: str,
    clip_dir: str,
    inspect_state: InspectState | None,
) -> None:
    st.divider()
    _render_state_messages(inspect_state)
    _render_langfuse_summary(inspect_state)

    # Tab layout: 각 Phase별 탭
    tab_summary, tab_intent, tab_plan, tab_relation, tab_eval, tab_groups, tab_scenes = st.tabs([
        "📊 요약",
        "🎯 Director Intent",
        "📋 Audio Plan",
        "🕸️ Relation Graph",
        "🔬 Evaluation",
        "📦 트랙 그룹",
        "🎬 씬 분석",
    ])

    with tab_summary:
        _render_summary_tab(grouped, scene_analysis, inspect_state)

    with tab_intent:
        _render_intent_tab(inspect_state)

    with tab_plan:
        _render_plan_tab(inspect_state)

    with tab_relation:
        _render_relation_tab(inspect_state)

    with tab_eval:
        _render_evaluation_tab(inspect_state)

    with tab_groups:
        _render_groups_tab(grouped, video_path, clip_dir, inspect_state)

    with tab_scenes:
        _render_scenes_tab(scene_analysis)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: 요약
# ─────────────────────────────────────────────────────────────────────────────

def _render_summary_tab(
    grouped: GroupedAnalysis,
    scene_analysis: VideoSceneAnalysis,
    inspect_state: InspectState | None,
) -> None:
    st.subheader("전체 파이프라인 결과 요약")

    # ── 씬 분석 메트릭 ──
    n_local_scenes = len(scene_analysis.scenes)
    n_macro_segments = len(scene_analysis.macro_segments)
    n_dialogues = sum(len(s.dialogues) for s in scene_analysis.scenes)
    n_sfx = sum(len(s.sfx) for s in scene_analysis.scenes)
    n_music = sum(len(seg.music) for seg in scene_analysis.macro_segments)
    n_ambience = sum(len(seg.ambience) for seg in scene_analysis.macro_segments)

    st.markdown("##### 씬 분석")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🎬 로컬 씬", n_local_scenes)
    c2.metric("🌍 글로벌 세그", n_macro_segments)
    c3.metric("🗣️ 대사", n_dialogues)
    c4.metric("💥 SFX", n_sfx)
    c5.metric("🎵 음악", n_music)
    c6.metric("🌬️ 환경음", n_ambience)

    # ── 그루핑 메트릭 ──
    n_raw = len(grouped.raw_tracks)
    n_groups = len(grouped.groups)
    n_merged = n_raw - n_groups
    n_multi = sum(1 for g in grouped.groups if len(g.member_ids) > 1)

    st.markdown("##### 트랙 그루핑")
    cg1, cg2, cg3, cg4 = st.columns(4)
    cg1.metric("📦 Raw 트랙", n_raw)
    cg2.metric("📦 그룹", n_groups, delta=f"-{n_merged}" if n_merged > 0 else None)
    cg3.metric("🔗 멀티멤버 그룹", n_multi)
    cg4.metric("✅ VLM 검증", sum(1 for g in grouped.groups if g.vlm_verified))

    # ── Phase 1/2/3 상태배지 ──
    if inspect_state:
        st.markdown("##### 파이프라인 단계 활성 여부")
        phases = [
            ("🎯 Director Intent", inspect_state.get("director_intent") is not None),
            ("📋 Audio Plan", inspect_state.get("audio_plan") is not None),
            ("🕸️ Relation Graph", inspect_state.get("relation_graph") is not None),
            ("🔬 Evaluation", inspect_state.get("evaluation_score") is not None),
            ("🎬 Final Video", inspect_state.get("mixed_video_path") is not None),
        ]
        cols = st.columns(len(phases))
        for col, (label, active) in zip(cols, phases):
            col.markdown(f"{'✅' if active else '⬜'} {label}")

    # ── Final video ──
    if inspect_state:
        mixed_path = inspect_state.get("mixed_video_path")
        if mixed_path and Path(mixed_path).exists():
            st.divider()
            st.subheader("🎬 최종 합성 비디오")
            st.video(str(mixed_path))


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Director Intent
# ─────────────────────────────────────────────────────────────────────────────

def _render_intent_tab(inspect_state: InspectState | None) -> None:
    st.subheader("🎯 Director Intent")
    st.caption("영상 전체를 보고 추출된 감독의 의도 — 모든 분석/생성에 최우선으로 주입됩니다.")

    if inspect_state is None:
        st.info("분석 결과가 없습니다.")
        return

    intent = inspect_state.get("director_intent")
    if intent is None:
        st.warning("Director Intent가 비활성화됐거나 추출되지 않았습니다.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🎞️ 장르:** `{intent.genre}`")
        st.markdown(f"**🎨 전체 분위기:** `{intent.overall_mood}`")
    with col2:
        st.markdown(f"**🎙️ 오디오 디렉션:**")
        st.info(intent.audio_direction)

    st.markdown("---")
    st.markdown("**📈 감정 아크 (Emotional Arc)**")

    if not intent.emotional_arc:
        st.caption("감정 아크 데이터 없음")
    else:
        # Timeline table
        rows = []
        for beat in intent.emotional_arc:
            rows.append({
                "시작(s)": f"{beat.time[0]:.1f}",
                "종료(s)": f"{beat.time[1]:.1f}",
                "감정": beat.emotion,
                "강도": f"{beat.intensity:.2f}",
                "Key Moment": "⭐ YES" if beat.key_moment else "",
            })
        st.dataframe(rows, width="stretch", hide_index=True)

        # Visual intensity bar
        st.markdown("**감정 강도 시각화:**")
        for beat in intent.emotional_arc:
            pct = int(beat.intensity * 100)
            km_badge = " ⭐" if beat.key_moment else ""
            label = f"{beat.emotion}{km_badge} [{beat.time[0]:.1f}s–{beat.time[1]:.1f}s]"
            st.progress(pct, text=f"{label} ({pct}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Audio Plan
# ─────────────────────────────────────────────────────────────────────────────

def _render_plan_tab(inspect_state: InspectState | None) -> None:
    st.subheader("📋 Audio Plan")
    st.caption("통합된 타임라인 기반 오디오 플랜 — silence 자동 삽입 포함.")

    if inspect_state is None:
        st.info("분석 결과가 없습니다.")
        return

    plan = inspect_state.get("audio_plan")
    if plan is None:
        st.warning("Audio Plan이 비활성화됐거나 생성되지 않았습니다.")
        return

    total_dur = getattr(plan, "total_duration", None)
    items = list(plan.items)
    n_silence = sum(1 for i in items if i.type == "silence")
    n_non_silence = len(items) - n_silence

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎵 총 항목", len(items))
    c2.metric("🔊 오디오 항목", n_non_silence)
    c3.metric("🤫 Silence 블록", n_silence)
    if total_dur:
        c4.metric("⏱️ 총 길이", f"{total_dur:.1f}s")

    # Type breakdown
    type_counts: dict[str, int] = {}
    for item in items:
        type_counts[item.type] = type_counts.get(item.type, 0) + 1

    tc_cols = st.columns(len(type_counts))
    icons = {"dialogue": "🗣️", "sfx": "💥", "music": "🎵", "ambience": "🌬️", "silence": "🤫"}
    for col, (t, cnt) in zip(tc_cols, sorted(type_counts.items())):
        col.metric(f"{icons.get(t, '🎯')} {t}", cnt)

    st.markdown("---")

    # Gantt-style table
    st.markdown("**타임라인 뷰:**")
    type_color = {
        "dialogue": "🟦",
        "sfx": "🟥",
        "music": "🟨",
        "ambience": "🟩",
        "silence": "⬜",
    }
    rows = []
    for item in sorted(items, key=lambda x: x.time[0]):
        rows.append({
            "ID": item.item_id,
            "타입": f"{type_color.get(item.type, '🔲')} {item.type}",
            "시작(s)": f"{item.time[0]:.1f}",
            "종료(s)": f"{item.time[1]:.1f}",
            "길이(s)": f"{item.time[1]-item.time[0]:.1f}",
            "Vol": f"{item.volume:.2f}",
            "Intensity": f"{item.intensity:.2f}",
            "Pan": f"{item.pan:+.2f}",
            "Conf": f"{item.confidence:.0%}",
            "설명": item.description[:60],
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    with st.expander("📋 Audio Plan JSON", expanded=False):
        st.json(plan.model_dump())


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Relation Graph
# ─────────────────────────────────────────────────────────────────────────────

def _render_relation_tab(inspect_state: InspectState | None) -> None:
    st.subheader("🕸️ Relation Graph")
    st.caption("오디오 항목 간 인과(causes) 및 마스킹(ducks) 관계와 위상 정렬 생성 순서.")

    if inspect_state is None:
        st.info("분석 결과가 없습니다.")
        return

    rg = inspect_state.get("relation_graph")
    if rg is None:
        st.warning("Relation Graph가 비활성화됐거나 생성되지 않았습니다.")
        return

    relations = list(rg.relations)
    causal_order = list(rg.causal_order)
    causes = [r for r in relations if r.relation == "causes"]
    ducks = [r for r in relations if r.relation == "ducks"]

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("🔗 총 관계", len(relations))
    rc2.metric("⚡ causes", len(causes))
    rc3.metric("🔇 ducks", len(ducks))

    if causes:
        st.markdown("**⚡ Causes 관계** (from_item → to_item: 먼저 생성해야 함)")
        rows = [
            {
                "From": r.from_item_id,
                "To": r.to_item_id,
                "강도": f"{r.strength:.2f}",
            }
            for r in causes
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    if ducks:
        st.markdown("**🔇 Ducks 관계** (from_item이 재생되는 동안 to_item의 볼륨 감소)")
        rows = [
            {
                "From": r.from_item_id,
                "To": r.to_item_id,
                "Duck 강도": f"{r.strength:.2f}",
            }
            for r in ducks
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    if not relations:
        st.success("인과/마스킹 관계가 없습니다. (단순 시간 순서로 생성)")

    if causal_order:
        st.markdown("**🏁 위상 정렬 생성 순서 (Causal Generation Order)**")
        ordered = " → ".join(f"`{iid}`" for iid in causal_order[:20])
        if len(causal_order) > 20:
            ordered += f" ... (+{len(causal_order)-20}개)"
        st.markdown(ordered)

    with st.expander("📋 Relation Graph JSON", expanded=False):
        st.json(rg.model_dump())


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _render_evaluation_tab(inspect_state: InspectState | None) -> None:
    st.subheader("🔬 Evaluation Score")
    st.caption(
        "Mid-level 평가 점수 — S = α·S_temp + β·S_sem + γ·S_global  \n"
        "통과 점수 미달 시 Refinement Loop로 약점 항목만 재개선합니다."
    )

    if inspect_state is None:
        st.info("분석 결과가 없습니다.")
        return

    score = inspect_state.get("evaluation_score")
    if score is None:
        st.warning("Evaluation이 비활성화됐거나 실행되지 않았습니다.")
        st.caption("CLI: `--evaluation` 플래그 또는 사이드바에서 활성화")
        return

    # Pass/Fail badge
    if score.passed:
        st.success(f"✅ 평가 통과 — 총점: **{score.total:.3f}** (iter #{score.iteration})")
    else:
        st.error(f"❌ 평가 미통과 — 총점: **{score.total:.3f}** (iter #{score.iteration})")

    # Score metrics
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🏆 Total", f"{score.total:.3f}")
    s2.metric("⏱️ Temporal", f"{score.temporal:.3f}", help="커버리지 기반 rule-based 점수")
    s3.metric("💬 Semantic", f"{score.semantic:.3f}", help="LLM-as-judge: 개별 트랙 설명 품질")
    s4.metric("🌍 Global Coherence", f"{score.global_coherence:.3f}", help="LLM-as-judge: Director Intent 부합도")

    # Score bar chart
    st.markdown("**Score 분포:**")
    bar_data = {
        "Temporal": score.temporal,
        "Semantic": score.semantic,
        "Global": score.global_coherence,
        "Total": score.total,
    }
    for name, val in bar_data.items():
        color = "🟢" if val >= 0.75 else "🟡" if val >= 0.5 else "🔴"
        st.progress(val, text=f"{color} {name}: {val:.3f}")

    if score.weak_item_ids:
        st.markdown(f"**⚠️ 약점 항목** (refinement 대상):")
        st.code(", ".join(score.weak_item_ids))

    if score.feedback:
        st.markdown("**📝 LLM Judge 피드백:**")
        st.info(score.feedback)

    refinement_iter = inspect_state.get("refinement_iteration", 0)
    if refinement_iter:
        st.caption(f"완료된 Refinement 반복: {refinement_iter}회")

    with st.expander("📋 Evaluation JSON", expanded=False):
        st.json(score.model_dump())

    _render_langfuse_eval_score(inspect_state, score)


def _render_langfuse_eval_score(inspect_state: InspectState, score) -> None:
    trace_id = inspect_state.get("trace_id") if inspect_state else None
    if not trace_id:
        return
    with st.expander("💾 Langfuse에 평가 점수 저장", expanded=False):
        if st.button("Save evaluation score to Langfuse", key="langfuse_save_eval_score"):
            for metric_name, value in [
                ("eval_total", score.total),
                ("eval_temporal", score.temporal),
                ("eval_semantic", score.semantic),
                ("eval_global_coherence", score.global_coherence),
            ]:
                create_trace_score(
                    trace_id=trace_id,
                    name=metric_name,
                    value=value,
                    data_type="NUMERIC",
                    score_id=build_score_id(trace_id, metric_name),
                    metadata={"iteration": score.iteration, "passed": score.passed},
                    flush=(metric_name == "eval_global_coherence"),
                )
            st.success("Evaluation scores saved to Langfuse.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab: 트랙 그룹
# ─────────────────────────────────────────────────────────────────────────────

def _render_groups_tab(
    grouped: GroupedAnalysis,
    video_path: str,
    clip_dir: str,
    inspect_state: InspectState | None,
) -> None:
    st.subheader("📦 트랙 그루핑 결과")
    st.caption(
        "각 그룹의 **canonical description**과 멤버별 원본 description·영상 클립을 비교합니다."
    )

    # Audio plan awarenesss — show generated audio if available
    plan_items_by_track: dict[str, Any] = {}
    if inspect_state:
        plan = inspect_state.get("audio_plan")
        if plan:
            for item in plan.items:
                if item.track_id:
                    plan_items_by_track[item.track_id] = item
        generated_audio: dict[str, str] = inspect_state.get("generated_audio") or {}
    else:
        generated_audio = {}

    tracks_by_id = {track.track_id: track for track in grouped.raw_tracks}
    trace_id = inspect_state.get("trace_id") if inspect_state else None

    for group in grouped.groups:
        members = [
            tracks_by_id[mid]
            for mid in group.member_ids
            if mid in tracks_by_id
        ]
        _render_group_expander(
            group=group,
            members=members,
            video_path=video_path,
            clip_dir=clip_dir,
            trace_id=trace_id,
            plan_items_by_track=plan_items_by_track,
            generated_audio=generated_audio,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab: 씬 분석
# ─────────────────────────────────────────────────────────────────────────────

def _render_scenes_tab(scene_analysis: VideoSceneAnalysis) -> None:
    st.subheader("🎬 씬별 분석 결과")

    view_mode = st.radio(
        "보기 모드", ["로컬 씬 (SFX/Dialogue)", "글로벌 세그 (Music/Ambience)", "JSON"], horizontal=True
    )

    if view_mode == "로컬 씬 (SFX/Dialogue)":
        for i, scene in enumerate(scene_analysis.scenes):
            with st.expander(
                f"[씬 {i}] {scene.time_range.start:.1f}s–{scene.time_range.end:.1f}s "
                f"| 대사 {len(scene.dialogues)}개 · SFX {len(scene.sfx)}개",
                expanded=(i < 3),
            ):
                col_d, col_s = st.columns(2)
                with col_d:
                    st.markdown("**🗣️ Dialogues**")
                    for obj in scene.dialogues:
                        pan_str = f"  pan={obj.pan:+.2f}" if hasattr(obj, "pan") else ""
                        st.markdown(
                            f"- `{obj.time_range.start:.1f}s–{obj.time_range.end:.1f}s`{pan_str}  \n  > {obj.description}"
                        )
                    if not scene.dialogues:
                        st.caption("(없음)")
                with col_s:
                    st.markdown("**💥 SFX**")
                    for obj in scene.sfx:
                        pan_str = f"  pan={obj.pan:+.2f}" if hasattr(obj, "pan") else ""
                        st.markdown(
                            f"- `{obj.time_range.start:.1f}s–{obj.time_range.end:.1f}s`{pan_str}  \n  > {obj.description}"
                        )
                    if not scene.sfx:
                        st.caption("(없음)")

    elif view_mode == "글로벌 세그 (Music/Ambience)":
        for i, seg in enumerate(scene_analysis.macro_segments):
            with st.expander(
                f"[세그 {i}] {seg.time_range.start:.1f}s–{seg.time_range.end:.1f}s "
                f"| 음악 {len(seg.music)}개 · 환경음 {len(seg.ambience)}개",
                expanded=(i < 3),
            ):
                col_m, col_a = st.columns(2)
                with col_m:
                    st.markdown("**🎵 Music**")
                    for obj in seg.music:
                        st.markdown(f"- > {obj.description}")
                    if not seg.music:
                        st.caption("(없음)")
                with col_a:
                    st.markdown("**🌬️ Ambience**")
                    for obj in seg.ambience:
                        st.markdown(f"- > {obj.description}")
                    if not seg.ambience:
                        st.caption("(없음)")
    else:
        with st.expander("📋 VideoSceneAnalysis JSON", expanded=True):
            st.json(scene_analysis.model_dump())


# ─────────────────────────────────────────────────────────────────────────────
# Group expander
# ─────────────────────────────────────────────────────────────────────────────

def _render_group_expander(
    *,
    group: TrackGroup,
    members: list[RawTrack],
    video_path: str,
    clip_dir: str,
    trace_id: str | None,
    plan_items_by_track: dict[str, Any],
    generated_audio: dict[str, str],
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
            st.markdown(f"**Canonical description:**  \n> {group.canonical_description}")
        with hcol_b:
            st.markdown(f"**{badge}**")
            st.caption(f"멤버 {len(members)}개")
            if group.model_selection:
                sel = group.model_selection
                icon = "🔵" if sel.model_type == "VTA" else "🟢"
                flag = " ⚠️" if sel.confidence < 0.6 else ""
                st.markdown(
                    f"{icon} **{sel.model_type}**{flag}  \n"
                    f"conf: {sel.confidence:.0%}"
                )

            current_override = st.session_state.model_overrides.get(group.group_id, "(자동)")
            override = st.selectbox(
                "모델 오버라이드",
                ["(자동)", "TTA", "VTA"],
                index=["(자동)", "TTA", "VTA"].index(current_override),
                key=f"model_override_{group.group_id}",
            )
            if override != current_override:
                st.session_state.model_overrides[group.group_id] = override

            _render_group_review_controls(trace_id=trace_id, group=group, override=override)

        st.markdown("---")

        if not is_multi and members:
            _render_singleton_member(
                members[0],
                video_path=video_path,
                clip_dir=clip_dir,
                plan_items_by_track=plan_items_by_track,
                generated_audio=generated_audio,
            )
            return

        max_cols = min(len(members), 4)
        columns = st.columns(max_cols)
        for index, track in enumerate(members):
            with columns[index % max_cols]:
                _render_member(
                    track,
                    video_path=video_path,
                    clip_dir=clip_dir,
                    heading_level=4,
                    plan_items_by_track=plan_items_by_track,
                    generated_audio=generated_audio,
                )


def _render_singleton_member(
    track: RawTrack,
    *,
    video_path: str,
    clip_dir: str,
    plan_items_by_track: dict[str, Any],
    generated_audio: dict[str, str],
) -> None:
    kind_icon = _get_kind_icon(track.kind)
    pan_str = ""
    if hasattr(track, "pan") and track.pan != 0.0:
        pan_str = f" | pan={track.pan:+.2f}"
    st.markdown(
        f"{kind_icon} `{track.track_id}` | "
        f"Scene {track.scene_index} | "
        f"{track.start:.1f}s–{track.end:.1f}s | "
        f"*{track.kind}*{pan_str}"
    )
    st.info(track.description)
    _render_track_model_selection(track)
    _render_plan_item_info(track, plan_items_by_track, generated_audio)
    _render_track_clip(track, video_path=video_path, clip_dir=clip_dir)


def _render_member(
    track: RawTrack,
    *,
    video_path: str,
    clip_dir: str,
    heading_level: int,
    plan_items_by_track: dict[str, Any],
    generated_audio: dict[str, str],
) -> None:
    kind_icon = _get_kind_icon(track.kind)
    st.markdown(f"{'#' * heading_level} {kind_icon} `{track.track_id}`")
    pan_str = f" | pan={track.pan:+.2f}" if hasattr(track, "pan") and track.pan != 0.0 else ""
    st.caption(
        f"Scene {track.scene_index} | {track.start:.1f}s–{track.end:.1f}s | *{track.kind}*{pan_str}"
    )
    st.info(track.description)
    _render_track_model_selection(track)
    _render_plan_item_info(track, plan_items_by_track, generated_audio)
    _render_track_clip(track, video_path=video_path, clip_dir=clip_dir)


def _render_plan_item_info(
    track: RawTrack,
    plan_items_by_track: dict[str, Any],
    generated_audio: dict[str, str],
) -> None:
    """Show Audio Plan item info and generated audio file if available."""
    plan_item = plan_items_by_track.get(track.track_id)
    if plan_item:
        st.caption(
            f"📋 AudioPlan: vol={plan_item.volume:.2f} | "
            f"intensity={plan_item.intensity:.2f} | "
            f"conf={plan_item.confidence:.0%}"
        )

    # Check for generated audio
    item_id = plan_item.item_id if plan_item else track.track_id
    wav_path = generated_audio.get(item_id) or generated_audio.get(track.track_id)
    if wav_path and Path(wav_path).exists():
        st.audio(wav_path)


# ─────────────────────────────────────────────────────────────────────────────
# Shared render helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_kind_icon(kind: str) -> str:
    if kind == "music":    return "🎵"
    if kind == "ambience": return "🌬️"
    if kind == "dialogue": return "🗣️"
    if kind == "sfx":      return "💥"
    return "🎯"


def _render_track_model_selection(track: RawTrack) -> None:
    if not track.model_selection:
        return
    sel = track.model_selection
    icon = "🔵" if sel.model_type == "VTA" else "🟢"
    rule_tag = " ⚡규칙" if sel.rule_based else ""
    st.caption(
        f"{icon} **{sel.model_type}**{rule_tag} ({sel.confidence:.0%})  \n"
        f"vta={sel.vta_score:.1f} / tta={sel.tta_score:.1f}  \n"
        f"{sel.reasoning}"
    )


def _render_track_clip(track: RawTrack, *, video_path: str, clip_dir: str) -> None:
    if not video_path or not clip_dir:
        return
    clip_path = extract_clip(video_path, track.start, track.end, clip_dir)
    if clip_path:
        st.video(clip_path)
    else:
        st.warning("영상 클립 추출 실패")


def _render_state_messages(inspect_state: InspectState | None) -> None:
    if inspect_state is None:
        return
    warnings = inspect_state.get("warnings", [])
    progress_messages = inspect_state.get("progress_messages", [])

    if warnings:
        with st.expander("⚠️ 워크플로우 경고", expanded=True):
            for msg in warnings:
                st.warning(msg)

    if progress_messages:
        with st.expander(f"🧭 파이프라인 로그 ({len(progress_messages)}개)", expanded=False):
            for msg in progress_messages:
                # Detect which stage the message belongs to
                badge = ""
                lower = msg.lower()
                if "intent" in lower:       badge = "🎯"
                elif "audio plan" in lower: badge = "📋"
                elif "relation" in lower:   badge = "🕸️"
                elif "evaluation" in lower: badge = "🔬"
                elif "refinement" in lower: badge = "🔁"
                elif "generat" in lower:    badge = "🎵"
                elif "mixed" in lower:      badge = "🎬"
                elif "group" in lower:      badge = "📦"
                elif "scene" in lower:      badge = "🎬"
                st.write(f"{badge} {msg}")


def _render_langfuse_summary(inspect_state: InspectState | None) -> None:
    if inspect_state is None:
        return
    trace_id = inspect_state.get("trace_id")
    if not trace_id:
        return
    st.caption(f"🔭 Langfuse trace: `{trace_id}`")
    with st.expander("🧪 Langfuse Review", expanded=False):
        quality_key = "langfuse_overall_grouping_quality"
        approval_key = "langfuse_approved_for_export"
        quality_score = st.slider(
            "Overall quality",
            min_value=1, max_value=5,
            value=int(st.session_state.get(quality_key, 3)),
            key=quality_key,
        )
        approved = st.checkbox(
            "Approved for export",
            value=bool(st.session_state.get(approval_key, False)),
            key=approval_key,
        )
        col_l, col_r = st.columns(2)
        with col_l:
            if st.button("Save overall score", key="langfuse_save_overall_score"):
                success = create_trace_score(
                    trace_id=trace_id,
                    name="overall_quality",
                    value=float(quality_score),
                    data_type="NUMERIC",
                    score_id=build_score_id(trace_id, "overall_quality"),
                    flush=True,
                )
                st.success("Saved.") if success else st.warning("Langfuse 미설정")
        with col_r:
            if st.button("Save approval", key="langfuse_save_approval"):
                success = create_trace_score(
                    trace_id=trace_id,
                    name="approved_for_export",
                    value=1.0 if approved else 0.0,
                    data_type="BOOLEAN",
                    score_id=build_score_id(trace_id, "approved_for_export"),
                    flush=True,
                )
                st.success("Saved.") if success else st.warning("Langfuse 미설정")


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
    if st.button("Langfuse에 그룹 리뷰 기록", key=f"langfuse_save_group_review_{group.group_id}"):
        if review_value == "(미기록)":
            st.warning("리뷰 값을 선택해주세요.")
        else:
            success = create_trace_score(
                trace_id=trace_id,
                name="group_review",
                value=review_value,
                data_type="CATEGORICAL",
                score_id=build_score_id(trace_id, "group_review", group.group_id),
                metadata={"group_id": group.group_id, "member_ids": group.member_ids},
                flush=True,
            )
            st.success(f"Saved {group.group_id}.") if success else st.warning("Langfuse 미설정")

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
            metadata={"group_id": group.group_id, "override": override},
            flush=True,
        )
        st.success(f"Saved override for {group.group_id}.") if success else st.warning("Langfuse 미설정")


def render_footer() -> None:
    st.divider()
    st.caption(
        "V2A Inspect | Director-First Pipeline · Gemini · OpenAI · ElevenLabs · MoviePy"
    )
