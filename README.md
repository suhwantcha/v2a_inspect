# V2A Inspect: Director-First 4-Track Audio Pipeline

이 문서는 V2A (Video-to-Audio) 자동 생성 및 합성 시스템의 최신 아키텍처와 **10-Step Self-Evolving 파이프라인**을 상세히 정리한 문서입니다. 단순한 시간대 매핑을 넘어, 사운드 디렉터의 관점으로 씬의 감정선(Emotional Arc)을 분석하고 소리 간의 인과관계(causes/ducks)를 고려하여 몰입감 높은(Cinematic) 오디오를 생성합니다.

---

## 🛠️ 파이프라인 단계 (10-Step Workflow)

### Phase 1: Planning (설계 및 분석)

**1. Video Upload & Bootstrap**
   - 유저가 프론트엔드(Streamlit)를 통해 비디오 파일 업로드
   - Langfuse 추적(Tracing) 및 프레임 분할 초기화

**2. Director Intent (Top-Down Anchor)**
   - 영상 전체의 분위기(Genre, Mood), 오디오 디렉션, 그리고 시간에 따른 '감정 아크(Emotional Arc)'와 'Key Moment'를 추출합니다. 이후 모든 오디오 생성의 최우선 기준으로 작용합니다.

**3. Hierarchical Scene Analysis (2-Pass)**
   - **Local Pass**: 대사(dialogue)와 효과음(sfx)을 프레임 단위로 미시 분석. 입체 음향을 위한 Pan(좌우 오디오 위치) 값 동시 추출.
   - **Global Pass**: 배경음악(music)과 환경음(ambience)을 매크로 세그먼트 단위로 거시 분석.

**4. Text Grouping & Assembly**
   - 동일한 객체에서 발생한 여러 개의 Local 트랙들을 논리적으로 그룹화(Grouping)하여 텍스트의 파편화를 방지.

**5. Audio Plan Generation**
   - 앞선 분석 결과를 통합해 하나의 `Unified Audio Plan`을 생성.
   - 🌟 **Silence Padding**: 감정적 하이라이트(Key Moment) 직전에 '침묵(silence) 윈도우'를 자동 삽입해 극적 긴장감을 높입니다.

**6. Relation Graph & Causal Ordering**
   - 생성될 오디오 항목 간의 **인과성(causes)**과 **마스킹(ducks)** 관계를 추출. 
   - 이 그래프를 위상 정렬(Topological Sort)하여, 어떤 소리를 먼저 생성하고 어떤 소리가 다른 소리에 영향을 받을지 결정합니다.

---

### Phase 2: UI Review (선검토)

- **Interruption Point**: Phase 1이 완료되면 파이프라인이 멈추고 Streamlit UI에 결과를 렌더링합니다.
- 사용자는 요약된 트랙 그룹, Director Intent, Audio Plan의 타임라인, Relation Graph를 검토한 뒤 **"🎵 오디오 생성 및 영상 합성 (Final Mixing)"** 버튼을 눌러 다음 단계로 진행합니다.

---

### Phase 3: Synthesis & Refinement (합성 및 자가 진화)

**7. Audio Generation (Causal Mode & Adaptive TTS)**
   - 위상 정렬된 순서대로 오디오를 생성. Director Intent에서 부여받은 감정 강도(Intensity)에 따라 프롬프트 톤을 조절합니다 (예: "Intense and powerful: ...").
   - **동적 대사(Dialogue) 생성**: 대사를 렌더링할 때, 해당 씬의 **재생 시간(Duration)**과 **텍스트 길이**를 비교 연산하여, 주어진 시간 안에 대사가 넘치지 않도록 말하기 속도(`speed`, 0.25배속~4.0배속)를 자동 감속/가속합니다. 더불어 묘사 안의 키워드(예: 'female', 'large man')를 파싱하여 적합한 성별과 톤의 성우(`voice`)로 교체합니다.
   - 대사(Dialogue)는 **OpenAI TTS**, 그 외(SFX/Music)는 **ElevenLabs SDK**로 동적 라우팅 수행. 

**8. Mid-Level Evaluation (LLM-as-Judge)**
   - 생성 계획 대비 품질을 3가지 지표로 자동 채점: 
     - **Temporal** (시간 매칭)
     - **Semantic** (장면 설명 일치도)
     - **Global** (디렉터 의도 부합도)
   - 평가 점수가 기준치(Threshold) 미만이면 실패한 `weak items`만 추려냅니다.

**9. Self-Evolving Refinement Loop (자가 개선)**
   - 점수 미달 시, 실패한 아이템에 대해 프롬프트를 보강하고 해당 항목만 **부분 재생성(Partial Re-generation)**하여 품질을 끌어올립니다.

**10. Final Audio Mixing**
   - MoviePy를 기반으로 Numpy 벡터 연산을 통해 정교한 사운드 믹싱 수행.
   - `Volume` (기본 볼륨), `Pan` (스테레오 패닝), `Ducking` (주요 소리 등장 시 배경음 감소), `Silence Attenuation` (의도적 침묵 구간) 효과 일체 적용.

---

## 🚀 파이프라인 아키텍처 (Architecture Flow)

```text
       [입력] Video (.mp4)
             │
[1] Bootstrap & Langfuse Tracing
             │
[2] Director Intent ⭐ (Top-Down Anchor)
    [ 감정 변화(Arc), 장르, Key Moment 추출 ]
             │
             ├─────────────── 2-Pass Analysis ───────────────┐
             ▼                                               ▼
[3A] Local Pass (미시적)                              [3B] Global Pass (거시적)
 (Dialogue/SFX + Pan 추출)                        (Music/Ambience 추출)
             │                                               │
             ▼                                               │
[4] Text Grouping (동일 객체 병합)                         │
             │                                               │
             └──────────────────┬────────────────────────────┘
                                ▼
[5] Audio Plan Generation
    [ Unified 타임라인, Silence 패딩, Volume/Intensity 부여 ]
                                │
[6] Relation Graph (인과 추론)
    [ 'causes'(생성 순서), 'ducks'(음량 억제) 등 위상정렬 추출 ]
                                │
                                🛑 (UI Review: 그룹화 및 플랜 선검토)
                                │
                        [Generate Button]
                                │  
[7] Audio Generation (위상 정렬 순서 기반)
    [ OpenAI TTS / ElevenLabs SFX / ElevenLabs Music ]
                                │
                                ▼
[8] Mid-Level Evaluation ◄──────┐ (Yes)
    [ Temporal/Semantic/Global 점수 채점 ]
                                │
                         (미달 시 / Limit 미만)
                                │
[9] Refinement Loop ────────────┘
    [ 피드백 반영 후 Weak Items 부분 재생성 ]
                                │
                          (통과 시 / Limit 도달)
                                │
                                ▼
[10] Final Video Mixing
    [ Pan, Ducking, Silence 윈도우 벡터 연산 적용 ]
                                │
                                ▼
                      최종 Cinematic Mix Video
```

---

## 🧩 핵심 분석 및 개선 로직 상세 (Deep Dive)

### 1. Director Intent (감독의 의도 추출)
단순한 텍스트 묘사를 넘어 파이프라인 전체의 **사운드 디자인 나침반(Top-Down Anchor)** 역할을 수행하는 대주제입니다. 하나의 영상 비디오에서 전체 영상의 장르, 무드, 오디오 디렉션, 그리고 시간에 따른 **감정 아크(Emotional Arc)**를 뽑아내어 다음 4단계에 걸쳐 강렬하게 관여합니다.

1. **Audio Plan의 텐션 조절**: 발견된 사운드 효과(발소리 등)의 묘사를 단순하게 쓰지 않고, 매칭된 타임라인의 장르와 분위기에 알맞도록 무겁고 긴장되게, 혹은 가볍게 쓰도록 유도하며 볼륨과 분위기 강도(`intensity`)를 부여합니다.
2. **Silence Padding(극적 침묵) 트리거**: 감정 아크에서 폭발점(`key_moment: true`)을 발견하면 구조적으로 그 직전 0.4초 구간에 '의도된 숨죽임(Silence 윈도우)'을 자동 강제 삽입하여 몰입감을 증폭시킵니다.
3. **Mid-Level Evaluation의 절대 채점 기준**: 최종적으로 완성된 오디오 사운드들이 초반에 세운 '장르'와 '의도'에 부합되게 만들어졌는지(`Global Coherence Score`), 심사관 LLM이 이 데이터를 기준 삼아 날카롭게 평가합니다.
4. **Refinement Loop의 자가 수정 교보재**: 생성 점수가 미달일 시, 이 의도 텍스트를 피드백과 함께 곁들여 "이 장르와 의도에 완벽히 들어맞도록 수정해라"라고 압박하여 실패한 사운드 묘사의 퀄리티를 수직 상승시킵니다.

- **추출되는 JSON 구조 (`DirectorIntent`)**:
  ```json
  {
    "genre": "thriller",
    "overall_mood": "tense, claustrophobic",
    "audio_direction": "Use silence strategically. Swell into the climax.",
    "emotional_arc": [
      {
        "time": [0.0, 10.5],
        "emotion": "building dread",
        "intensity": 0.6,
        "key_moment": false
      },
      {
        "time": [10.5, 15.0],
        "emotion": "sudden shock",
        "intensity": 1.0,
        "key_moment": true
      }
    ]
  }
  ```

### 2. Audio Plan & Silence Padding (오디오 계획 및 여백)
단순한 사운드 묘사의 나열이 아닙니다. Local Pass와 Global Pass 데이터를 취합한 뒤, Director Intent 내용을 기반으로 **Volume**과 **Intensity**를 부여합니다.
- **프롬프트 특징**: Key Moment(`key_moment: true`)가 감지되면, 그 직전에 극적 효과를 위해 임의의 "silence"(침묵) 구간 항목을 강제 삽입하도록 지시합니다.
- **결과 구조 (`AudioPlanItem`)**:
  ```json
  {
    "item_id": "plan_sfx_0",
    "type": "silence", 
    "time": [9.0, 10.5], 
    "description": "intentional pause before the jumpscare",
    "volume": 0.0,
    "intensity": 0.0,
    "pan": 0.0
  }
  ```

### 3. Relation Graph (위상 정렬 인과 추론)
단순 병렬 생성이 가진 한계점(서로 영향을 주고받는 소리들 간의 이질감)을 극복하고, 사운드 믹서의 역할을 대체하기 위해 오디오 도면(Audio Plan)에서 항목 간의 **방향성 있는 그래프(Directed Graph)** 엣지(Edge)를 두 가지로 추출합니다.

LLM이 두 엣지를 헷갈리거나 남발하여 파이프라인을 꼬이게(Circular Dependency) 만드는 것을 방지하기 위해 다음과 같은 매우 엄격하고 제한적인 프롬프트 룰링이 적용됩니다:

**🗣 [릴레이션 추출 프롬프트 핵심 규칙]**
> - **CAUSES (`causes`)**: 1번 소리가 내러티브/물리적으로 2번 소리를 유발할 때만 사용.  (예: 폭발음 → 사람의 이명 소리, 총성 → 군중의 비명). `from_item`이 반드시 '먼저' 모델에 의해 생성되어야 그 소리의 질감을 바탕으로 `to_item`이 조화롭게 생성됨.
> - **DUCKS (`ducks`)**: 1번 소리가 재생 중일 때, 명료함(Clarity)을 확보하기 위해 2번 소리의 볼륨이 자동으로 줄어들어야 할 때 사용. (예: 대화 중 → 배경음악, 쾅 하는 폭발음 → 주변기류음). `strength` (더킹 강도)를 0.0~1.0 사이로 설정할 것.
> - **순환 참조 금지**: A가 B를 원인으로 하고, B가 다시 A를 원인으로 하는 그래프(Cycles)는 절대 만들지 말 것. 확신할 수 없으면 엣지를 만들지 마라(Empty).

- **구조화된 추출 JSON 리턴 예시 (`AudioRelation`)**:
  ```json
  {
    "relations": [
      {
        "from_item_id": "plan_sfx_explosion_1",
        "to_item_id": "plan_sfx_scream_2",
        "relation": "causes",
        "strength": 1.0
      },
      {
        "from_item_id": "plan_dlg_speech_0",
        "to_item_id": "plan_music_bg_0",
        "relation": "ducks",
        "strength": 0.8  // 배경음악을 아주 깊게(약 -10dB 수준) 줄임
      }
    ]
  }
  ```
- **Topological Sort (위상 정렬) 알고리즘**: 추출된 Relation JSON의 `causes` 배열 정보를 Python의 `graphlib.TopologicalSorter`에 입력합니다. 만약 LLM이 실수로 순환 참조(Cycle)를 만들었더라도 파이프라인이 터지는 대신 시스템단 예외 처리(Fallback)가 작동하여 단순히 "시간 순서(Timestamp) 기반 정렬"로 안전하게 우회됩니다.

### 4. Mid-Level Evaluation (평가 점수 산정 로직)
오디오가 모두 생성되면, **LLM-as-Judge**(평가자 LLM)가 생성 계획과 원래 비디오의 정보를 재차 종합하여 3가지 지표로 오디오를 채점합니다.
1. **Temporal Coverage**: (규칙 기반) 영상의 타임라인상 주요 객체 및 이벤트가 오디오 타임라인에 얼마나 누락 없이 커버되었는지의 비율 점수 (0~1.0).
2. **Semantic Coherence**: 각 Item의 묘사가 시각적 요소와 논리적으로 들어맞는지 0-100점 척도 평가 후 정규화.
3. **Global Alignment**: 전체 플랜 및 생성 묘사가 처음 추출했던 `Director Intent`의 `overall_mood` 및 `genre`에 부합하는지 평가.

**Score (가중치 채점)** = `α * Temporal + β * Semantic + γ * Global`

### 5. Refinement Loop (프롬프트 동적 보강 프로세스)
채점된 총 점수가 `threshold`(기본값 0.75) 미만일 경우 `iteration`을 증가시키며 **자가 수정 모드(Refinement)** 에 돌입합니다. LLM 평가자가 점수가 낮은 `weak_items`을 추출함과 동시에 피드백(`feedback`)을 반환합니다.

- **프롬프트 변경 구조**: 
  다음 Refine 단계에서는 Audio Plan 노드에 "이전 생성에 대한 피드백 지시사항"이 컨텍스트로 함께 전달됩니다.
  > "이전 생성 시도에서 다음과 같은 피드백이 나왔습니다: [LLM Feedback]. 이 피드백을 수용하여 `weak_items`에 해당하는 ID의 항목들의 description(묘사)를 더욱 구체적이거나, 타이밍을 미세 조정하여 재작성해 주세요."
- 수정이 완료된 새로운 Audio Plan 챕터가 반환되면, 기존 생성에 성공한 오디오 캐시 정보는 유지한 채로 **실패한(`weak_items`) 트랙만 비용을 들여 재호출(Re-generate)** 결합합니다.
