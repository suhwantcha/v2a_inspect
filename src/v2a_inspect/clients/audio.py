from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import openai
from elevenlabs.client import ElevenLabs
import scipy.io.wavfile as wavfile

logger = logging.getLogger(__name__)


def generate_dialogue_openai(
    text: str, out_path: str, duration: float | None = None
) -> str:
    """Generate speech using OpenAI TTS API with adaptive speed and voice."""
    import re
    # Extract quoted text to speak, but keep the full text for context
    match = re.search(r'["\'](.*?)["\']', text)
    spoken_text = match.group(1) if match else text

    # Select voice based on speaker description
    desc_lower = text.lower()
    if any(w in desc_lower for w in ["female", "woman", "girl", "lady", "여성", "여자", "소녀"]):
        voice = "nova"
    elif any(w in desc_lower for w in ["deep", "monster", "large man", "giant", "괴물", "거인", "거친"]):
        voice = "onyx"
    elif any(w in desc_lower for w in ["male", "man", "boy", "guy", "남성", "남자", "소년"]):
        voice = "echo"
    else:
        voice = "alloy"  # Neutral default

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    client = openai.OpenAI(api_key=api_key)

    # Determine speech speed to fit the duration
    speed = 1.0
    if duration and duration > 0.1:
        # Heuristic: ~12 characters per second is a normal speaking rate
        standard_duration = max(len(spoken_text) / 12.0, 0.5)
        raw_speed = standard_duration / duration
        # Clamp speed to OpenAI's supported range [0.25, 4.0]
        speed = max(0.25, min(raw_speed, 4.0))

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=spoken_text,
        speed=speed,
    )

    response.stream_to_file(out_path)
    return out_path


def generate_sfx_elevenlabs(
    text: str, out_path: str, duration: float | None = None
) -> str:
    """Generate sound effects using ElevenLabs API."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.warning(
            "ELEVENLABS_API_KEY not found. Falling back to dummy audio."
        )
        return generate_dummy_audio(duration or 1.0, out_path)

    try:
        client = ElevenLabs(api_key=api_key)
        
        # ElevenLabs duration is in seconds (0.5 to 30)
        # However when sending None, let the model decide.
        dur_seconds = min(max(duration, 0.5), 30.0) if duration else None

        # Returns an Iterator[bytes]
        audio_generator = client.text_to_sound_effects.convert(
            text=text,
            duration_seconds=dur_seconds,
        )

        with open(out_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        return out_path
    except Exception as e:
        logger.error(f"ElevenLabs SFX generation failed: {e}")
        return generate_dummy_audio(duration or 1.0, out_path)


def generate_music_elevenlabs(
    text: str, out_path: str, duration: float | None = None
) -> str:
    """Generate background music or score using ElevenLabs API."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.warning(
            "ELEVENLABS_API_KEY not found. Falling back to dummy audio."
        )
        return generate_dummy_audio(duration or 1.0, out_path)

    try:
        client = ElevenLabs(api_key=api_key)
        
        # ElevenLabs music duration is in milliseconds (min 3000)
        dur_ms = int(min(max(duration, 3.0), 30.0) * 1000) if duration else 10000

        # Returns an Iterator[bytes]
        audio_generator = client.music.compose(
            prompt=text,
            music_length_ms=dur_ms,
        )

        with open(out_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        return out_path
    except Exception as e:
        if "paid_plan_required" in str(e) or "402" in str(e):
            logger.error("ElevenLabs Music API is only available for paid users. Falling back to dummy audio.")
        else:
            logger.error(f"ElevenLabs Music generation failed: {e}")
        return generate_dummy_audio(duration or 1.0, out_path)


def generate_dummy_audio(
    duration_sec: float, out_path: str, sample_rate: int = 44100
) -> str:
    """Generate a placeholder silent/beep audio file (fallback)."""
    if duration_sec <= 0:
        duration_sec = 0.1

    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

    # Generate a very soft beep at the beginning
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # Fade out quickly after 0.1s
    fade_len = int(0.1 * sample_rate)
    if audio.size > fade_len:
        audio[fade_len:] = 0

    wavfile.write(out_path, sample_rate, (audio * 32767).astype(np.int16))
    return out_path
