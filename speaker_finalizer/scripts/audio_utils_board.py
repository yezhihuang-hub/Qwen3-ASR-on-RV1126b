#!/usr/bin/env python3
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def read_wav_float_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Only int16 PCM WAV is supported, got sample_width={sample_width}: {path}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def write_wav_float_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    duration = len(audio) / float(src_rate)
    dst_len = int(round(duration * dst_rate))
    src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }
