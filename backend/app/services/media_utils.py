from __future__ import annotations

import json
import subprocess
from pathlib import Path


class MediaTooLong(Exception):
    pass


def ffprobe_duration_seconds(path: str | Path) -> float:
    path = str(path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    data = json.loads(p.stdout or "{}")
    dur = data.get("format", {}).get("duration")
    if dur is None:
        raise RuntimeError("ffprobe duration missing")
    return float(dur)


def enforce_media_duration(path: str | Path, *, kind: str, max_seconds: int) -> float:
    dur = ffprobe_duration_seconds(path)
    if dur > max_seconds:
        raise MediaTooLong(f"{kind} duration {dur:.1f}s exceeds {max_seconds}s limit")
    return dur


def extract_audio_from_video(video_path: str | Path, out_wav_path: str | Path) -> Path:
    """
    Extract speech-friendly mono 16kHz wav from video.

    Key change (Phase 5 fix):
    - Add filters to improve speech recognition:
      * highpass/lowpass removes rumble + hiss
      * volume boost helps quiet WhatsApp videos
    """
    video_path = Path(video_path)
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        # Speech-friendly filter chain
        "-af",
        "highpass=f=80,lowpass=f=8000,volume=2.5",
        str(out_wav_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg extract_audio failed: {p.stderr.strip()}")
    return out_wav_path


def convert_audio_to_wav(input_audio_path: str | Path, out_wav_path: str | Path) -> Path:
    """
    Convert arbitrary audio to speech-friendly mono 16kHz wav.
    (Used by audio uploads; video uses extract_audio_from_video.)
    """
    input_audio_path = Path(input_audio_path)
    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "highpass=f=80,lowpass=f=8000,volume=2.5",
        str(out_wav_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg convert_audio failed: {p.stderr.strip()}")
    return out_wav_path


def segment_audio_to_wavs(input_audio_path: str | Path, out_dir: str | Path, segment_seconds: int) -> list[Path]:
    """
    Segment a wav (or other audio) into N-second mono 16kHz wav chunks,
    using the same speech-friendly filter chain.
    """
    input_audio_path = Path(input_audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pattern = str(out_dir / "segment_%03d.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "highpass=f=80,lowpass=f=8000,volume=2.5",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        out_pattern,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg segment failed: {p.stderr.strip()}")

    segments = sorted(out_dir.glob("segment_*.wav"))
    if not segments:
        raise RuntimeError("No segments produced")
    return segments
