from __future__ import annotations

from typing import Any, Dict, Tuple

# Simple in-process cache so we don't re-load the model every job/chunk
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}


def _pick_device() -> Tuple[str, str]:
    """
    Returns (device, compute_type)
    - GPU optional (CUDA): float16
    - CPU fallback: int8
    """
    try:
        import torch  # optional
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _get_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    key = (model_name, device, compute_type)
    m = _MODEL_CACHE.get(key)
    if m is not None:
        return m

    m = WhisperModel(model_name, device=device, compute_type=compute_type)
    _MODEL_CACHE[key] = m
    return m


def transcribe_with_faster_whisper(audio_path: str, model_name: str = "base") -> Dict[str, Any]:
    """
    Requires: pip install faster-whisper

    Improvements:
    - GPU optional (CUDA), CPU fallback
    - VAD filter enabled (better on noisy/silent media)
    - Model cached per-process (much faster across jobs)
    - Fallback decode if result is suspiciously short
    """
    device, compute_type = _pick_device()
    model = _get_model(model_name, device, compute_type)

    # Pass 1: good defaults for general speech
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segs = []
    texts = []
    for s in segments:
        txt = (s.text or "").strip()
        segs.append({"start": float(s.start), "end": float(s.end), "text": txt})
        if txt:
            texts.append(txt)

    text_out = " ".join(texts).strip()

    # Fallback if output is tiny (common with noisy WhatsApp clips)
    # Pass 2: more aggressive decoding
    if len(text_out) < 10:
        segments2, info2 = model.transcribe(
            audio_path,
            beam_size=1,               # faster, sometimes better on weak audio
            temperature=0.2,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )

        segs = []
        texts = []
        for s in segments2:
            txt = (s.text or "").strip()
            segs.append({"start": float(s.start), "end": float(s.end), "text": txt})
            if txt:
                texts.append(txt)

        text_out = " ".join(texts).strip()
        info = info2  # use latest

    return {
        "text": text_out,
        "segments": segs,
        "meta": {
            "model": f"faster-whisper:{model_name}",
            "device": device,
            "compute_type": compute_type,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "segments_count": len(segs),
            "vad_filter": True,
        },
    }
