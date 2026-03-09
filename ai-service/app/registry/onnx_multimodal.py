from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np


def _softmax_2d(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _np_f32(x: Any) -> np.ndarray:
    arr = x if isinstance(x, np.ndarray) else np.asarray(x)
    return arr.astype(np.float32, copy=False)


def _np_bool(x: Any) -> np.ndarray:
    arr = x if isinstance(x, np.ndarray) else np.asarray(x)
    return arr.astype(np.bool_, copy=False)


def onnx_predict_multimodal(
    session: Any,
    text_emb: Any,
    ocr_emb: Any,
    audio_emb: Any,
    table_emb: Any,
    mask: Any,
    temperature: float = 1.0,
    *,
    output_index: int = 0,
) -> Tuple[int, np.ndarray]:
    """
    Single-head multimodal predictor.

    Inputs:
      - text_emb/ocr_emb/audio_emb: [B,384] float32
      - table_emb: [B,64] float32
      - mask: [B,4] bool (True means modality present)

    Returns:
      - pred (int)
      - probs (np.ndarray shape [num_classes])
    """
    text_emb = _np_f32(text_emb)
    ocr_emb = _np_f32(ocr_emb)
    audio_emb = _np_f32(audio_emb)
    table_emb = _np_f32(table_emb)

    if mask is None:
        mask = np.array([[True, False, False, False]], dtype=np.bool_)
    else:
        mask = _np_bool(mask)

    if text_emb.ndim != 2 or text_emb.shape[1] != 384:
        raise ValueError(f"text_emb must be [B,384], got {text_emb.shape}")
    if ocr_emb.ndim != 2 or ocr_emb.shape[1] != 384:
        raise ValueError(f"ocr_emb must be [B,384], got {ocr_emb.shape}")
    if audio_emb.ndim != 2 or audio_emb.shape[1] != 384:
        raise ValueError(f"audio_emb must be [B,384], got {audio_emb.shape}")
    if table_emb.ndim != 2 or table_emb.shape[1] != 64:
        raise ValueError(f"table_emb must be [B,64], got {table_emb.shape}")
    if mask.ndim != 2 or mask.shape[1] != 4:
        raise ValueError(f"mask must be [B,4], got {mask.shape}")

    inputs = {
        "text_emb": text_emb.astype(np.float32, copy=False),
        "ocr_emb": ocr_emb.astype(np.float32, copy=False),
        "audio_emb": audio_emb.astype(np.float32, copy=False),
        "table_emb": table_emb.astype(np.float32, copy=False),
        # ✅ ONNX expects bool
        "mask": mask.astype(np.bool_, copy=False),
    }

    outputs = session.run(None, inputs)
    if output_index >= len(outputs):
        raise IndexError(f"output_index={output_index} but model returned {len(outputs)} outputs")

    logits = np.asarray(outputs[output_index], dtype=np.float32)
    logits = logits / max(float(temperature), 1e-6)

    probs = _softmax_2d(logits)
    pred = int(np.argmax(probs, axis=1)[0])
    return pred, probs[0]


def onnx_predict_multimodal_multitask(
    session: Any,
    text_emb: Any,
    ocr_emb: Any,
    audio_emb: Any,
    table_emb: Any,
    mask: Any,
    *,
    head_order: List[str],
    head_temperatures: Optional[Dict[str, float]] = None,
    fallback_temperature: float = 1.0,
) -> Dict[str, Dict[str, Union[int, float, List[float]]]]:
    """
    Multi-head predictor: outputs = [logits_head0, logits_head1, ...]

    Returns per head:
      { head: { "pred": int, "confidence": float, "probs": [...], "temperature": float } }
    """
    text_emb = _np_f32(text_emb)
    ocr_emb = _np_f32(ocr_emb)
    audio_emb = _np_f32(audio_emb)
    table_emb = _np_f32(table_emb)

    if mask is None:
        mask = np.array([[True, False, False, False]], dtype=np.bool_)
    else:
        mask = _np_bool(mask)

    inputs = {
        "text_emb": text_emb.astype(np.float32, copy=False),
        "ocr_emb": ocr_emb.astype(np.float32, copy=False),
        "audio_emb": audio_emb.astype(np.float32, copy=False),
        "table_emb": table_emb.astype(np.float32, copy=False),
        # ✅ ONNX expects bool
        "mask": mask.astype(np.bool_, copy=False),
    }

    outputs = session.run(None, inputs)

    res: Dict[str, Dict[str, Union[int, float, List[float]]]] = {}
    for i, head in enumerate(head_order):
        if i >= len(outputs):
            break

        logits = np.asarray(outputs[i], dtype=np.float32)

        t = float(fallback_temperature)
        if head_temperatures and head in head_temperatures:
            t = max(float(head_temperatures[head]), 1e-6)

        logits = logits / t
        probs = _softmax_2d(logits)
        pred = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])

        res[head] = {
            "pred": pred,
            "confidence": conf,
            "temperature": t,
            "probs": probs[0].astype(float).tolist(),
        }

    return res