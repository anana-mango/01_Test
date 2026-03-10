# custom_tmos/utils_io.py
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly


LOGGER = logging.getLogger(__name__)


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a dict at top-level: {path}")
    return data


def write_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str: str, root_dir: Optional[Path] = None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    if root_dir is None:
        return p.resolve()
    return (root_dir / p).resolve()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        LOGGER.warning("Torch seed setup skipped.", exc_info=True)


def load_audio(
    path: Path,
    mono: bool = False,
    sr: Optional[int] = None,
    channel: Optional[int] = None,
    dtype: str = "float32",
) -> Tuple[np.ndarray, int]:
    audio, file_sr = sf.read(str(path), always_2d=True, dtype=dtype)
    # audio shape: [samples, channels]
    if channel is not None:
        if channel < 0 or channel >= audio.shape[1]:
            raise ValueError(f"Invalid channel={channel} for {path}, channels={audio.shape[1]}")
        audio = audio[:, [channel]]

    if mono:
        audio = np.mean(audio, axis=1, keepdims=True)

    if sr is not None and sr != file_sr:
        audio = fast_resample(audio, src_sr=file_sr, dst_sr=sr)
        file_sr = sr

    return audio, file_sr


def fast_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    from math import gcd

    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    if audio.ndim == 1:
        return resample_poly(audio, up, down).astype(np.float32)
    out = []
    for ch in range(audio.shape[1]):
        out.append(resample_poly(audio[:, ch], up, down))
    return np.stack(out, axis=1).astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def match_length(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def build_stereo_ref_deg_wav(
    ref_path: Path,
    deg_path: Path,
    out_path: Path,
    out_sr: Optional[int] = None,
    ref_channel: Optional[int] = None,
    deg_channel: Optional[int] = None,
    trim_to_shorter: bool = True,
    ref_left: bool = True,
) -> Path:
    """
    Utility for your legacy 2ch input requirement.
    Default layout:
      ch0 = ref, ch1 = deg
    """
    ref, sr_ref = load_audio(ref_path, mono=True, sr=out_sr, channel=ref_channel)
    deg, sr_deg = load_audio(deg_path, mono=True, sr=out_sr, channel=deg_channel)

    if sr_ref != sr_deg:
        raise ValueError(f"Sampling rate mismatch after resample: ref={sr_ref}, deg={sr_deg}")

    ref_1d = ref[:, 0]
    deg_1d = deg[:, 0]

    if trim_to_shorter:
        ref_1d, deg_1d = match_length(ref_1d, deg_1d)
    else:
        n = max(len(ref_1d), len(deg_1d))
        ref_pad = np.zeros(n, dtype=np.float32)
        deg_pad = np.zeros(n, dtype=np.float32)
        ref_pad[: len(ref_1d)] = ref_1d
        deg_pad[: len(deg_1d)] = deg_1d
        ref_1d, deg_1d = ref_pad, deg_pad

    stereo = np.stack([ref_1d, deg_1d], axis=1) if ref_left else np.stack([deg_1d, ref_1d], axis=1)
    save_audio(out_path, stereo.astype(np.float32), sr_ref)
    return out_path


def validate_audio_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Only .wav is supported here: {path}")


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")