# custom_tmos/utils_mapping.py
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .utils_io import load_audio, match_length


def safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def normalize_key_parts(parts: Sequence[object]) -> Tuple[str, ...]:
    return tuple(safe_str(p) for p in parts)


def extract_named_groups(pattern: str, text: str) -> Dict[str, str]:
    m = re.search(pattern, text)
    if not m:
        return {}
    return {k: v for k, v in m.groupdict().items() if v is not None}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearsonr_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def srcc_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred)[0])


def first_order_map_fit(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Fit y_true ~= a * y_pred + b
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        return 1.0, 0.0
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def first_order_map_apply(y_pred: np.ndarray, a: float, b: float) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return a * y_pred + b


def first_order_mapped_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = first_order_map_fit(y_true, y_pred)
    y_hat_map = first_order_map_apply(y_pred, a, b)
    return rmse(y_true, y_hat_map)


def estimate_alignment_seconds(
    ref_path: Path,
    deg_path: Path,
    target_sr: int = 16000,
    max_seconds: float = 3.0,
) -> Dict[str, float]:
    """
    Lightweight diagnostics only.
    Does not modify NISQA input.
    """
    ref, sr_ref = load_audio(ref_path, mono=True, sr=target_sr)
    deg, sr_deg = load_audio(deg_path, mono=True, sr=target_sr)
    if sr_ref != sr_deg:
        raise ValueError("Unexpected SR mismatch after resampling.")

    ref_1d, deg_1d = ref[:, 0], deg[:, 0]
    ref_1d, deg_1d = match_length(ref_1d, deg_1d)

    ref_1d = ref_1d - np.mean(ref_1d)
    deg_1d = deg_1d - np.mean(deg_1d)

    max_lag = int(max_seconds * target_sr)
    corr = np.correlate(deg_1d, ref_1d, mode="full")
    center = len(corr) // 2
    lo = max(0, center - max_lag)
    hi = min(len(corr), center + max_lag + 1)
    corr_win = corr[lo:hi]

    best_idx = int(np.argmax(np.abs(corr_win)))
    lag = best_idx + lo - center
    lag_sec = lag / float(target_sr)

    denom = (np.linalg.norm(ref_1d) * np.linalg.norm(deg_1d)) + 1e-12
    peak_corr = float(corr_win[best_idx] / denom)

    return {
        "estimated_lag_samples": int(lag),
        "estimated_lag_seconds": float(lag_sec),
        "alignment_peak_corr": peak_corr,
    }


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "pearson": pearsonr_safe(y_true, y_pred),
        "srcc": srcc_safe(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "rmse_first_order_mapped": first_order_mapped_rmse(y_true, y_pred),
    }