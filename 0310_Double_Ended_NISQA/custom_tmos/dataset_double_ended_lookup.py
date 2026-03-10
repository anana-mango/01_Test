# custom_tmos/dataset_double_ended_lookup.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .lookup import LookupConfig, ReferenceLookupEngine
from .utils_io import resolve_path
from .utils_mapping import estimate_alignment_seconds, safe_str

LOGGER = logging.getLogger(__name__)


def read_csv_resolve_paths(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def _resolve_deg_paths(df: pd.DataFrame, root_dir: Optional[Path], deg_column: str) -> pd.DataFrame:
    if deg_column not in df.columns:
        raise ValueError(f"Missing degraded path column: {deg_column}")
    df = df.copy()
    df[deg_column] = df[deg_column].apply(lambda x: str(resolve_path(str(x), root_dir)))
    return df


def build_training_manifest(
    train_csv: Path,
    val_csv: Path,
    output_csv: Path,
    deg_column: str = "filepath_deg",
    tmos_column: str = "tmos",
    root_dir: Optional[Path] = None,
    lookup_cfg: Optional[LookupConfig] = None,
    validate_ref_exists: bool = True,
) -> pd.DataFrame:
    df_train = read_csv_resolve_paths(train_csv)
    df_val = read_csv_resolve_paths(val_csv)

    df_train = _resolve_deg_paths(df_train, root_dir, deg_column)
    df_val = _resolve_deg_paths(df_val, root_dir, deg_column)

    df_train["db"] = "train"
    df_val["db"] = "val"

    full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    if tmos_column not in full.columns:
        raise ValueError(f"Missing target column: {tmos_column}")

    if "filepath_ref" not in full.columns or full["filepath_ref"].fillna("").eq("").any():
        if lookup_cfg is None:
            raise ValueError("filepath_ref missing and no lookup_cfg provided.")
        engine = ReferenceLookupEngine(lookup_cfg)
        full = engine.resolve_dataframe(full)

    if validate_ref_exists:
        missing = full["filepath_ref"].fillna("").eq("").sum()
        if missing > 0:
            raise ValueError(f"{missing} rows still do not have filepath_ref after lookup.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_csv, index=False)
    LOGGER.info("Training manifest saved to %s (%d rows)", output_csv, len(full))
    return full


def build_inference_manifest_from_csv(
    infer_csv: Path,
    output_csv: Path,
    deg_column: str = "filepath_deg",
    root_dir: Optional[Path] = None,
    lookup_cfg: Optional[LookupConfig] = None,
) -> pd.DataFrame:
    df = read_csv_resolve_paths(infer_csv)
    df = _resolve_deg_paths(df, root_dir, deg_column)

    if "filepath_ref" not in df.columns or df["filepath_ref"].fillna("").eq("").any():
        if lookup_cfg is None:
            raise ValueError("Inference CSV lacks filepath_ref and no lookup_cfg was provided.")
        engine = ReferenceLookupEngine(lookup_cfg)
        df = engine.resolve_dataframe(df)
    else:
        df["lookup_matched"] = True
        df["lookup_source"] = "explicit_filepath_ref"
        df["lookup_confidence"] = 1.0
        df["lookup_message"] = "filepath_ref already supplied in CSV."
        df["lookup_key"] = ""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    LOGGER.info("Inference manifest saved to %s (%d rows)", output_csv, len(df))
    return df


def build_inference_manifest_from_single_file(
    deg_path: Path,
    output_csv: Path,
    root_dir: Optional[Path] = None,
    lookup_cfg: Optional[LookupConfig] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    rec: Dict[str, Any] = {"filepath_deg": str(resolve_path(str(deg_path), root_dir))}
    if extra_metadata:
        rec.update(extra_metadata)
    df = pd.DataFrame([rec])

    if lookup_cfg is None and safe_str(rec.get("filepath_ref")) == "":
        raise ValueError("Single-file inference needs filepath_ref or lookup_cfg.")

    if lookup_cfg is not None:
        engine = ReferenceLookupEngine(lookup_cfg)
        df = engine.resolve_dataframe(df)
    else:
        df["lookup_matched"] = True
        df["lookup_source"] = "explicit_filepath_ref"
        df["lookup_confidence"] = 1.0
        df["lookup_message"] = "filepath_ref supplied explicitly."
        df["lookup_key"] = ""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def build_inference_manifest_from_dir(
    deg_dir: Path,
    output_csv: Path,
    glob_pattern: str = "*.wav",
    root_dir: Optional[Path] = None,
    lookup_cfg: Optional[LookupConfig] = None,
) -> pd.DataFrame:
    files = sorted(deg_dir.rglob(glob_pattern))
    if not files:
        raise ValueError(f"No files found in directory: {deg_dir}")

    df = pd.DataFrame({"filepath_deg": [str(resolve_path(str(p), root_dir)) for p in files]})

    if lookup_cfg is None:
        raise ValueError("Directory inference requires lookup_cfg.")
    engine = ReferenceLookupEngine(lookup_cfg)
    df = engine.resolve_dataframe(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def append_alignment_diagnostics(
    df: pd.DataFrame,
    output_csv: Path,
    sr: int = 16000,
    max_seconds: float = 3.0,
) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        ref = safe_str(row.get("filepath_ref"))
        deg = safe_str(row.get("filepath_deg"))
        if ref and deg:
            try:
                diag = estimate_alignment_seconds(Path(ref), Path(deg), target_sr=sr, max_seconds=max_seconds)
            except Exception as exc:
                diag = {
                    "estimated_lag_samples": None,
                    "estimated_lag_seconds": None,
                    "alignment_peak_corr": None,
                    "alignment_diag_error": str(exc),
                }
            rec.update(diag)
        records.append(rec)

    out = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
