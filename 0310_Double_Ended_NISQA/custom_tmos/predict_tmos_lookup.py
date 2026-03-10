# custom_tmos/predict_tmos_lookup.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from nisqa.NISQA_model import nisqaModel

from .dataset_double_ended_lookup import (
    append_alignment_diagnostics,
    build_inference_manifest_from_csv,
    build_inference_manifest_from_dir,
    build_inference_manifest_from_single_file,
)
from .lookup import LookupConfig
from .utils_io import ensure_dir, merge_dict, read_yaml, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)


def make_lookup_config(cfg: Dict[str, Any], root_dir: Path) -> Optional[LookupConfig]:
    lookup = cfg.get("lookup", {})
    mapping_csv = lookup.get("mapping_csv")
    fixed_ref_path = lookup.get("fixed_ref_path")

    if not mapping_csv and not fixed_ref_path:
        return None

    return LookupConfig(
        mapping_csv=(root_dir / mapping_csv) if mapping_csv else None,
        fixed_ref_path=(root_dir / fixed_ref_path) if fixed_ref_path else None,
        strict=bool(lookup.get("strict", True)),
        key_columns=tuple(lookup.get("key_columns", ["test_id", "scenario_id"])),
        regex_pattern=lookup.get("regex_pattern"),
        priority_column=lookup.get("priority_column", "priority"),
        active_column=lookup.get("active_column", "is_active"),
        version_column=lookup.get("version_column", "version"),
        root_dir=root_dir,
    )


def build_predict_args(cfg: Dict[str, Any], root_dir: Path, manifest_path: Path) -> Dict[str, Any]:
    out = dict(cfg)
    out["mode"] = "predict_csv"
    out["data_dir"] = str(root_dir)
    out["csv_file"] = str(manifest_path.relative_to(root_dir))
    out["csv_deg"] = out.get("csv_deg", "filepath_deg")
    out["csv_ref"] = out.get("csv_ref", "filepath_ref")
    out["tr_bs_val"] = int(out.get("bs", out.get("tr_bs_val", 8)))
    out["tr_num_workers"] = int(out.get("num_workers", out.get("tr_num_workers", 0)))
    out["double_ended"] = True
    out["dim"] = False
    out["name"] = out.get("name", "TMOS_DE_PRED")
    return out


def add_default_tmos_column_names(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    The native NISQA predict() returns ds_val.df with predicted columns injected by the framework.
    Column name can differ depending on checkpoint/setup, so normalize here.
    """
    df = pred_df.copy()
    pred_candidates = [c for c in df.columns if c.lower() in {"mos_pred", "mos_hat", "y_hat", "mos"}]
    if "tmos_pred" not in df.columns:
        if pred_candidates:
            df["tmos_pred"] = df[pred_candidates[0]]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, type=str, help="Path to custom TMOS predict YAML")
    parser.add_argument("--mode", choices=["single", "dir", "csv"], default=None)
    parser.add_argument("--deg", type=str, default=None)
    parser.add_argument("--deg_dir", type=str, default=None)
    parser.add_argument("--infer_csv", type=str, default=None)
    parser.add_argument("--test_id", type=str, default=None)
    parser.add_argument("--scenario_id", type=str, default=None)
    args_ns = parser.parse_args()

    yaml_path = Path(args_ns.yaml).resolve()
    custom_cfg = read_yaml(yaml_path)
    root_dir = yaml_path.parent.parent if yaml_path.parent.name == "config" else yaml_path.parent

    output_dir = ensure_dir(Path(custom_cfg["output_dir"]).resolve())
    setup_logging(output_dir / "logs" / "predict_tmos_lookup.log")
    seed_everything(int(custom_cfg.get("seed", 42)))

    lookup_cfg = make_lookup_config(custom_cfg, root_dir)

    manifest_dir = ensure_dir(output_dir / "manifests")
    manifest_path = manifest_dir / "inference_resolved.csv"

    mode = args_ns.mode or custom_cfg.get("input_mode", "csv")

    if mode == "single":
        if not args_ns.deg and "deg" not in custom_cfg:
            raise ValueError("single mode requires --deg or deg in YAML")
        deg = Path(args_ns.deg or custom_cfg["deg"]).resolve()
        extra = {}
        if args_ns.test_id or custom_cfg.get("test_id"):
            extra["test_id"] = args_ns.test_id or custom_cfg.get("test_id")
        if args_ns.scenario_id or custom_cfg.get("scenario_id"):
            extra["scenario_id"] = args_ns.scenario_id or custom_cfg.get("scenario_id")
        if custom_cfg.get("filepath_ref"):
            extra["filepath_ref"] = custom_cfg["filepath_ref"]

        build_inference_manifest_from_single_file(
            deg_path=deg,
            output_csv=manifest_path,
            root_dir=root_dir,
            lookup_cfg=lookup_cfg,
            extra_metadata=extra,
        )

    elif mode == "dir":
        deg_dir = Path(args_ns.deg_dir or custom_cfg["deg_dir"]).resolve()
        build_inference_manifest_from_dir(
            deg_dir=deg_dir,
            output_csv=manifest_path,
            glob_pattern=custom_cfg.get("glob_pattern", "*.wav"),
            root_dir=root_dir,
            lookup_cfg=lookup_cfg,
        )

    elif mode == "csv":
        infer_csv = Path(args_ns.infer_csv or custom_cfg["infer_csv"]).resolve()
        build_inference_manifest_from_csv(
            infer_csv=infer_csv,
            output_csv=manifest_path,
            deg_column=custom_cfg.get("csv_deg", "filepath_deg"),
            root_dir=root_dir,
            lookup_cfg=lookup_cfg,
        )
    else:
        raise NotImplementedError(mode)

    pred_args = build_predict_args(custom_cfg, root_dir, manifest_path)
    model = nisqaModel(pred_args)

    # Defensive patch for csv_ref handling.
    model.args["csv_ref"] = pred_args["csv_ref"]
    model.args["double_ended"] = True
    model.args["model"] = "NISQA_DE"
    model._loadDatasets()

    LOGGER.info("Running prediction on resolved manifest.")
    pred_df = model.predict()
    pred_df = add_default_tmos_column_names(pred_df)

    resolved_df = pd.read_csv(manifest_path)
    merged = pred_df.copy()

    for col in resolved_df.columns:
        if col not in merged.columns:
            merged[col] = resolved_df[col]

    if bool(custom_cfg.get("append_alignment_diag", False)):
        align_path = output_dir / "prediction_with_alignment_diag.csv"
        merged = append_alignment_diagnostics(
            merged,
            output_csv=align_path,
            sr=int(custom_cfg.get("alignment_sr", 16000)),
            max_seconds=float(custom_cfg.get("alignment_max_seconds", 3.0)),
        )

    final_csv = output_dir / "TMOS_predictions.csv"
    merged.to_csv(final_csv, index=False)

    LOGGER.info("Prediction CSV saved to %s", final_csv)
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()