# custom_tmos/evaluate_tmos_lookup.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .lookup import LookupConfig
from .predict_tmos_lookup import add_default_tmos_column_names, build_predict_args
from .dataset_double_ended_lookup import build_inference_manifest_from_csv
from .utils_io import ensure_dir, read_yaml, seed_everything, setup_logging
from .utils_mapping import summarize_metrics
from nisqa.NISQA_model import nisqaModel

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, type=str)
    args_ns = parser.parse_args()

    yaml_path = Path(args_ns.yaml).resolve()
    cfg = read_yaml(yaml_path)
    root_dir = yaml_path.parent.parent if yaml_path.parent.name == "config" else yaml_path.parent

    output_dir = ensure_dir(Path(cfg["output_dir"]).resolve())
    setup_logging(output_dir / "logs" / "evaluate_tmos_lookup.log")
    seed_everything(int(cfg.get("seed", 42)))

    eval_csv = Path(cfg["eval_csv"]).resolve() if Path(cfg["eval_csv"]).is_absolute() else (root_dir / cfg["eval_csv"]).resolve()
    tmos_column = cfg.get("tmos_column", "tmos")
    deg_column = cfg.get("csv_deg", "filepath_deg")

    lookup_cfg = make_lookup_config(cfg, root_dir)

    manifest_path = output_dir / "manifests" / "eval_resolved.csv"
    build_inference_manifest_from_csv(
        infer_csv=eval_csv,
        output_csv=manifest_path,
        deg_column=deg_column,
        root_dir=root_dir,
        lookup_cfg=lookup_cfg,
    )

    pred_args = build_predict_args(cfg, root_dir, manifest_path)
    model = nisqaModel(pred_args)
    model.args["csv_ref"] = pred_args["csv_ref"]
    model.args["double_ended"] = True
    model.args["model"] = "NISQA_DE"
    model._loadDatasets()

    pred_df = model.predict()
    pred_df = add_default_tmos_column_names(pred_df)

    if tmos_column not in pred_df.columns:
        original = pd.read_csv(eval_csv)
        if tmos_column in original.columns and len(original) == len(pred_df):
            pred_df[tmos_column] = original[tmos_column].values
        else:
            raise ValueError(f"Ground-truth column '{tmos_column}' not found.")

    if "tmos_pred" not in pred_df.columns:
        raise ValueError("Prediction column 'tmos_pred' could not be inferred from NISQA output.")

    metrics = summarize_metrics(pred_df[tmos_column].to_numpy(), pred_df["tmos_pred"].to_numpy())

    pred_out = output_dir / "TMOS_eval_predictions.csv"
    pred_df.to_csv(pred_out, index=False)

    metrics_out = output_dir / "TMOS_eval_metrics.json"
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    LOGGER.info("Saved eval predictions to %s", pred_out)
    LOGGER.info("Saved eval metrics to %s", metrics_out)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()