from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from nisqa.NISQA_model import nisqaModel

from .dataset_double_ended_lookup import build_training_manifest
from .lookup import LookupConfig
from .utils_io import ensure_dir, merge_dict, read_yaml, seed_everything, setup_logging

LOGGER = logging.getLogger(__name__)


def make_lookup_config(cfg: Dict[str, Any], root_dir: Path) -> LookupConfig | None:
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


def patch_args_for_double_ended_tmos(args: Dict[str, Any], output_dir: Path, manifest_name: str) -> Dict[str, Any]:
    args = dict(args)

    # custom TMOS / new model training 기준으로 강제 설정
    args["name"] = args.get("name", "TMOS_DE")
    args["model"] = "NISQA_DE"
    args["output_dir"] = str(output_dir)
    args["csv_file"] = manifest_name

    # custom manifest 컬럼명 강제
    args["csv_deg"] = "filepath_deg"
    args["csv_ref"] = "filepath_ref"
    args["csv_mos_train"] = "tmos"
    args["csv_mos_val"] = "tmos"

    # custom manifest split 이름 강제
    args["csv_db_train"] = ["train"]
    args["csv_db_val"] = ["val"]

    args["mode"] = "main"
    args["dim"] = False
    args["double_ended"] = True

    # 완전 새 모델 학습
    args["pretrained_model"] = ""

    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, type=str, help="Path to custom TMOS training YAML")
    args_ns = parser.parse_args()

    yaml_path = Path(args_ns.yaml).resolve()
    custom_cfg = read_yaml(yaml_path)
    root_dir = yaml_path.parent.parent if yaml_path.parent.name == "config" else yaml_path.parent

    output_dir = ensure_dir(Path(custom_cfg["output_dir"]).resolve())
    setup_logging(output_dir / "logs" / "train_tmos_lookup.log")

    seed = int(custom_cfg.get("seed", 42))
    seed_everything(seed)

    base_yaml = custom_cfg.get("nisqa_base_yaml")
    if not base_yaml:
        raise ValueError("nisqa_base_yaml must be set, e.g. config/train_nisqa_double_ended.yaml")

    base_cfg = read_yaml((root_dir / base_yaml).resolve())
    merged_cfg = merge_dict(base_cfg, custom_cfg)

    lookup_cfg = make_lookup_config(merged_cfg, root_dir)

    manifest_dir = ensure_dir(output_dir / "manifests")
    manifest_path = manifest_dir / "train_val_resolved.csv"

    train_csv = Path(merged_cfg["train_csv"]).resolve() if Path(merged_cfg["train_csv"]).is_absolute() else (root_dir / merged_cfg["train_csv"]).resolve()
    val_csv = Path(merged_cfg["val_csv"]).resolve() if Path(merged_cfg["val_csv"]).is_absolute() else (root_dir / merged_cfg["val_csv"]).resolve()

    tmos_column = merged_cfg.get("tmos_column", "tmos")
    deg_column = "filepath_deg"
    ref_column = "filepath_ref"

    build_training_manifest(
        train_csv=train_csv,
        val_csv=val_csv,
        output_csv=manifest_path,
        deg_column=deg_column,
        ref_column=ref_column,
        tmos_column=tmos_column,
        root_dir=root_dir,
        lookup_cfg=lookup_cfg,
        validate_ref_exists=True,
        train_db_label="train",
        val_db_label="val",
    )

    nisqa_args = patch_args_for_double_ended_tmos(
        args=merged_cfg,
        output_dir=output_dir,
        manifest_name=str(manifest_path.relative_to(root_dir)),
    )
    nisqa_args["data_dir"] = str(root_dir)

    LOGGER.info("Initializing NISQA double-ended TMOS training...")
    LOGGER.info("csv_file=%s", nisqa_args["csv_file"])
    LOGGER.info("csv_deg=%s", nisqa_args["csv_deg"])
    LOGGER.info("csv_ref=%s", nisqa_args["csv_ref"])
    LOGGER.info("csv_mos_train=%s", nisqa_args["csv_mos_train"])
    LOGGER.info("csv_mos_val=%s", nisqa_args["csv_mos_val"])
    LOGGER.info("csv_db_train=%s", nisqa_args["csv_db_train"])
    LOGGER.info("csv_db_val=%s", nisqa_args["csv_db_val"])
    LOGGER.info("pretrained_model='%s'", nisqa_args["pretrained_model"])

    model = nisqaModel(nisqa_args)

    # 방어적으로 다시 고정
    model.args["csv_ref"] = "filepath_ref"
    model.args["csv_deg"] = "filepath_deg"
    model.args["csv_mos_train"] = "tmos"
    model.args["csv_mos_val"] = "tmos"
    model.args["csv_db_train"] = ["train"]
    model.args["csv_db_val"] = ["val"]
    model.args["double_ended"] = True
    model.args["model"] = "NISQA_DE"
    model.args["pretrained_model"] = ""

    LOGGER.info("Start training.")
    model.train()
    LOGGER.info("Training completed.")


if __name__ == "__main__":
    main()