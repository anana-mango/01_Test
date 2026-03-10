# custom_tmos/lookup.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils_io import resolve_path
from .utils_mapping import extract_named_groups, normalize_key_parts, safe_str

LOGGER = logging.getLogger(__name__)


@dataclass
class LookupResult:
    filepath_ref: Optional[Path]
    matched: bool
    source: str
    confidence: float
    message: str
    matched_key: Optional[Tuple[str, ...]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LookupConfig:
    mapping_csv: Optional[Path] = None
    fixed_ref_path: Optional[Path] = None
    strict: bool = True
    key_columns: Tuple[str, ...] = ("test_id", "scenario_id")
    regex_pattern: Optional[str] = None
    priority_column: str = "priority"
    active_column: Optional[str] = "is_active"
    version_column: Optional[str] = "version"
    root_dir: Optional[Path] = None


class ReferenceLookupEngine:
    def __init__(self, config: LookupConfig):
        self.config = config
        self.mapping_df: Optional[pd.DataFrame] = None
        self.mapping_dict: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        if self.config.mapping_csv is not None:
            self._load_mapping_csv(self.config.mapping_csv)

    def _load_mapping_csv(self, mapping_csv: Path) -> None:
        mapping_path = resolve_path(str(mapping_csv), self.config.root_dir)
        if not mapping_path.exists():
            raise FileNotFoundError(f"reference mapping CSV not found: {mapping_path}")

        df = pd.read_csv(mapping_path)
        required = list(self.config.key_columns) + ["filepath_ref"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in mapping CSV: {missing}")

        if self.config.active_column and self.config.active_column in df.columns:
            df = df[df[self.config.active_column].astype(str).isin(["1", "True", "true", "TRUE"])].copy()

        if self.config.priority_column in df.columns:
            df = df.sort_values(self.config.priority_column, ascending=True).reset_index(drop=True)

        self.mapping_df = df
        self.mapping_dict = {}

        for _, row in df.iterrows():
            key = normalize_key_parts([row[c] for c in self.config.key_columns])
            if key not in self.mapping_dict:
                self.mapping_dict[key] = row.to_dict()

        LOGGER.info("Loaded reference map: %d entries from %s", len(self.mapping_dict), mapping_path)

    def _try_explicit_filepath_ref(self, row: Dict[str, Any]) -> Optional[LookupResult]:
        val = safe_str(row.get("filepath_ref"))
        if not val:
            return None
        p = resolve_path(val, self.config.root_dir)
        if p.exists():
            return LookupResult(
                filepath_ref=p,
                matched=True,
                source="explicit_filepath_ref",
                confidence=1.0,
                message="Used filepath_ref from input row.",
            )
        return LookupResult(
            filepath_ref=None,
            matched=False,
            source="explicit_filepath_ref",
            confidence=0.0,
            message=f"filepath_ref was given but file does not exist: {p}",
        )

    def _row_to_key(self, row: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
        vals = [safe_str(row.get(c)) for c in self.config.key_columns]
        if all(v != "" for v in vals):
            return tuple(vals)
        return None

    def _try_mapping_key(self, row: Dict[str, Any]) -> Optional[LookupResult]:
        if not self.mapping_dict:
            return None
        key = self._row_to_key(row)
        if key is None:
            return None
        rec = self.mapping_dict.get(key)
        if rec is None:
            return None
        ref_path = resolve_path(str(rec["filepath_ref"]), self.config.root_dir)
        return LookupResult(
            filepath_ref=ref_path if ref_path.exists() else None,
            matched=ref_path.exists(),
            source="mapping_csv_exact",
            confidence=1.0 if ref_path.exists() else 0.0,
            message=f"Matched mapping CSV with key={key}",
            matched_key=key,
            metadata=rec,
        )

    def _try_regex_extraction(self, row: Dict[str, Any]) -> Optional[LookupResult]:
        if not self.mapping_dict or not self.config.regex_pattern:
            return None

        deg_path = safe_str(row.get("filepath_deg"))
        if not deg_path:
            return None

        filename = Path(deg_path).name
        groups = extract_named_groups(self.config.regex_pattern, filename)
        if not groups:
            return None

        key = tuple(safe_str(groups.get(c)) for c in self.config.key_columns)
        if any(v == "" for v in key):
            return None

        rec = self.mapping_dict.get(key)
        if rec is None:
            return None

        ref_path = resolve_path(str(rec["filepath_ref"]), self.config.root_dir)
        return LookupResult(
            filepath_ref=ref_path if ref_path.exists() else None,
            matched=ref_path.exists(),
            source="regex_to_mapping",
            confidence=0.8 if ref_path.exists() else 0.0,
            message=f"Matched by regex extracted key={key}",
            matched_key=key,
            metadata={**rec, "regex_groups": groups},
        )

    def _try_fixed_ref(self) -> Optional[LookupResult]:
        if self.config.fixed_ref_path is None:
            return None
        ref_path = resolve_path(str(self.config.fixed_ref_path), self.config.root_dir)
        return LookupResult(
            filepath_ref=ref_path if ref_path.exists() else None,
            matched=ref_path.exists(),
            source="fixed_ref_fallback",
            confidence=0.4 if ref_path.exists() else 0.0,
            message=f"Fallback to fixed ref: {ref_path}",
        )

    def resolve_row(self, row: Dict[str, Any]) -> LookupResult:
        candidates = [
            self._try_explicit_filepath_ref(row),
            self._try_mapping_key(row),
            self._try_regex_extraction(row),
            self._try_fixed_ref(),
        ]
        for cand in candidates:
            if cand is not None and cand.matched:
                return cand

        messages = [c.message for c in candidates if c is not None]
        msg = " | ".join(messages) if messages else "No lookup rule applied."
        result = LookupResult(
            filepath_ref=None,
            matched=False,
            source="unmatched",
            confidence=0.0,
            message=msg,
        )
        if self.config.strict:
            raise LookupError(msg)
        return result

    def resolve_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out_records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            result = self.resolve_row(row_dict)
            rec = dict(row_dict)
            rec["filepath_ref"] = str(result.filepath_ref) if result.filepath_ref else ""
            rec["lookup_matched"] = result.matched
            rec["lookup_source"] = result.source
            rec["lookup_confidence"] = result.confidence
            rec["lookup_message"] = result.message
            if result.matched_key is not None:
                rec["lookup_key"] = "|".join(result.matched_key)
            else:
                rec["lookup_key"] = ""
            out_records.append(rec)
        return pd.DataFrame(out_records)