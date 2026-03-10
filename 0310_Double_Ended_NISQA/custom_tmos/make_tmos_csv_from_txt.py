from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TMOS CSV from wav_score_mapping.txt and wav folder."
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="Project2/NISQA_master/data/wavs",
        help="Directory containing input WAV files",
    )
    parser.add_argument(
        "--mapping_txt",
        type=str,
        default="Project2/NISQA_master/data/wavs/wav_score_mapping.txt",
        help="Path to wav_score_mapping.txt",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="Project2/NISQA_master/data/metadata_tmos.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="*.wav",
        help="Glob pattern for WAV files",
    )
    parser.add_argument(
        "--db_label",
        type=str,
        default="train",
        help="Value to write into the db column",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if any wav file cannot be matched to TMOS",
    )
    return parser.parse_args()


def parse_mapping_txt(mapping_txt: Path) -> Dict[Tuple[str, int], float]:
    """
    Parse lines like:
    0309_M1Q_data, Index 1        3.74
    """
    if not mapping_txt.exists():
        raise FileNotFoundError(f"Mapping txt not found: {mapping_txt}")

    mapping: Dict[Tuple[str, int], float] = {}

    pattern = re.compile(
        r"^\s*(?P<basename>[^,]+)\s*,\s*Index\s*(?P<index>\d+)\s+(?P<tmos>\d+(?:\.\d+)?)\s*$"
    )

    with mapping_txt.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if not match:
                raise ValueError(
                    f"Failed to parse line {line_no} in {mapping_txt}:\n{line}"
                )

            basename = match.group("basename").strip()
            index = int(match.group("index"))
            tmos = float(match.group("tmos"))

            key = (basename, index)
            if key in mapping:
                raise ValueError(
                    f"Duplicate mapping found for key={key} in {mapping_txt}"
                )
            mapping[key] = tmos

    return mapping


def parse_wav_filename(wav_name: str) -> Tuple[str, int]:
    """
    Parse filename like:
    0309_M1Q_data_SMD202_Index1_TimeSignal.wav

    Returns:
        basename = 0309_M1Q_data
        index = 1
    """
    pattern = re.compile(
        r"^(?P<basename>.+?)_SMD\d+_Index(?P<index>\d+)_TimeSignal\.wav$",
        re.IGNORECASE,
    )
    match = pattern.match(wav_name)
    if not match:
        raise ValueError(f"Unexpected WAV filename format: {wav_name}")

    basename = match.group("basename")
    index = int(match.group("index"))
    return basename, index


def inspect_wav(wav_path: Path) -> Tuple[int, int]:
    """
    Returns:
        channels, samplerate
    """
    info = sf.info(str(wav_path))
    return info.channels, info.samplerate


def build_records(
    wav_dir: Path,
    mapping: Dict[Tuple[str, int], float],
    glob_pattern: str,
    db_label: str,
    strict: bool,
) -> List[dict]:
    wav_files = sorted(wav_dir.glob(glob_pattern))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in: {wav_dir}")

    records: List[dict] = []
    unmatched_files: List[str] = []

    for wav_path in wav_files:
        if not wav_path.is_file():
            continue

        try:
            basename, index = parse_wav_filename(wav_path.name)
        except ValueError:
            if strict:
                raise
            unmatched_files.append(wav_path.name)
            continue

        key = (basename, index)
        if key not in mapping:
            if strict:
                raise KeyError(f"No TMOS mapping found for wav: {wav_path.name}, key={key}")
            unmatched_files.append(wav_path.name)
            continue

        channels, samplerate = inspect_wav(wav_path)

        records.append(
            {
                "filepath": str(wav_path.as_posix()),
                "filename": wav_path.name,
                "stem": wav_path.stem,
                "basename": basename,
                "index": index,
                "tmos": mapping[key],
                "db": db_label,
                "channels": channels,
                "samplerate": samplerate,
            }
        )

    if unmatched_files:
        print("\n[WARNING] Unmatched WAV files:")
        for name in unmatched_files:
            print(f"  - {name}")

    return records


def main() -> None:
    args = parse_args()

    wav_dir = Path(args.wav_dir)
    mapping_txt = Path(args.mapping_txt)
    output_csv = Path(args.output_csv)

    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    mapping = parse_mapping_txt(mapping_txt)
    records = build_records(
        wav_dir=wav_dir,
        mapping=mapping,
        glob_pattern=args.glob_pattern,
        db_label=args.db_label,
        strict=args.strict,
    )

    if not records:
        raise RuntimeError("No valid records were created.")

    df = pd.DataFrame(records)
    df = df.sort_values(["basename", "index"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n[INFO] CSV created successfully.")
    print(f"[INFO] Output: {output_csv}")
    print(f"[INFO] Number of rows: {len(df)}")
    print("\n[INFO] Preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()