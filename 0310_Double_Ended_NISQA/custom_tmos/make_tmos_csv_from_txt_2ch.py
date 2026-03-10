from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split 2ch WAV files into degraded(L) and ref(R), "
            "map TMOS from txt, and create NISQA-ready training CSV."
        )
    )

    parser.add_argument(
        "--wav_dir",
        type=str,
        default="Project2/NISQA_master/data/wavs",
        help="Directory containing original 2ch WAV files",
    )
    parser.add_argument(
        "--mapping_txt",
        type=str,
        default="Project2/NISQA_master/data/wavs/wav_score_mapping.txt",
        help="Path to wav_score_mapping.txt",
    )
    parser.add_argument(
        "--output_deg_dir",
        type=str,
        default="Project2/NISQA_master/data/wavs_deg",
        help="Directory to save split degraded WAV files",
    )
    parser.add_argument(
        "--output_ref_dir",
        type=str,
        default="Project2/NISQA_master/data/wavs_ref",
        help="Directory to save split reference WAV files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="Project2/NISQA_master/data/metadata_tmos_2ch_split.csv",
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
        help="Value for db column in output CSV",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if any file fails parsing or mapping",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite split wav files if they already exist",
    )
    parser.add_argument(
        "--save_float32",
        action="store_true",
        help="Save output wav as FLOAT subtype explicitly",
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
                raise ValueError(f"Duplicate mapping found for key={key}")
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


def inspect_wav(wav_path: Path) -> sf.SoundFile:
    return sf.SoundFile(str(wav_path))


def split_stereo_wav(
    input_wav: Path,
    output_deg_wav: Path,
    output_ref_wav: Path,
    overwrite: bool = False,
    save_float32: bool = False,
) -> Tuple[int, int, int]:
    """
    Assumption:
      Left channel  (ch=0) = degraded
      Right channel (ch=1) = reference

    Returns:
      samplerate, frames, channels_original
    """
    if (output_deg_wav.exists() or output_ref_wav.exists()) and not overwrite:
        info = sf.info(str(input_wav))
        return info.samplerate, info.frames, info.channels

    data, sr = sf.read(str(input_wav), always_2d=True)
    # soundfile returns shape: (samples, channels) for multi-channel audio. 1

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"Expected stereo 2ch wav, but got shape={data.shape} for file={input_wav}"
        )

    deg = data[:, 0]  # L = degraded
    ref = data[:, 1]  # R = ref

    output_deg_wav.parent.mkdir(parents=True, exist_ok=True)
    output_ref_wav.parent.mkdir(parents=True, exist_ok=True)

    if save_float32:
        sf.write(str(output_deg_wav), deg.astype("float32"), sr, subtype="FLOAT")
        sf.write(str(output_ref_wav), ref.astype("float32"), sr, subtype="FLOAT")
    else:
        sf.write(str(output_deg_wav), deg, sr)
        sf.write(str(output_ref_wav), ref, sr)

    info = sf.info(str(input_wav))
    return info.samplerate, info.frames, info.channels


def build_output_names(stem: str) -> Tuple[str, str]:
    deg_name = f"{stem}_deg.wav"
    ref_name = f"{stem}_ref.wav"
    return deg_name, ref_name


def build_records(
    wav_dir: Path,
    mapping: Dict[Tuple[str, int], float],
    output_deg_dir: Path,
    output_ref_dir: Path,
    glob_pattern: str,
    db_label: str,
    strict: bool,
    overwrite: bool,
    save_float32: bool,
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

        try:
            deg_name, ref_name = build_output_names(wav_path.stem)
            output_deg_wav = output_deg_dir / deg_name
            output_ref_wav = output_ref_dir / ref_name

            samplerate, num_frames, channels_original = split_stereo_wav(
                input_wav=wav_path,
                output_deg_wav=output_deg_wav,
                output_ref_wav=output_ref_wav,
                overwrite=overwrite,
                save_float32=save_float32,
            )

            records.append(
                {
                    "filepath_deg": str(output_deg_wav.as_posix()),
                    "filepath_ref": str(output_ref_wav.as_posix()),
                    "tmos": mapping[key],
                    "db": db_label,
                    "filename_2ch": wav_path.name,
                    "stem_2ch": wav_path.stem,
                    "basename": basename,
                    "index": index,
                    "samplerate": samplerate,
                    "num_frames": num_frames,
                    "channels_original": channels_original,
                }
            )

        except Exception:
            if strict:
                raise
            unmatched_files.append(wav_path.name)

    if unmatched_files:
        print("\n[WARNING] Files skipped or unmatched:")
        for name in unmatched_files:
            print(f"  - {name}")

    return records


def main() -> None:
    args = parse_args()

    wav_dir = Path(args.wav_dir)
    mapping_txt = Path(args.mapping_txt)
    output_deg_dir = Path(args.output_deg_dir)
    output_ref_dir = Path(args.output_ref_dir)
    output_csv = Path(args.output_csv)

    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    mapping = parse_mapping_txt(mapping_txt)

    records = build_records(
        wav_dir=wav_dir,
        mapping=mapping,
        output_deg_dir=output_deg_dir,
        output_ref_dir=output_ref_dir,
        glob_pattern=args.glob_pattern,
        db_label=args.db_label,
        strict=args.strict,
        overwrite=args.overwrite,
        save_float32=args.save_float32,
    )

    if not records:
        raise RuntimeError("No valid records were created.")

    df = pd.DataFrame(records)
    df = df.sort_values(["basename", "index"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n[INFO] Split completed and CSV created successfully.")
    print(f"[INFO] Output CSV: {output_csv}")
    print(f"[INFO] Degraded WAV dir: {output_deg_dir}")
    print(f"[INFO] Reference WAV dir: {output_ref_dir}")
    print(f"[INFO] Number of rows: {len(df)}")
    print("\n[INFO] Preview:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
