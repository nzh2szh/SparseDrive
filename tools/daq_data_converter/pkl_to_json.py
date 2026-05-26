#!/usr/bin/env python3
"""Convert a DAQ `daq_data_infos_infe.pkl` file to a JSON file.

Usage:
  python tools/data_converter/pkl_to_json.py --input data/infos/daq_data_infos_infe.pkl --output out.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert daq_data_infos_infe.pkl to JSON")
    parser.add_argument("--input", "-i", required=True, help="输入 pkl 路径")
    parser.add_argument("--output", "-o", required=True, help="输出 json 路径")
    return parser.parse_args()


def to_primitive(obj: Any) -> Any:
    """Recursively convert numpy / pathlib / bytes types to JSON-serializable primitives."""
    from pathlib import Path

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("latin1", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    # numpy types
    if isinstance(obj, np.ndarray):
        return to_primitive(obj.tolist())
    if isinstance(obj, (np.integer, np.floating, np.bool_, np.number, np.bool_)):
        try:
            return obj.item()
        except Exception:
            return int(obj)
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_primitive(v) for v in obj]

    # fallback: try to stringify
    try:
        return str(obj)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    # If user provided an output directory (existing dir or path ending with a separator),
    # write the JSON file inside that directory using the input stem.
    if (output_path.exists() and output_path.is_dir()) or str(args.output).endswith(os.sep):
        output_path = output_path / (input_path.stem + ".json")

    if not input_path.is_file():
        print(f"输入文件不存在: {input_path}")
        return 2

    with input_path.open("rb") as fh:
        payload = pickle.load(fh)

    primitive = to_primitive(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(primitive, fh, ensure_ascii=False, indent=2)

    print(f"已写入 JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
