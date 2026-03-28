#!/usr/bin/env python3
"""Batch statistics calculator for evaluate_ret_<id>.json files.

Usage:
  # Manual id range
  python3 metric.py --task-dir /path/to/<task_name> --start-id 220 --end-id 229

  # Use CKPT_LOG_MAP key
  python3 metric.py --task-dir /path/to/<task_name> --ckpt 30k
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

FILE_RE = re.compile(r"^evaluate_ret_(\d+)\.json$")

CKPT_LOG_MAP = {
    '30k': [220,229],
    '20k': [230,239],
    '25k': [240,249],
    '36.9k': [282,291],
    'base': [292,301],
    '35k':[332,341],
    'many_17.9k': [312,321],
    '30k_2': [342,351],
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Read evaluate_ret_<id>.json files in a directory and compute the "
            "ratio of entries under statistics.scores whose value is all 1.0."
        ),
        epilog=(
            "Examples:\n"
            "  # Manual id range\n"
            "  python3 metric.py --task-dir /path/to/<task_name> --start-id 220 --end-id 229\n\n"
            "  # Use CKPT_LOG_MAP key\n"
            "  python3 metric.py --task-dir /path/to/<task_name> --ckpt 30k"
        ),
    )
    parser.add_argument("--task-dir", required=True, help="Directory containing evaluate_ret_*.json")
    parser.add_argument("--start-id", type=int, help="Start id (inclusive)")
    parser.add_argument("--end-id", type=int, help="End id (inclusive)")
    parser.add_argument(
        "--ckpt",
        choices=sorted(CKPT_LOG_MAP.keys()),
        help="Use id range from CKPT_LOG_MAP key, e.g. 30k",
    )
    return parser.parse_args()


def resolve_id_range(args: argparse.Namespace) -> tuple[int, int]:
    if args.ckpt is not None:
        if args.start_id is not None or args.end_id is not None:
            raise ValueError("Use either --ckpt or --start-id/--end-id, not both.")
        start_id, end_id = CKPT_LOG_MAP[args.ckpt]
        return int(start_id), int(end_id)

    if args.start_id is None or args.end_id is None:
        raise ValueError("Provide --ckpt, or provide both --start-id and --end-id.")

    return int(args.start_id), int(args.end_id)


def is_all_ones(value: Any) -> bool:
    """Return True when value represents all-1.0 data.

    Rules:
    - number: must equal 1.0
    - list/tuple: non-empty and every element equals 1.0
    - other types: False
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-9)

    if isinstance(value, (list, tuple)):
        if not value:
            return False
        for item in value:
            if not (isinstance(item, (int, float)) and not isinstance(item, bool)):
                return False
            if not math.isclose(float(item), 1.0, rel_tol=0.0, abs_tol=1e-9):
                return False
        return True

    return False


def find_target_files(task_dir: Path, start_id: int, end_id: int) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in sorted(task_dir.glob("evaluate_ret_*.json")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        file_id = int(match.group(1))
        if start_id <= file_id <= end_id:
            files.append((file_id, path))

    files.sort(key=lambda x: (x[0], x[1].name))
    return files


def get_scores_dict(payload: dict[str, Any]) -> dict[str, Any]:
    statistics = payload.get("statistics", {})
    if not isinstance(statistics, dict):
        return {}

    scores = statistics.get("scores")
    if not isinstance(scores, dict):
        return {}
    return scores


def calc_file_counts(scores: dict[str, Any]) -> tuple[int, int]:
    total = len(scores)
    success = sum(1 for value in scores.values() if is_all_ones(value))
    return success, total


def main() -> None:
    args = parse_args()
    start_id, end_id = resolve_id_range(args)

    if start_id > end_id:
        raise ValueError("start_id must be <= end_id")

    task_dir = Path(args.task_dir).expanduser().resolve()
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {task_dir}")

    files = find_target_files(task_dir, start_id, end_id)
    if not files:
        print(f"No evaluate_ret_*.json found in id range [{start_id}, {end_id}] under: {task_dir}")
        return

    total_success = 0
    total_items = 0

    if args.ckpt is not None:
        print(f"Using CKPT_LOG_MAP[{args.ckpt!r}] -> id range [{start_id}, {end_id}]")
    print(f"Found {len(files)} file(s) in id range [{start_id}, {end_id}]")
    for file_id, path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] skip {path.name}: failed to parse json ({exc})")
            continue

        scores = get_scores_dict(payload)
        success, total = calc_file_counts(scores)

        total_success += success
        total_items += total

        ratio_str = "N/A" if total == 0 else f"{(success / total):.4%}"
        print(f"- {path.name} (id={file_id}): {success}/{total} = {ratio_str}")

    if total_items == 0:
        print("\nOverall ratio: N/A (no valid entries under statistics.scores)")
        return

    overall_ratio = total_success / total_items
    print(f"\nOverall ratio across selected files: {total_success}/{total_items} = {overall_ratio:.4%}")


if __name__ == "__main__":
    main()
