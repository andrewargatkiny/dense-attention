#!/usr/bin/env python3

# This script checks maximum number of epochs given `--config` and `--world-size`
# till the language modeling data stored on disk gets exhausted.

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training config JSON.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of nodes or processes sharing the dataset.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message: str) -> None:
    raise SystemExit(message)


def ensure_path_exists(path: Path) -> Path:
    if not path.exists():
        fail(f"Configured dataset path does not exist: {path}")
    return path


def count_rows(dataset_path: Path) -> int:
    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if not parquet_files:
        fail(f"No parquet files found under: {dataset_path}")

    total_rows = 0
    for file_path in parquet_files:
        total_rows += pq.ParquetFile(file_path).metadata.num_rows
    return total_rows


def exhaustion_epochs(rank_rows: float, chunk_size: int) -> dict:
    full_epochs = int(rank_rows // chunk_size)
    remainder = rank_rows - (full_epochs * chunk_size)
    first_non_full = full_epochs + 1
    first_empty = first_non_full if remainder == 0 else first_non_full + 1
    return {
        "rank_rows": rank_rows,
        "full_epochs": full_epochs,
        "remainder": remainder,
        "first_non_full": first_non_full,
        "first_empty": first_empty,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if args.world_size <= 0:
        fail(f"world_size must be positive, got {args.world_size}")

    config = load_json(config_path)

    training_data = config["data"]["training"]
    sources = training_data.get("hf_sources", [])
    if len(sources) != 1:
        fail(
            "This one-off script expects exactly one training hf_source, "
            f"found {len(sources)}."
        )

    source = dict(sources[0])
    dataset_path = ensure_path_exists(Path(source["name"]))

    chunk_size = source["chunk_size"]
    total_rows = count_rows(dataset_path)
    rank_rows = total_rows / args.world_size
    summary = exhaustion_epochs(rank_rows, chunk_size)

    print(f"Config: {config_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"World size: {args.world_size}")
    print(f"Chunk size per rank per epoch: {chunk_size:,}")
    print()
    print(f"Total rows: {total_rows:,}")
    print(f"Approx rows per rank: {math.floor(rank_rows):,}")
    print(f"Full chunk epochs: {summary['full_epochs']:,}")
    print(f"First non-full epoch: {summary['first_non_full']:,}")
    print(f"First empty epoch: {summary['first_empty']:,}")
    print()
    print(
        "This is a simple metadata-based estimate: total parquet rows "
        f"divided by {args.world_size}."
    )


if __name__ == "__main__":
    main()

