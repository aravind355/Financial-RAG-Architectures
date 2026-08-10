"""
scripts/download_datasets.py
============================
Download the FinQA test benchmark for HierFinRAG evaluation.

Downloads the official FinQA test split from GitHub and saves a
configurable subset to data/datasets/finqa_subset.json.

Usage::

    python scripts/download_datasets.py
    python scripts/download_datasets.py --n 100
"""

import os
import json
import argparse
import requests

# Official FinQA test split — hosted on the czyssrs/FinQA GitHub repo
FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json"

OUTPUT_DIR  = "data/datasets"
OUTPUT_PATH = "data/datasets/finqa_subset.json"


def main(n: int = 30) -> None:
    """
    Download and save the first n FinQA test items.

    Args:
        n : Number of items to save (default: 30).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/extracted", exist_ok=True)

    print(f"Downloading FinQA test split from:\n  {FINQA_URL}")
    response  = requests.get(FINQA_URL, timeout=30)
    response.raise_for_status()

    all_data = response.json()
    subset   = all_data[:n]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)

    print(f"Saved {len(subset)} FinQA samples to: {OUTPUT_PATH}")
    print(f"Run evaluation with:  python evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FinQA test benchmark for HierFinRAG evaluation"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of items to download (default: 30)",
    )
    args = parser.parse_args()
    main(n=args.n)
