from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


def main():
    parser = argparse.ArgumentParser(description="Aggregate raw benchmark JSON files into a single CSV-friendly table.")
    parser.add_argument("--results-dir", default=None, help="Defaults to results/raw")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results" / "raw"
    rows: List[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    output_path = repo_root / "results" / "aggregated" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = repo_root / "results" / "aggregated" / "benchmark_results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
