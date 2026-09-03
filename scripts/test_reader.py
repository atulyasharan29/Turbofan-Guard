"""Test script for raw data reader and column grouping."""

import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_suite_split, load_manifests, get_column_groups, get_split_summary


def test_suite(suite_name: str):
    print(f"\n{'='*60}")
    print(f"TESTING DATA INGESTION: {suite_name}")
    print(f"{'='*60}")

    for split in ["train", "val", "test"]:
        df = load_suite_split(suite_name, split=split)
        summary = get_split_summary(df)
        print(f"[{split.upper()} SPLIT]")
        print(f"  Rows loaded         : {summary['total_rows']}")
        print(f"  Total columns       : {summary['total_columns']}")
        print(f"  Unique engines      : {summary['num_engines']}")
        print(f"  Cycle range         : {summary['cycle_range']}")
        print(f"  Conditions          : {summary['num_conditions']}")
        print(f"  Observed sensors    : {summary['num_observed_sensors']}")
        print(f"  Truth sensors       : {summary['num_truth_sensors']}")
        print(f"  Health states (leak): {summary['num_health_indices_isolated']}")

    manifest_df, meta_json = load_manifests(suite_name)
    if manifest_df is not None:
        print(f"\nManifest loaded: {len(manifest_df)} engines.")
    if meta_json is not None:
        print(f"Suite description: {meta_json.get('description', '')[:70]}...")

    print(f"\n[PASS] {suite_name} read successfully.")


def main():
    print("TurbofanGuard: Testing Raw Data Ingestion Reader...")
    test_suite("DS01")
    test_suite("DS03")
    print(f"\n{'='*60}")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
