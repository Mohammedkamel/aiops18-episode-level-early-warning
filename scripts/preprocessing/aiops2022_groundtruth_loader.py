"""Load the CCF AIOps 2022 fault-injection groundtruth into one unified events table.

Training labels: data/raw/aiops2022/groundtruth/train/groundtruth-k8s-{cloudbed}-2022-{date}.csv
Test labels:     data/raw/aiops2022/groundtruth/test/groundtruth-2022-{date}.json (parallel-array format)
"""

import json
import re
from pathlib import Path

import pandas as pd

GT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "aiops2022" / "groundtruth"

TRAIN_NAME_RE = re.compile(r"groundtruth-k8s-(\d)-2022-(\d{2}-\d{2})\.csv")
TEST_NAME_RE = re.compile(r"groundtruth-2022-(\d{2}-\d{2})\.json")


def load_train_events() -> pd.DataFrame:
    """One row per fault-injection event from the 7 training groundtruth CSVs."""
    rows = []
    for path in sorted((GT_DIR / "train").glob("groundtruth-k8s-*.csv")):
        m = TRAIN_NAME_RE.match(path.name)
        cloudbed, date_mmdd = int(m.group(1)), m.group(2)
        df = pd.read_csv(path)
        df["cloudbed"] = cloudbed
        df["date_mmdd"] = date_mmdd
        df["source_file"] = path.name
        rows.append(df)
    events = pd.concat(rows, ignore_index=True)
    events["event_id"] = [f"train-{i:04d}" for i in range(len(events))]
    return events


def load_test_events() -> pd.DataFrame:
    """One row per fault-injection event from the 5 test groundtruth JSONs.

    Test labels have no cloudbed field (single merged 'cloudbed' folder per date) —
    cloudbed is left null here; see prior finding that the metric_service.csv service
    topology matched cloudbed1 2022-03-20 exactly (11/11 services), so cloudbed identity
    is not resolvable from the label file alone.
    """
    rows = []
    for path in sorted((GT_DIR / "test").glob("groundtruth-2022-*.json")):
        m = TEST_NAME_RE.match(path.name)
        date_mmdd = m.group(1)
        d = json.loads(path.read_text())
        df = pd.DataFrame(
            {
                "timestamp": d["timestamp"],
                "level": d["level"],
                "cmdb_id": d["cmdb_id"],
                "failure_type": d["failure_type"],
            }
        )
        df["date_mmdd"] = date_mmdd
        df["source_file"] = path.name
        rows.append(df)
    events = pd.concat(rows, ignore_index=True)
    events["event_id"] = [f"test-{i:04d}" for i in range(len(events))]
    return events


if __name__ == "__main__":
    train = load_train_events()
    test = load_test_events()
    print("train events:", len(train), "expected 300")
    print(train["cloudbed"].value_counts().sort_index())
    print(train["date_mmdd"].value_counts().sort_index())
    print()
    print("test events:", len(test), "expected 241")
    print(test["date_mmdd"].value_counts().sort_index())
