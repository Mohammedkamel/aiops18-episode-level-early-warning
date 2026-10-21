"""Stream-read specific CSVs out of the CCF AIOps 2022 archives without full extraction.

Disk space is constrained (~11-12GB free); training_data_with_faults.zip (4.98GB) and
test_data.7z (2.17GB) would need 30-50GB+ if fully extracted, so every reader here pulls
a single named member into memory (or a subprocess pipe) and returns a DataFrame,
touching disk only for the archive itself.
"""

import io
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "aiops2022"
FAULTS_ZIP = RAW_DIR / "training_data_with_faults.zip"
TEST_7Z = RAW_DIR / "test_data.7z"
CLOUDBED1_EXTRACTED = RAW_DIR / "training_data_normal_2022-03-19-cloudbed1_extracted" / "cloudbed-1"

FAULTS_ZIP_PREFIX = "training_data_with_faults/training_data_with_faults/tar"


def read_metric_service_from_faults_zip(cloudbed: int, date_mmdd: str) -> pd.DataFrame:
    """Stream metric/service/metric_service.csv for one cloudbed/day out of training_data_with_faults.zip.

    date_mmdd like '03-20'. cloudbed in {1,2,3}.
    """
    member = f"{FAULTS_ZIP_PREFIX}/2022-{date_mmdd}-cloudbed{cloudbed}/metric/service/metric_service.csv"
    with zipfile.ZipFile(FAULTS_ZIP) as z:
        with z.open(member) as f:
            return pd.read_csv(f)


def read_metric_node_from_faults_zip(cloudbed: int, date_mmdd: str) -> pd.DataFrame:
    """Stream metric/node/kpi_cloudbed{N}_metric_{MMDD}.csv out of training_data_with_faults.zip."""
    member = (
        f"{FAULTS_ZIP_PREFIX}/2022-{date_mmdd}-cloudbed{cloudbed}/metric/node/"
        f"kpi_cloudbed{cloudbed}_metric_{date_mmdd.replace('-', '')}.csv"
    )
    with zipfile.ZipFile(FAULTS_ZIP) as z:
        with z.open(member) as f:
            return pd.read_csv(f)


# The node metric filename inside each test date folder embeds a stale/misleading
# cloudbed+date suffix (confirmed earlier: internal timestamps don't match the suffix,
# e.g. '2022-05-01/.../kpi_cloudbed1_metric_0324.csv' actually contains 04-30/05-01 data).
# metric_service.csv has no such suffix, so only the node reader needs this lookup.
TEST_NODE_FILENAME_BY_DATE = {
    "05-01": "kpi_cloudbed1_metric_0324.csv",
    "05-03": "kpi_cloudbed2_metric_0324.csv",
    "05-05": "kpi_cloudbed2_metric_0325.csv",
    "05-07": "kpi_cloudbed3_metric_0325.csv",
    "05-09": "kpi_cloudbed1_metric_0315.csv",
}


def read_metric_node_from_test_7z(date_mmdd: str) -> pd.DataFrame:
    """Stream metric/node for one test date out of test_data.7z via `7z e -so`."""
    filename = TEST_NODE_FILENAME_BY_DATE[date_mmdd]
    member = f"初赛评分数据/2022-{date_mmdd}/cloudbed/metric/node/{filename}"
    proc = subprocess.run(
        ["7z", "e", "-so", str(TEST_7Z), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return pd.read_csv(io.BytesIO(proc.stdout))


def read_metric_service_from_test_7z(date_mmdd: str) -> pd.DataFrame:
    """Stream metric/service/metric_service.csv for one test date out of test_data.7z via `7z e -so`.

    date_mmdd like '05-01'.
    """
    member = f"初赛评分数据/2022-{date_mmdd}/cloudbed/metric/service/metric_service.csv"
    proc = subprocess.run(
        ["7z", "e", "-so", str(TEST_7Z), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return pd.read_csv(io.BytesIO(proc.stdout))


def read_metric_service_from_cloudbed1_extracted() -> pd.DataFrame:
    """The one exception: cloudbed1 2022-03-19 is already fully extracted on disk."""
    return pd.read_csv(CLOUDBED1_EXTRACTED / "metric" / "service" / "metric_service.csv")


if __name__ == "__main__":
    df_zip = read_metric_service_from_faults_zip(1, "03-20")
    print("faults_zip 2022-03-20-cloudbed1 metric_service.csv:", df_zip.shape)
    print(df_zip.head(2))

    df_7z = read_metric_service_from_test_7z("05-01")
    print("\ntest_7z 2022-05-01 metric_service.csv:", df_7z.shape)
    print(df_7z.head(2))

    df_extracted = read_metric_service_from_cloudbed1_extracted()
    print("\ncloudbed1_extracted 2022-03-19 metric_service.csv:", df_extracted.shape)
    print(df_extracted.head(2))
