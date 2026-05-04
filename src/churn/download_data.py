import argparse
import shutil
from pathlib import Path

import kagglehub

from churn.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "blastchar/telco-customer-churn"
DEFAULT_OUTPUT = Path("data/raw/churn.csv")
SOURCE_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def download_churn_dataset(
    dataset: str = DEFAULT_DATASET,
    output: str | Path = DEFAULT_OUTPUT,
) -> Path:
    # KaggleHub baixa para cache local; depois copiamos para data/raw/churn.csv.
    dataset_path = Path(kagglehub.dataset_download(dataset))
    source_file = dataset_path / SOURCE_FILENAME
    if not source_file.exists():
        raise FileNotFoundError(f"Arquivo original nao encontrado: {source_file}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_file, output_path)
    logger.info(
        "dataset_downloaded",
        dataset=dataset,
        source_file=str(source_file),
        output=str(output_path),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    download_churn_dataset(dataset=args.dataset, output=args.output)
