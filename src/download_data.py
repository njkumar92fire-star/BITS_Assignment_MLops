"""Download the dataset and produce the cleaned CSV.

Usage:
    python -m src.download_data                  # download from UCI, then clean
    python -m src.download_data --from-zip heart_disease.zip   # use a local zip
    python -m src.download_data --skip-download  # only re-clean existing raw file
"""
import argparse
import logging
import zipfile

from src import config
from src.data import download_raw, prepare_clean_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-zip", help="path to a local heart_disease.zip")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.from_zip:
        with zipfile.ZipFile(args.from_zip) as zf:
            config.RAW_PATH.write_bytes(zf.read("processed.cleveland.data"))
        logger.info("Extracted raw file from %s", args.from_zip)
    elif not args.skip_download:
        try:
            download_raw()
        except Exception as exc:  # network issues: fall back to the committed copy
            if not config.RAW_PATH.exists():
                raise
            logger.warning("Download failed (%s); using existing %s", exc, config.RAW_PATH)

    df = prepare_clean_dataset()
    logger.info("Clean dataset shape: %s | class balance: %s",
                df.shape, df[config.TARGET].value_counts().to_dict())


if __name__ == "__main__":
    main()
