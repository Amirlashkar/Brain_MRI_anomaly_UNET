"""
python submission.py --data-dir /path/to/data-dir --predictions-file-path /path/to/submission.csv
"""

import click
from pathlib import Path
from predictor import Predictor


HERE = Path(__file__).absolute().resolve().parent


@click.command()
@click.option(
    "--data-dir",
    type=Path,
    help="path to data directory, which consists of folders of Dicom files, each one corresponding to a Dicom series.",
)
@click.option("--predictions-file-path", type=Path)
def main(data_dir: Path, predictions_file_path: Path):
    p = Predictor(data_dir)
    predictions_df = p.predict()
    predictions_df.to_csv(predictions_file_path, index=False)


if __name__ == "__main__":
    main()
