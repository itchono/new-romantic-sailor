import argparse
from pathlib import Path

from qlipper.constants import OUTPUT_DIR
from qlipper.postprocess import postprocess_from_folder


def main():
    parser = argparse.ArgumentParser(description="Postprocess a qlipper run.")
    parser.add_argument(
        "--folder", type=str, help="Folder containing the run.", required=False
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to show the plots, default False.",
        required=False,
    )

    args = parser.parse_args()

    if args.folder is None:
        from tkinter.filedialog import askdirectory

        args.folder = askdirectory(initialdir=OUTPUT_DIR, title="Select case folder")

    postprocess_from_folder(Path(args.folder), args.show)


if __name__ == "__main__":
    main()
