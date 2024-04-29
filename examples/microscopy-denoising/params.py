# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import argparse

input_parser = argparse.ArgumentParser(add_help=False)
input_parser.add_argument(
    "--rescale",
    type=float,
    help="If specified, the size of the clean image will be rescaled by this factor "
    "(only for demonstration purposes to minimize computational effort)",
    default=0.1,
)

sarscov2_parser = argparse.ArgumentParser(add_help=False)
sarscov2_parser.add_argument(
    "--id",
    choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    type=str,
    help="Specify image ID (compare Tab. S1 in the Supplement of the manuscript)",
    default="8",
)

sarscov2_parser.add_argument(
    "--sbregions",
    type=str,
    help="Specify the path to the coordinates of the signal and background regions",
    default='',
)

cilia_parser = argparse.ArgumentParser(add_help=False)
cilia_parser.add_argument(
    "noisy",
    type=str,
    help="Path to noisy Cilia image",
)

cilia_parser.add_argument(
    "gt",
    type=str,
    help="Path to Pseudo Ground Truth Cilia image",
)

fm_parser = argparse.ArgumentParser(add_help=False)
fm_parser.add_argument(
    "--id",
    choices=["convallaria", "mouse_skull_nuclei", "mouse_actin"],
    type=str,
    help="Specify image ID",
    default="convallaria",
)

patch_parser = argparse.ArgumentParser(add_help=False)
patch_parser.add_argument(
    "--patch_height",
    type=int,
    help="Patch height",
    default=6,
)

patch_parser.add_argument(
    "--patch_width",
    type=int,
    help="Patch width (defaults to patch_height if not specified)",
    default=None,
)

experiment_parser = argparse.ArgumentParser(add_help=False)
experiment_parser.add_argument(
    "-C",
    type=int,
    help="Number of cluster centers to learn",
    default=1000,
)

experiment_parser.add_argument(
    "--no_epochs",
    type=int,
    help="Number of epochs to train",
    default=50,
)

experiment_parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch_size",
    default=None,
)

output_parser = argparse.ArgumentParser(add_help=False)
output_parser.add_argument(
    "--output_directory",
    type=str,
    help="Directory to write H5 training output and visualizations to (will be output/<TIMESTAMP> "
    "if not specified)",
    default=None,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Microscopy Denoising with PMM"
    )
    data_parsers = parser.add_subparsers(
        help="Select data to train", dest="data", required=True
    )
    comm_parents = [
        input_parser,
        patch_parser,
        experiment_parser,
        output_parser,
    ]

    data_parsers.add_parser(
        "sarscov2",
        help="Run experiment with SARS-CoV-2 data",
        parents=comm_parents
        + [
            sarscov2_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_parsers.add_parser(
        "cilia",
        help="Run experiment with Cilia data",
        parents=comm_parents
        + [
            cilia_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_parsers.add_parser(
        "fm",
        help="Run experiment with Fluorescence Microscopy data",
        parents=comm_parents
        + [
            fm_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args()
