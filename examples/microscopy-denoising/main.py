# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import os
import sys
import time
import datetime
import torch as to
from params import get_args

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sys.path.append("../../")
from models.pmm import PMM
from models.utils.param_init import *
from models.utils.viz import viz_theta
import matplotlib.pyplot as plt

from tvutil.tvutil.prepost import (
    OverlappingPatches,
    median_merger,
)
from utils import (
    get_data,
    store_as_h5,
    eval_fn,
)

# DEVICE = to.device("cuda:0" if to.cuda.is_available() else "cpu")
DEVICE = to.device("cpu")
PRECISION = to.float64


def get_timestamp_or_slurmid():
    return (
        os.environ["SLURM_JOBID"]
        if "SLURM_JOBID" in os.environ
        else datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d_%H-%M-%S")
    )


def exp(out, args):
    cmap = "magma" if args.data == "fm" else "gray"

    # load noisy image
    noisy, clean = get_data(args)
    plt.imsave(out + "noisy.png", noisy, cmap=cmap)
    if clean is not None:
        plt.imsave(out + "pseudo-gt.png", clean, cmap=cmap)

    # cut patches of the image
    patch_width = args.patch_height if args.patch_width is None else args.patch_width
    ovp = OverlappingPatches(noisy, args.patch_height, patch_width, patch_shift=1)
    data = ovp.get().t().clone()
    store_as_h5({"data": data}, out + "training_patches.h5")
    D = args.patch_height * patch_width

    # Initialize Model
    C = args.C
    W_init = init_W_data_mean(data=data, C=C, type="kmeans++").contiguous()
    model = PMM(
        C=args.C,
        D=D,
        W_init=W_init,
        pies_init=to.full((C,), 1.0 / C),
        precision=PRECISION,
        device=DEVICE,
    )

    # Train Model
    print("Train Model...")
    if args.batch_size:
        model.batch_fit(data, args.no_epochs, args.batch_size, out=out)
    else:
        model.fit(data, args.no_epochs, out=out)

    # Reconstruct Patches
    if args.batch_size:
        reconstructions = model.estimate_batch(data, args.batch_size)
    else:
        reconstructions = model.estimate(data)
    store_as_h5({"data": reconstructions}, out + "reconstructed_patches.h5")

    # Reconstruct image with mergers
    reco = ovp.set_and_merge(reconstructions.t(), merge_method=median_merger)

    # Save reconstructed images
    plt.imsave(out + "reconstruction.png", reco, cmap=cmap)
    store_as_h5({"data": reco}, out + "reconstruction.h5")

    p_snr, blureffect = eval_fn(clean, reco, args)

    out += "evaluations.txt"
    with open(out, "w") as f:
        f.writelines("P/SNR: " + str(p_snr) + "\n")
        f.writelines("Blur Effect: " + str(blureffect) + "\n")


if __name__ == "__main__":
    args = get_args()

    # define output directory
    out = (
        "./out/" + get_timestamp_or_slurmid() + "/"
        if args.output_directory is None
        else args.output_directory
    )
    os.makedirs(out, exist_ok=True)

    start = time.time()
    exp(out, args)
    runtime = time.time() - start

    with open(out + "/runtimes.txt", "a") as file:
        file.write("Runtime of PMM is " + str(round(runtime, 2)) + "s.")

    viz_theta(out)
