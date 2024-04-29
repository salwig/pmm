# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import os
import h5py
import urllib
import zipfile
import numpy as np
import torch as to
import tifffile
import progressbar
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import blur_effect
from typing import Dict

from compute_snr import compute_snr

links = {
    "Dataset_02": "https://zenodo.org/record/3985103/files/Dataset_02_SARS-CoV-2.zip?download=1",
    "Dataset_03": "https://zenodo.org/record/3985110/files/Dataset_03_SARS-CoV-2.zip?download=1",
    "Dataset_07": "https://zenodo.org/record/3986580/files/Dataset_07_SARS-CoV-2.zip?download=1",
    "convallaria": "https://zenodo.org/record/5156913/files/Convallaria_diaphragm.zip?download=1",
    "mouse_skull_nuclei": "https://zenodo.org/record/5156960/files/Mouse%20skull%20nuclei.zip?download=1",
    "mouse_actin": "https://zenodo.org/record/5156937/files/Mouse%20actin.zip?download=1",
}

sarscov2_table = {
    "1": "Dataset_02_SARS-CoV-2_007.tif",
    "2": "Dataset_02_SARS-CoV-2_009.tif",
    "3": "Dataset_02_SARS-CoV-2_038.tif",
    "4": "Dataset_03_SARS-CoV-2_043.tif",
    "5": "Dataset_03_SARS-CoV-2_070.tif",
    "6": "Dataset_03_SARS-CoV-2_080.tif",
    "7": "Dataset_07_SARS-CoV-2_036.tif",
    "8": "Dataset_07_SARS-CoV-2_077.tif",
    "9": "Dataset_07_SARS-CoV-2_102.tif",
}

fm_table = {
    "convallaria": [
        "Convallaria_diaphragm/20190520_tl_25um_50msec_05pc_488_130EM_Conv.tif",
        74,
    ],
    "mouse_skull_nuclei": ["Mouse skull nuclei/example2_digital_offset300.tif", 176],
    "mouse_actin": ["Mouse actin/sample_attempt2.tif", 40],
}


class MyProgressBar:
    """Progressbar for visualization of the download progress

    Source: https://stackoverflow.com/a/53643011
    """

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def get_data(args) -> to.Tensor:
    """Read image from file, optionally rescale image size and return as to.Tensor

    :param args: Argument parser contatining name of the data set (args.data) and rescale factor (args.rescale)
    :return: Image as torch tensor
    """
    if args.data == "sarscov2":
        if not args.sbregions and args.rescale==1.0:
            print("Warning: File for Coordinates of Signal and Background Regions not specified!")
        img, clean = get_sarscov2(args.id)
    elif args.data == "cilia":
        img, clean = get_cilia(args.noisy,args.gt)
    elif args.data == "fm":
        img, clean = get_fm(args.id)

    if args.rescale != 1.0:
        orig_shape = img.shape
        target_shape = [
            int(orig_shape[1] * args.rescale),
            int(orig_shape[0] * args.rescale),
        ]
        img = np.asarray(
            Image.fromarray(img).resize(target_shape, resample=Image.NEAREST),
            dtype=np.float64,
        )
        if clean is not None:
            clean = np.asarray(
                Image.fromarray(clean).resize(target_shape, resample=Image.NEAREST),
                dtype=np.float64,
            )
        print(
            "Resized input image from {}->{}".format(orig_shape, np.asarray(img).shape)
        )
        return to.from_numpy(img), to.from_numpy(clean) if clean is not None else None
    else:
        return (
            to.from_numpy(np.asarray(img, dtype=np.float64)),
            to.from_numpy(np.asarray(clean, dtype=np.float64))
            if clean is not None
            else None,
        )


def get_sarscov2(id):
    '''Open (and if necessary download) a SARS-CoV-2 image 

    :param id: Id of the image 
    '''
    file = sarscov2_table[id]
    dataset = file[:10]

    zip_file = f"./data/sars-cov-2/{dataset}.zip"
    folder = f"./data/sars-cov-2/{dataset}/"

    if not os.path.isdir(folder) and not os.path.exists(zip_file):
        os.makedirs("./data/sars-cov-2/", exist_ok=True)
        print("Downloading data... (may take a while)", flush=True)
        urllib.request.urlretrieve(links[dataset], zip_file, MyProgressBar())

    if not os.path.isdir(folder):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(folder)

    img = tifffile.imread(folder + file)
    return img, None


def get_cilia(file_noisy,file_gt):
    """Open the cilia image (No. 91) and the pseudo-ground-truth image
    """
    noisy = tifffile.imread(file_noisy)
    gt = tifffile.imread(file_gt)
    return noisy, gt


def get_fm(id):
    """Open (and if necessary download) the fluorescence microscopy images and the pseudo-ground-truth image
    
    :param id: Id of the image 
    """
    zip_file = f"./data/fm/{id}.zip"
    folder = f"./data/fm/{id}/"

    if not os.path.isdir(folder) and not os.path.exists(zip_file):
        os.makedirs("./data/fm/", exist_ok=True)
        print("Downloading data... (may take a while)", flush=True)
        urllib.request.urlretrieve(links[id], zip_file, MyProgressBar())

    if not os.path.isdir(folder):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(folder)

    img = tifffile.imread(folder + fm_table[id][0])
    return img[fm_table[id][1]], np.mean(img, axis=0)


def store_as_h5(to_store_dict: Dict[str, to.Tensor], output_name: str) -> None:
    """Takes dictionary of tensors and writes to H5 file

    :param to_store_dict: Dictionary of torch Tensors
    :param output_name: Full path of H5 file to write data to
    """
    os.makedirs(os.path.split(output_name)[0], exist_ok=True)
    with h5py.File(output_name, "w") as f:
        for key, val in to_store_dict.items():
            f.create_dataset(
                key, data=val if isinstance(val, float) else val.detach().cpu()
            )
    print(f"Wrote {output_name}")


def eval_sarscov2(reco, file, id, rescale):
    """Compute SNR for SARS-CoV-2 images (only implemented for the full images without rescale)
    
    :param reco: Reconstructed image
    :param file: Path to file of the coordinates of the signal and background regions
    :param id: ID of the SARS-CoV-2 image (compare Tab.S1)
    :param rescale: Rescale factor of the image
    """
    if not os.path.isfile(file):
        return "File for Coordinates of Signal and Background Regions not found!"
    if rescale == 1.0:
        return compute_snr(reco.clone().detach().cpu().numpy(), file, id=id)
    else:
        print("Calculations of SNR implemented only for the whole image", flush=True)
        return "Calculations of SNR implemented only for the whole image"


def eval_fn(target, reco, args) -> to.Tensor:
    """Compute PSNR/ SNR and blur effect measures for the data

    :param target: Target image (unused for SARS-CoV-2)
    :param reco: Reconstructed image
    :param args: Argument parser
    """
    blureffect_value = blur_effect(reco) 
    if args.data == "sarscov2":
        return eval_sarscov2(reco, args.sbregions, args.id, args.rescale), blureffect_value
    else:
        return peak_signal_noise_ratio(
            target.clone().detach().cpu().numpy()
            if isinstance(target, to.Tensor)
            else target,
            reco.clone().detach().cpu().numpy()
            if isinstance(reco, to.Tensor)
            else reco,
            data_range=target.max() - target.min(),
        ).item(), blureffect_value
