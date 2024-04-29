# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# Compute Eq. 4 of paper "Topaz-Denoise: general deep denoising models for cryoEM and cryoET"

import numpy as np
from typing import List, Tuple
from typing import Tuple

def read_labels(file: str,id: int) -> List:
    """Open coordinates of the signal and background regions.

    :param file: Path to file containing signal and background regions coordinates
    :param id: ID of the SARS-CoV-2 image (compare Tab.S1)
    """
    with open(file, "r") as f:
        labels = f.readlines()
    idx = labels.index(f"Image{id}\n") + 1
    labels = labels[idx:idx+20]
    return labels

def _get_signal_and_background_regions(
    img: np.ndarray,
    labels: List,
    verbose: bool = False,
) -> Tuple[List, List]:
    """ Extract the signal and background regions from the image

    :param img: Image array
    :param labels: List of the coordinates of the signal and background regions
    :param verbose: Wheter to print read message
    """
    signal_regions, background_regions = [], []
    for l in labels:
        
        inds = l.split("\t")
        inds[-1] = inds[-1].replace("\n", "")
        inds = [int(x) for x in inds]

        signal_region = img[inds[0]:inds[1], inds[2]:inds[3]]
        background_region = img[inds[4]:inds[5], inds[6]:inds[7]]

        signal_regions.append(signal_region)
        background_regions.append(background_region)
    if verbose:
        print("Read pairs of signal/background regions")
    return signal_regions, background_regions


def _compute_variances_of_regions(
    signal_regions: List, background_regions: List
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the variances of the signal and background regions.

    :param signal_regions: List of signal region arrays
    :param background_regions: List of background region arrays
    """
    signal_variances, background_variances = [], []
    for signal, background in zip(signal_regions, background_regions):
        background_variances += [np.var(background)]
        signal_variances += [(np.mean(signal) - np.mean(background)) ** 2]
    return np.array(signal_variances), np.array(background_variances)


def _compute_snr(signal: np.ndarray, background: np.ndarray) -> float:
    """Compute SNR for a signal and background region

    :param signal: Signal region array
    :param background: Background region array
    """
    return (np.log10(signal) - np.log10(background)).sum() / len(signal) * 10.0


def compute_snr(image: np.ndarray, file: str, id: int):
    """Compute SNR like Eq. 4 of paper "Topaz-Denoise: general deep denoising models for cryoEM and cryoET"

    :param image: Image array
    :param file: Path to file containing signal and background regions coordinates
    :param id: ID of the SARS-CoV-2 image (compare Tab.S1)
    """
    labels = read_labels(file,id)
    signal_regions, background_regions = _get_signal_and_background_regions(image,labels)
    var_s, var_b = _compute_variances_of_regions(signal_regions, background_regions)
    return _compute_snr(var_s, var_b)
