# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import numpy as np
import torch as to
from torch import Tensor
from sklearn.cluster import kmeans_plusplus


def init_W_data_mean(
    data: Tensor,
    C: int,
    factor: float = 1,
    type: str = "poisson",
    dtype: to.dtype = to.float64,
    device: to.device = None,
) -> Tensor:
    """Initialize weights W.

    :param data: Data set with shape (N, D).
    :param C: Number of clusters.
    :param factor: Scalar to control amount of additive Poisson noise
    :param type: Inititilizations method. Choose from ("poisson","kmeans++")
    :param dtype: dtype of output Tensor. Defaults to torch.float64.
    :param device: torch.device of output Tensor. Defaults to tvo.get_device().
    :returns: Weight matrix W with shape (D,H).
    """
    data_nanmean = to.from_numpy(np.nanmean(data.detach().cpu().numpy(), axis=0)).to(
        dtype=dtype, device=device
    )
    if type == "poisson":
        return data_nanmean.repeat((C, 1)) + factor * to.poisson(
            to.ones([C, len(data_nanmean)], dtype=dtype, device=device)
        )
    elif type == "kmeans++":
        print("Running k-means++ algorithm...", flush=True)
        centers, _ = kmeans_plusplus(data.numpy(), C)
        return to.from_numpy(centers)
    else:
        raise NotImplementedError("Noise type {} not supported".format(type))

def init_pies_default(
    C: int,
    dtype: to.dtype = to.float64,
    device: to.device = None,
):
    """Initialize pi parameter.

    :param C: Length of pi vector.
    :param dtype: dtype of output Tensor. Defaults to torch.float64.
    :param device: torch.device of output Tensor. Defaults to tvo.get_device().
    :returns: Vector pi.
    """

    return to.full((C,), fill_value=1 / C, dtype=dtype, device=device)
