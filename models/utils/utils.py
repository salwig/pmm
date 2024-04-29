# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import torch as to
from torch import Tensor
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Custom Dataset to get indices of the training data
    """

    def __init__(self, data: Tensor, device: to.device = None):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Tensor):
        return self.data[idx].to(self.device), idx


def to_cpu(tensor: Tensor):
    """Move Tensor to cpu
    
    :param tensor: Tensor to be moved on cpu
    """
    return tensor.cpu() if tensor.is_cuda else tensor


def lpj2pjc(lpj: to.Tensor, mode: str = "pjc"):
    """Shift log-pseudo-joint and convert log- to actual probability

    :param mode: wheter return probability tensor or loglikelihood
    :returns: probability tensor or loglikelihood
    """
    up_lpg_bound = 0.0
    shft = up_lpg_bound - lpj.max(dim=1, keepdim=True)[0]
    tmp = to.exp(lpj + shft)
    s = tmp.sum(dim=1, keepdim=True)
    return tmp.div_(s) if mode in ("pjc",) else (to.log(s) - shft).sum()
