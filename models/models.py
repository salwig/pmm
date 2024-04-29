# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import torch as to
from abc import ABCMeta, abstractmethod
from torch import Tensor
import time
import math

from torch.utils.data import DataLoader

from models.utils.logger import store_as_h5
from models.utils.utils import CustomDataset, to_cpu


class Model:
    __metaclass__ = ABCMeta

    def __init__(
        self,
        C: int,
        D: int,
        W_init: Tensor = None,
        pies_init: Tensor = None,
        precision: to.dtype = to.float64,
        device: to.device = None,
    ):
        """Abstract base class to derive concrete models.

        :param C: Number of clusters
        :param D: Number of observables
        :param W_init: Tensor with shape (C,D), initializes weights.
        :param pies_init: Tensor with shape (C,), initializes priors.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.
        :param device: torch.device.
        """
        self.C = C
        self.D = D
        self.eps = to.finfo(precision).tiny
        self.precision = precision
        self.device = device

        if W_init is not None:
            assert W_init.shape == (C, D)
            W_init = W_init.to(dtype=precision, device=device)
        else:
            W_init = to.rand((C, D), dtype=precision, device=device)

        W_init += self.eps

        if pies_init is not None:
            assert pies_init.shape == (C,)
            pies_init = pies_init.to(dtype=precision, device=device)
        else:
            pies_init = to.full((C,), 1.0 / C, dtype=precision, device=device)

        self.theta = {
            "pies": pies_init,
            "W": W_init,
        }
        inf = math.inf
        self.policy = {
            "W": [
                None,
                to.full_like(self.theta["W"], self.eps),
                to.full_like(self.theta["W"], inf),
            ],
            "pies": [
                None,
                to.full_like(self.theta["pies"], self.eps),
                to.full_like(self.theta["pies"], 1.0 - self.eps),
            ],
        }

        self.my_Wp = to.zeros((C, D), dtype=precision, device=device)
        self.my_Wq = to.zeros((C, D), dtype=precision, device=device)
        self.my_pies = to.zeros(C, dtype=precision, device=device)

    @abstractmethod
    def log_pseudo_joint(self, data: Tensor) -> Tensor:
        """Evaluate log-pseudo-joints.

        :param data: dataset 
        :return: log-pseudo joints
        """
        pass

    @abstractmethod
    def log_joint(self, data: Tensor, lpj: Tensor = None) -> float:
        """Evaluate log-joints.

        :param data: dataset
        :param: lpj: Precomputed log-pseudo joints 
        :return: log joints
        """
        pass

    @abstractmethod
    def m_step(self, data: Tensor, lpj: Tensor):
        """M-step: Compute parts of the M-step updates.

        :param data: dataset
        :param: lpj: log-pseudo joints 
        """
        pass

    @abstractmethod
    def m_step_finalize(self, N: int):
        """M-step: Compute the updated model parameters.

        :param N: Number of data points
        """
        pass

    def fit(self, data: Tensor, no_epochs: int, out: str = None, verbose: str = True):
        """Fit model to data (if whole data set fits in memory)

        :param data: data set
        :param no_epochs: Number of epochs
        :param out: Path to folder in which the model parameters will be stored
        :param verbose: Whether to print loglikelihood, epochs and runtime
        """
        timer = 0
        N = data.shape[0]
        data = data.to(self.device)

        for n in range(no_epochs):
            start = time.time()

            # e-step
            pjc = self.log_pseudo_joint(data)
            ll = self.log_joint(data, pjc)

            # m-step
            self.m_step(data, pjc)
            self.m_step_finalize(N)

            timer += start - time.time()

            if verbose:
                print(f"Loglikelihood/N: {ll/N:<10.5f}", flush=True)
                print(f"Iteration: {n+1}", flush=True)
                print(
                    f"\ttotal epoch runtime: {start - time.time():<5.2f} s", flush=True
                )

            if out:
                store_as_h5(self.theta, ll, out)

    def batch_fit(
        self,
        data: Tensor,
        no_epochs: int,
        batch_size: int,
        out: str = None,
        verbose: str = True,
    ):
        """Fit model to data batchwise

        :param data: data set
        :param no_epochs: Number of epochs
        :param batch_size: Batch size used by the data loader
        :param out: Path to folder in which the model parameters will be stored
        :param verbose: Whether to print loglikelihood, epochs and runtime
        """

        timer = 0
        N = data.shape[0]
        dl = DataLoader(CustomDataset(data, self.device), batch_size, shuffle=False)
        pjc = to.zeros([N, self.C], dtype=self.precision)

        for n in range(no_epochs):
            ll = 0
            start = time.time()
            # do batch-wise e-step
            for batch, idx in dl:
                pjc[idx] = to_cpu(self.log_pseudo_joint(batch))
                ll += to_cpu(self.log_joint(batch, pjc[idx].to(self.device)))

            # do batch-wise m-step
            for batch, idx in dl:
                self.m_step(batch, pjc[idx].to(self.device))
            self.m_step_finalize(N)

            timer += start - time.time()

            if verbose:
                print(f"Loglikelihood/N: {ll/N:<10.5f}", flush=True)
                print(f"Iteration: {n+1}", flush=True)
                print(
                    f"\ttotal epoch runtime: {time.time() - start:<5.2f} s", flush=True
                )

            if out:
                store_as_h5(self.theta, ll, out)

    def compute_ll(self, data: Tensor) -> float:
        """Compute the LogLikelihood

        :param data: data set
        :return: loglikelihood
        """
        return self.log_joint(data)

    @abstractmethod
    def generate_data(self, N: int) -> Tensor:
        """Generate data according to the model.
        
        :param N: Number of data points to be generated
        :return: generated data with shape (N,D)
        """
        pass

    @abstractmethod
    def estimate(self, data: Tensor) -> Tensor:
        """Estimate non-noisy data based on posterior predictive distribution (only if data set fits in memory).

        :param data: data set
        :return: reconstructed data with shape (N,D)
        """
        pass

    @abstractmethod
    def estimate_batch(self, data: Tensor, batch_size: int) -> Tensor:
        """Estimate non-noisy data based on posterior predictive distribution batchwise.

        :param data: data set
        :param batch_size: batch size used by the data loader
        :return: reconstructed data with shape (N,D)
        """
        pass
