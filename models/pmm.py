# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import torch as to
from torch import Tensor
from torch.utils.data import DataLoader
from torch.distributions.one_hot_categorical import OneHotCategorical

from models.models import Model
from models.utils.utils import CustomDataset, lpj2pjc, to_cpu
from models.utils.sanity import fix_theta

class PMM(Model):
    def __init__(
        self,
        C: int,
        D: int,
        W_init: Tensor = None,
        pies_init: Tensor = None,
        precision: to.dtype = to.float64,
        device: to.device = None,
    ):
        """Model class to derive PMM model.

        :param C: Number of clusters
        :param D: Number of observables
        :param W_init: Tensor with shape (C,D), initializes weights.
        :param pies_init: Tensor with shape (C,), initializes priors.
        :param precision: Floating point precision required. Must be one of torch.float32 or
                          torch.float64.
        :param device: torch.device.
        """
        Model.__init__(self, C, D, W_init, pies_init, precision, device)

    def log_pseudo_joint(self, data: Tensor) -> Tensor:
        """Evaluate log-pseudo-joints.

        :param data: dataset 
        :return: log-pseudo joints
        """
        W = self.theta["W"]
        lpj = (
            to.matmul(data, to.log(W).t())
            - W.sum(dim=1)[None, :]
            + to.log(self.theta["pies"])[None, :]
        )
        return lpj

    def log_joint(self, data: Tensor, lpj: Tensor = None) -> float:
        """Evaluate log-joints.

        :param data: dataset
        :param: lpj: Precomputed log-pseudo joints 
        :return: log joints
        """
        if lpj is None:
            lpj = self.log_pseudo_joint(data)

        lpj -= to.sum(to.lgamma(data + 1), dim=1).unsqueeze(1)
        return lpj2pjc(lpj.to(self.device), mode="ll")

    def m_step(self, data: Tensor, lpj: Tensor):
        """M-step: Compute parts of the weights and pies updates.

        :param data: dataset
        :param: lpj: log-pseudo joints 
        """
        pjc = lpj2pjc(lpj)

        self.my_pies += to.sum(pjc, dim=0)
        self.my_Wp += to.matmul(pjc.t(), data)
        self.my_Wq += to.sum(pjc, dim=0).unsqueeze(1)

    def m_step_finalize(self, N: int):
        """M-step: Compute the updated weights and pies.

        :param N: Number of data points
        """
        theta = self.theta
        policy = self.policy

        theta_new = {}
        # Calculate updated W
        theta_new["W"] = self.my_Wp / self.my_Wq + self.eps

        theta_new["pies"] = self.my_pies / N

        policy["W"][0] = theta["W"]
        policy["pies"][0] = theta["pies"]
        fix_theta(theta_new, policy)
        for key in theta_new:
            theta[key][:] = theta_new[key]

        self.my_Wp[:] = 0.0
        self.my_Wq[:] = 0.0
        self.my_pies[:] = 0.0

        self.theta = theta

    def generate_data(self, N: int) -> Tensor:
        """Generate data according to the PMM model.
        
        :param N: Number of data points to be generated
        :return: generated data with shape (N,D)
        """
        W = self.theta["W"]
        pies = self.theta["pies"]
        hidden_state = to.argmax(OneHotCategorical(probs=pies).sample([N]), dim=1)
        return to.poisson(W[hidden_state, :])

    def estimate(self, data: Tensor) -> Tensor:
        """Estimate non-noisy data based on posterior predictive distribution (only if data set fits in memory).

        :param data: data set
        :return: reconstructed data with shape (N,D)
        """
        data = data.to(self.device)
        lpj = self.log_pseudo_joint(data)
        pjc = lpj2pjc(lpj)
        return to_cpu(to.matmul(pjc, self.theta["W"]))

    def estimate_batch(self, data: Tensor, batch_size: int) -> Tensor:
        """Estimate non-noisy data based on posterior predictive distribution batchwise.

        :param data: data set
        :param batch_size: batch size used by the data loader
        :return: reconstructed data with shape (N,D)
        """
        dl = DataLoader(CustomDataset(data, self.device), batch_size, shuffle=False)
        reconstructions = to.zeros_like(data, dtype=self.precision)
        for batch, idx in dl:
            lpj = self.log_pseudo_joint(batch)
            pjc = lpj2pjc(lpj)
            reconstructions[idx] = to_cpu(to.matmul(pjc, self.theta["W"]))
        return reconstructions
