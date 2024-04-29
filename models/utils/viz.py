# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_as_line(name: str, array: np.ndarray, out: str):
    """Visualize variable as line.

    :param name: Name of variable
    :param array: Array to be visualized
    :param out: path to output folder
    """
    array = np.array(array)
    plt.figure()
    x = np.arange(len(array))
    plt.plot(x, array)
    plt.ylabel(name)
    plt.xlabel("iterations")
    plt.savefig(out + name + ".png")


def plot_pi(pies: np.ndarray, out: str):
    """Visualize pies according to their activations.

    :param pies: pies with shape (C,)
    :param out: path to output folder
    """
    pies = np.array(pies)
    pies.sort()
    pies = pies[::-1]
    plt.figure()
    x = np.arange(len(pies))
    plt.scatter(x, pies, marker=".")
    plt.ylabel("pi")
    plt.xlabel("c")
    plt.savefig(out + "pi.png")


def plot_weights(W: np.ndarray, pies: np.ndarray, out: str, number: int=10, cmap: str="gray"):
    """ Visualize number**2 weights (in a square layout).

    :param W: weights with shape (C,D)
    :param pies: pies with shape (C,) (to sort the weights according to their activations)
    :param out: path to output folder
    :param number: Size of the axes of the layout square
    :param cmap: Colormap
    """
    W, pies = np.array(W), np.array(pies)
    D = int(np.sqrt(W.shape[-1]))
    idx = pies.argsort()[::-1][: number**2]
    shape = round(np.sqrt(len(idx)))

    W_plot = W[idx]
    vmax, vmin = W_plot.max(), W_plot.min()
    f, axs = plt.subplots(ncols=shape, nrows=shape, figsize=(number, number))

    for i in range(shape**2):
        x, y = int(i / shape), i % shape
        axs[x, y].imshow(W_plot[i].reshape(D, D), cmap=cmap, vmin=vmin, vmax=vmax)
        axs[x, y].axis("off")

    plt.savefig(out + "weights.png")


def viz_theta(out: str):
    """Visualize the model parameters.

    :param out: Path to folder where the model parameters are stored.
    """
    f = h5py.File(out + "model.h5", "r")
    out += "theta/"
    os.makedirs(out, exist_ok=True)

    for k in ["Loglikelihood"]:
        if k in f:
            plot_as_line(k, f[k], out)
    plot_pi(f["pies"], out)
    plot_weights(f["W"], f["pies"], out)
