# Denoising with Poisson Mixture Models

This repository provides a PyTorch package for Denoising with Poisson Mixture Models (PMM).

To run experiments described in the paper with the transmission electron microscopy data (SARS-CoV-2 and Cilia) and fluorescence microscopy data, check out the [examples/microscopy-denoising](/examples/microscopy-denoising/).

After following the [Setup](#setup) instructions described below, you will be able to run these experiments. Please consult the READMEs in [examples/microscopy-denoising](examples/microscopy-denoising/README.md) for further instructions.

The code has only been tested on Linux systems.

## Setup

Clone this repository:

```bash
git clone git@github.com:salwig/pmm.git
cd pmm
```

After cloning the repository, do:

```bash
git submodule init
git submodule update
```

to get the [tvutil](https://github.com/tvlearn/tvutil) as submodule.

We recommend [Anaconda](https://www.anaconda.com/) to manage the installation, and to create a new environment for hosting installed packages:

```bash
conda create -n pmm python==3.9
conda activate pmm
```

The packages specified in [`requirements.txt`](requirements.txt) can be installed with:

```bash
pip install -r requirements.txt
```
