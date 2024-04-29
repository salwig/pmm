# Denoising microscopy images with PMM

This folder contains code for denoising the SARS-CoV-2 images, Cilia image or the Fluorescence Microscopy images FU-PN2V Convallaria, FU-PN2V Mouse actin and FU-PN2V Mouse nuclei with PMM.

## Datasets

This section contains information about the used datasets. For [SARS-CoV-2](#sars-cov-2) and [Fluorescence Microscopy](#fluorescence-microscopy) the datasets are downloaded automatically as described in [Get started](#get-started).

### SARS-CoV-2

Original data made available by [Laue et al.(2021)](https://www.nature.com/articles/s41598-021-82852-7) who recorded images of ultrathin plastic sections using transmission electron microscopy. We downloaded the data from these Zenodo repositories ([Dataset02](https://zenodo.org/record/3985103#.ZHcVMmFBzJU), [Dataset03](https://zenodo.org/record/3985110#.ZHcVL2FBzJU) and [Dataset07](https://zenodo.org/record/3986580#.Ypd6onVBxH5)).

### Cilia

Original data made available by [Bajic et al.(2018)](https://ieeexplore.ieee.org/document/8363721) who recorded 100 noisy short exposure images showing cilia using transmission electron microscopy. We obtained the original data via personal communication with the authors.  We provide the (preprocessed) noisy image and the "pseudo-ground truth" image for testing [here](https://cloud.uol.de/s/KFYwzQ5jYdjrPoi).

### Fluorescence Microscopy

Original data made available by [Krull et al.(2020)](https://doi.org/10.3389/fcomp.2020.00005). We adapted this benchmark from [Prakash et al.(2021)](https://openreview.net/forum?id=agHLCOBM5jP). We downloaded the data from these Zenodo repositories ([Convallaria](https://doi.org/10.5281/zenodo.5156913), [Mouse skull nuclei](https://doi.org/10.5281/zenodo.5156960) and [Mouse actin](https://doi.org/10.5281/zenodo.5156937))

For further information, see paper.

## Requirements

To run this example, make sure to have completed the installation instructions [described here](../../README.md) and to have the `poissondenoising` environment activated.

```bash
conda activate poissondenoising
```

## Get started

To start an experiment with PMM on the respective dataset, run `python main.py <DATASET>` with `<DATASET>` one of `{sarscov2,cilia,fm}`.

To start an experiment with the Cilia data, run `python main.py cilia <noisy> <gt>`, where `<noisy>` and `<gt>` specify the paths to the noisy cilia image and the pseudo ground truth image, respectively.

To specify the image for the SARS-CoV-2 dataset, run `python main.py sarscov2 --id <ID>` with `<ID>` one of `{1,2,...9} default: 8` (compare Tab. S1 in the Supplement of the manuscript).

To specify the image for the the Fluorescence microscopy dataset, run `python main.py fm --id <ID>` with `<ID>` one of `{convallaria, mouse_skull_nuclei, mouse_actin} default: convallaria`.

The specified image is automatically downloaded for the SARS-CoV-2 dataset and the FM dataset.

For example, to start an experiment with PMM on the SARS-CoV-2 image (ID 3), run:

```bash
python main.py sarscov2 --id 3
```

To see further possible options, run, e.g.:

```bash
python main.py sarscov2 -h 
```

Note that the images are preprocessed in this examples s.t. exemplary executions of the algorithm on a standard personal computer can be performed in short time.
To reproduce the results reported in the paper see the [Reproducibility](#reproducibility) section below.

## Reproducibility

To reproduce the results of the paper, run for PMM:

```bash
python main.py sarscov2 --rescale 1.0
python main.py cilia --rescale 1.0
python main.py fm --rescale 1.0 
```

Additionally, to calculate SNR values for the SARS-CoV-2 dataset, run `python main.py sarscov2 --rescale 1.0 --sbregions <sbregions>`, where `<sbregions>` specifies the path to the file containing the coordinates of the signal and background regions (note that this is only implemented if `--rescale 1.0` is specified).
