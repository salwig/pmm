# Copyright (C) 2023 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import os
import h5py

def store_as_h5(theta: dict, ll: float, out: str):
    '''Store the model paramaters and loglikelihood
    
    :param theta: dict with model paramaters
    :param ll: loglikelihood
    :param out: Path to folder in which the model parameters will be stored
    '''
    h5_file = out + "model.h5"

    if not os.path.isfile(h5_file):
        C, D = theta["W"].shape
        f = h5py.File(h5_file, "w")
        f.create_dataset("C", data=(C,))
        f.create_dataset("D", data=(D,))
        if "W" in theta:
            f.create_dataset("W", data=theta["W"].cpu())
        if "pies" in theta:
            f.create_dataset("pies", data=theta["pies"].cpu())
        f.create_dataset("Loglikelihood", data=(ll,), chunks=True, maxshape=(None,))
    else:
        f = h5py.File(h5_file, "a")
        f["W"][...] = theta["W"].cpu()
        f["pies"][...] = theta["pies"].cpu()
        f["Loglikelihood"].resize((f["Loglikelihood"].shape[0] + 1), axis=0)
        f["Loglikelihood"][-1:] = ll
    f.close()
