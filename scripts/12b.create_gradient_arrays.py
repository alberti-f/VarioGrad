import os
import sys
from itertools import product
import warnings
import numpy as np
from variograd_utils import dataset, subject, vector_wise_corr
from variograd_utils.brain_utils import vertex_info_10k as vinfo

dataset_id = sys.argv[1]

grads = [0, 1, 2]
alpha_time = ["a05_t0"]
Hs = ["L", "R"]

print(
    f"\n\nGenerating Gratient CSV arrays for the {dataset_id} sample",
    f"\n\talpha_diffusiontime: {alpha_time}\n",
)

params = product(grads, alpha_time, Hs)

data = dataset(dataset_id)

for g, at, H in params:
    if H=="L":
        nvtx = vinfo.grayl.size
        hemi = slice(None, vinfo.grayl.size)
    elif H=="R":
        nvtx = vinfo.grayr.size
        hemi = slice(vinfo.grayl.size, None)
    
    
    # Load gradients
    grads = np.zeros([data.N, nvtx])
    
    for ID in data.subj_list:
        
        subj = subject(ID, data.id)
        grads[subj.idx] = np.load(subj.outpath(f"{ID}.FC_embeddings.npz"))[at][hemi, g]

    ref = grads[0,:].reshape(-1,1)
    corrcoefs = vector_wise_corr(ref, grads.T)
    if np.any(corrcoefs < 0):
        warnings.warn(f"{sum(corrcoefs < 0)} inverted gradients in the array",
                      RuntimeWarning)
        
    outpath = data.outpath(f"{data.id}.{H}.FC_embeddings.{at}.G{g+1}.csv")
    np.savetxt(outpath, grads, delimiter=",")
    print(
        f"\t\tHemisphere {H}, Gradient {g+1}: done"
    )