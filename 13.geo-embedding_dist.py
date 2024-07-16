# inter-subject distance of vertex weight on latent dimensions

import numpy as np
from variograd_utils import *
from joblib import Parallel, delayed
import sys

algorithm = "JE_cauchy100"

data =  dataset()
LEl, LEr = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))


n_subj = data.N
n_comps = LEl[list(LEl.keys())[0]].shape[2]

LE_dists_l = {k: np.empty_like([n_subj, n_comps]) for k in LEl.keys()}
LE_dists_r = {k: np.empty_like([n_subj, n_comps]) for k in LEr.keys()}

# indices of triu of the subject-component pairs to compare for a given vertex
row, col = diagmat_triu_idx(n_subj*n_comps, n_subj, 1)

n_vtx = vertex_info_10k.grayl.size
for k, LE in LEl.items():
    LE = np.transpose(LE, (1,2,0))
    dists = Parallel(n_jobs=-1)(delayed(np.subtract)(v_mat.flatten()[row], v_mat.flatten()[col]) for v_mat in LE)
    LE_dists_l = abs(np.array(dists)).reshape(n_vtx, n_comps, -1).transpose((1,0,2))
    np.save(data.outpath(f"All.L.embedded_dist.{k}.npy"), LE_dists_l)

n_vtx = vertex_info_10k.grayr.size
for k, LE in LEr.items():
    LE = np.transpose(LE, (1,2,0))
    dists = Parallel(n_jobs=-1)(delayed(np.subtract)(v_mat.flatten()[row], v_mat.flatten()[col]) for v_mat in LE)
    LE_dists_r = abs(np.array(dists)).reshape(n_vtx, n_comps, -1).transpose((1,0,2))
    np.save(data.outpath(f"All.R.embedded_dist.{k}.npy"), LE_dists_r)



# 0m35.9s