# inter-subject distance of vertex weight on latent dimensions

import numpy as np
from variograd_utils import *
from joblib import Parallel, delayed
import sys

algorithm = "JE_cauchy100"

data =  dataset()
LEl = data.load_embeddings("L", algorithm)
print(f"Loaded left embeddings: {LEl.keys()}")

LEr = data.load_embeddings("R", algorithm)
print(f"Loaded right embeddings: {LEr.keys()}")

n_subj = data.N
n_comps = LEl[list(LEl.keys())[0]].shape[2]


# indices of triu of the subject-component pairs to compare for a given vertex
row, col = diagmat_triu_idx(n_subj*n_comps, n_subj, 1)


print ("Calculating left distances:")
n_vtx = vertex_info_10k.grayl.size
for k, LE in LEl.items():
    LE = np.transpose(LE, (1,2,0))
    dists = Parallel(n_jobs=-1)(delayed(np.subtract)(v_mat.flatten()[row], v_mat.flatten()[col]) for v_mat in LE)
    LE_dists_l = abs(np.array(dists)).reshape(n_vtx, n_comps, -1).transpose((1,0,2))
    np.save(data.outpath(f"All.L.embedded_dist.{k}.npy"), LE_dists_l)

    print(f"\t{k} done")

n_vtx = vertex_info_10k.grayr.size
for k, LE in LEr.items():
    LE = np.transpose(LE, (1,2,0))
    dists = Parallel(n_jobs=-1)(delayed(np.subtract)(v_mat.flatten()[row], v_mat.flatten()[col]) for v_mat in LE)
    LE_dists_r = abs(np.array(dists)).reshape(n_vtx, n_comps, -1).transpose((1,0,2))
    np.save(data.outpath(f"All.R.embedded_dist.{k}.npy"), LE_dists_r)

    print(f"\t{k} done")

