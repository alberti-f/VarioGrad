# identification accuracy

import numpy as np
from variograd_utils import *
import gc
from os.path import exists

algorithm = "JE_m1000"
h="L"

data =  dataset()
LEl, LEr = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))

data.subj_list = data.subj_list
n_subj = len(data.subj_list)
n_comps = LEl[list(LEl.keys())[0]].shape[2]
n_ks = len(LEl.keys())

if exists(data.outpath(f"All.{h}.embed_detection_mats.npz")):
        skip_algs = np.load(data.outpath(f"All.{h}.embed_detection_mats.npz")).keys()
        
corrmats_l = {k: np.zeros([n_comps, n_subj, n_subj]) for k in LEl.keys() if k not in skip_algs}
corrmats_r = {k: np.zeros([n_comps, n_subj, n_subj]) for k in LEr.keys() if k not in skip_algs}

corrmats_l = {k: np.zeros([n_comps, n_subj, n_subj]) for k in LEl.keys()}
corrmats_r = {k: np.zeros([n_comps, n_subj, n_subj]) for k in LEr.keys()}

for key, embed in LEl.items():
    if key in skip_algs:
         print(key, "skipped\n")
         continue
    print(key)
    for id in data.subj_list:
        subj_i = subject(id)
        i = subj_i.idx
        gdist_vec_i =subj_i.load_gdist_vector(h).reshape(-1, 1)

        
        for j, edist_vec_j in enumerate(embed):
            edist_vec_j = euclidean_triu(edist_vec_j)
            R = vector_wise_corr(gdist_vec_i.copy(), edist_vec_j)
            del edist_vec_j
            gc.collect()

            corrmats_l[key][:, i, j] = R
        
        print(f"\tCompleted subject {id}")
        
    detection_pct_l = np.array([(np.diagonal(M) == M.max(axis=1)).mean(axis=0) * 100 for M in corrmats_l[key]])
    npz_update(data.outpath(f"All.{h}.embed_detection_mats.npz"), {key: corrmats_l[key]})
    npz_update(data.outpath(f"All.{h}.detection_pct.npz"), {key: detection_pct_l})
    
    print(key, "Done\n")



