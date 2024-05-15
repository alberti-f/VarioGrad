# within subject correlation of embedded and anatomical vertex distances

import numpy as np
from variograd_utils import *
from joblib import Parallel, delayed
from os.path import exists

algorithm = "JE_m1500_Cauchy"
overwrite = True

data =  dataset()
LEl, LEr = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))

data.subj_list = data.subj_list
n_subj = len(data.subj_list)
n_comps = LEl[list(LEl.keys())[0]].shape[2]
n_ks = len(LEl.keys())


skip_algs_l = np.load(data.outpath(f"AllToAll.L.embed_gdist_correlation.npz")).keys()
skip_algs_r = np.load(data.outpath(f"AllToAll.R.embed_gdist_correlation.npz")).keys()

if overwrite:
    correlations_l = {k: np.zeros([n_subj, n_comps]) for k in LEl.keys()}
    correlations_r = {k: np.zeros([n_subj, n_comps]) for k in LEr.keys()}
else:
    correlations_l = {k: np.zeros([n_subj, n_comps]) for k in LEl.keys() if k not in skip_algs_l}
    correlations_r = {k: np.zeros([n_subj, n_comps]) for k in LEr.keys() if k not in skip_algs_r}


for id in data.subj_list:
    subj = subject(id)

    gdist_vector= subj.load_gdist_vector("L")
    for k, eig_vectors in LEl.items():
        if not overwrite and k in skip_algs_l:
            continue
        
        r = Parallel(n_jobs=5, prefer="threads")(delayed(np.corrcoef)
                                                (gdist_vector, euclidean_triu(eig_vector)) 
                                                for eig_vector in eig_vectors[subj.idx].T)
        correlations_l[k][subj.idx, :] = [corrmat[0,1] for corrmat in r]
    
    gdist_vector= subj.load_gdist_vector("R")
    for k, eig_vectors in LEr.items():
        if k in skip_algs_r and not overwrite:
            continue

        r = Parallel(n_jobs=5, prefer="threads")(delayed(np.corrcoef)
                                                (gdist_vector, euclidean_triu(eig_vector)) 
                                                for eig_vector in eig_vectors[subj.idx].T)
        correlations_r[k][subj.idx, :] = [corrmat[0,1] for corrmat in r]

    print(f"Subject {id} completed")

npz_update(data.outpath("AllToAll.L.embed_gdist_correlation.npz"), correlations_l)
npz_update(data.outpath("AllToAll.R.embed_gdist_correlation.npz"), correlations_r)

