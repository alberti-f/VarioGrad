# compute between-subject similarity

import numpy as np
from variograd_utils import *
from joblib import Parallel, delayed
from itertools import combinations
from os.path import exists

overwrite = True
algorithm = ["GCCA", "JE"]


data =  dataset()
embed_l, embed_r = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))

data.subj_list = data.subj_list      ####################
n_subj = len(data.subj_list)         #################### can be removed and changed to data.N downstream
n_comps = embed_l[list(embed_l.keys())[0]].shape[2]
n_ks = len(embed_l.keys())
vinfo = vertex_info_10k

subj_pairs = list(combinations(data.subj_list, 2))



skip_algs_l = skip_algs_r =[]
if exists(data.outpath("All.L.between_subj_similarity.npz")) and not overwrite:
    skip_algs_l = np.load(data.outpath(f"All.L.between_subj_similarity.npz")).keys()
if exists(data.outpath("All.R.between_subj_similarity.npz")) and not overwrite:
    skip_algs_r = np.load(data.outpath(f"All.R.between_subj_similarity.npz")).keys()



for k in embed_l.keys():

    if k in skip_algs_l and not overwrite:
        print(f"{k} skipped")
        continue


    correlations_l = {k: np.zeros([len(subj_pairs), vinfo.grayl.size])}
    correlations_r = {k: np.zeros([len(subj_pairs), vinfo.grayr.size])}
    
    
    correlations_l = Parallel(n_jobs=-2, prefer="threads")(delayed(vector_wise_corr)(subject(i).load_embeddings("L", k)[k].T, 
                                                                                     subject(j).load_embeddings("L", k)[k].T) 
                                                                                     for i, j in subj_pairs)
    npz_update(data.outpath("All.L.between_subj_similarity.npz"), {k: np.array(correlations_l)})
    
    correlations_r = Parallel(n_jobs=-2, prefer="threads")(delayed(vector_wise_corr)(subject(i).load_embeddings("R", k)[k].T, 
                                                                                     subject(j).load_embeddings("R", k)[k].T) 
                                                                                     for i, j in subj_pairs)
    npz_update(data.outpath("All.R.between_subj_similarity.npz"), {k: np.array(correlations_r)})
    
    print(f"{k} completed")


###################################################################################

