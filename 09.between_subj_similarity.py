# compute between-subject similarity

import numpy as np
from variograd_utils import *
from os.path import exists
import sys

pair_idx = sys.argv[1]-1

overwrite = True
algorithm = ["GCCA", "JE"]


data =  dataset()
embed_l, embed_r = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))

data.subj_list = data.subj_list      ####################
n_comps = embed_l[list(embed_l.keys())[0]].shape[2]
n_ks = len(embed_l.keys())
vinfo = vertex_info_10k

skip_algs_l = skip_algs_r =[]
if exists(data.outpath("All.L.between_subj_similarity.npz")) and not overwrite:
    skip_algs_l = np.load(data.outpath(f"All.L.between_subj_similarity.npz")).keys()
if exists(data.outpath("All.R.between_subj_similarity.npz")) and not overwrite:
    skip_algs_r = np.load(data.outpath(f"All.R.between_subj_similarity.npz")).keys()


subj_pair = data.pairs[pair_idx]
subj_i = subject(subj_pair[0])
subj_j = subject(subj_pair[1])


correlations_l = {}
correlations_r = {}
for k in embed_l.keys():

    if k in skip_algs_l and not overwrite:
        print(f"{k} skipped")
        continue

    correlations_l[k] = vector_wise_corr(embed_l[k][subj_i.idx].T, embed_l[k][subj_j.idx].T).astype("float32")
    correlations_r[k] = vector_wise_corr(embed_r[k][subj_i.idx].T, embed_r[k][subj_j.idx].T).astype("float32")

npz_update(data.outpath(f"{subj_i.id}-{subj_j.id}.L.embed_similarity.npz"), correlations_l)
npz_update(data.outpath(f"{subj_i.id}-{subj_j.id}.R.embed_similarity.npz"), correlations_r)

print(f"{k} completed")


# If all pairs are done, stack them for significance testing
# and remove the individual files
if pair_idx == 9:      #len(data.pairs)-1:
    print("stacking all pairs for significance testing")
    
    correlations_l = {k : np.zeros([len(data.pairs), vinfo.grayl.size]) for k in embed_l.keys()}
    correlations_r = {k : np.zeros([len(data.pairs), vinfo.grayr.size]) for k in embed_r.keys()}

    for id_i, id_j in data.pairs:
        idx_i = subject(id_i).idx
        idx_j = subject(id_j).idx

        simil_l = np.load(data.outpath(f"{id_i}-{id_j}.L.embed_similarity.npz"))
        for k, v in simil_l.items():
            correlations_l[k][pair_idx] = v.astype("float32")

        simil_r = np.load(data.outpath(f"{id_i}-{id_j}.R.embed_similarity.npz"))
        for k, v in simil_r.items():
            correlations_r[k][pair_idx] = v.astype("float32")

        os.remove(data.outpath(f"{id_i}-{id_j}.L.embed_similarity.npz"))
        os.remove(data.outpath(f"{id_i}-{id_j}.R.embed_similarity.npz"))


###################################################################################

