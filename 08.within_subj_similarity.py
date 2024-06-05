# compute within-subject similarity

import numpy as np
from variograd_utils import *
from sklearn.metrics.pairwise import euclidean_distances
from os.path import exists

# Parameters used by the script
overwrite = True
algorithm = ["JE", "GCCA"]
radius = [50, 999]
size = 150

print("Computing within subject similarity")
print("\toverwrite previous data:", overwrite)
print("\tcompared algorithms:", algorithm)
print("\tradius of the comparisons:", radius)
print("\tN vertices sampled per radius:", size)

data =  dataset()
data.subj_list = data.subj_list[:10]
data.N = len(data.subj_list)

embed_l, embed_r = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))

n_comps = embed_l[list(embed_l.keys())[0]].shape[2]
n_ks = len(embed_l.keys())
vinfo = vertex_info_10k


# Check for the existence of previous computations and avoit overwriting if overwrite is False
skip_algs_l = skip_algs_r =[]
if exists(data.outpath("All.L.within_subj_similarity.npz")) and not overwrite:
    skip_algs_l = np.load(data.outpath(f"All.L.within_subj_similarity.npz")).keys()
if exists(data.outpath("All.R.within_subj_similarity.npz")) and not overwrite:
    skip_algs_r = np.load(data.outpath(f"All.R.within_subj_similarity.npz")).keys()

if overwrite:
    correlations_l = {k: np.zeros([data.N, len(radius), vinfo.grayl.size]) for k in embed_l.keys()}
    correlations_r = {k: np.zeros([data.N, len(radius), vinfo.grayr.size]) for k in embed_r.keys()}
else:
    correlations_l = {k: np.zeros([data.N, len(radius), vinfo.grayl.size]) for k in embed_l.keys() if k not in skip_algs_l}
    correlations_r = {k: np.zeros([data.N, len(radius), vinfo.grayr.size]) for k in embed_r.keys() if k not in skip_algs_r}



# Randomly mask vertices within a given radius
def random_masking(x, rad=None, size=None):
    return np.random.choice(np.argwhere(x<=rad).squeeze(), size=size, replace=False)

radius_masks_l = {}
for r in radius:
    radius_masks_l[r] = np.apply_along_axis(random_masking, 0, data.load_gdist_matrix("L").astype("int32"), rad=r, size=size)

radius_masks_r = {}
for r in radius:
    radius_masks_r[r] = np.apply_along_axis(random_masking, 0, data.load_gdist_matrix("R").astype("int32"), rad=r, size=size)


print("Iterating over subjects:")
# Compute the correlation between the vertex distances in physical and latent space
for id in data.subj_list:

    subj = subject(id)

    gdist_matrix = subj.load_gdist_matrix("L").astype("float32")
    idx = np.arange(gdist_matrix.shape[0])
    for k in embed_l.keys():

        if k in skip_algs_l and not overwrite:
            continue

        edist_matrix = euclidean_distances(subj.load_embeddings("L", k)[k]).astype("float32")

        for i, r in enumerate(radius):

            mask = radius_masks_l[r]
            correlations_l[k][subj.idx, i] = vector_wise_corr(gdist_matrix.copy()[idx, mask], 
                                                              edist_matrix.copy()[idx, mask]).astype("float32")
        
    
    gdist_matrix = subj.load_gdist_matrix("R").astype("float32")
    idx = np.arange(gdist_matrix.shape[0])
    for k in embed_r.keys():

        if k in skip_algs_r and not overwrite:
            continue

        edist_matrix = euclidean_distances(subj.load_embeddings("R", k)[k]).astype("float32")

        for i, r in enumerate(radius):

            mask = radius_masks_r[r]
            correlations_r[k][subj.idx, i] = vector_wise_corr(gdist_matrix.copy()[idx, mask], 
                                                              edist_matrix.copy()[idx, mask]).astype("float32")
        

        
    print(f"\tSubject {subj.idx}/{data.N} completed")



npz_update(data.outpath("All.L.within_subj_similarity.npz"), correlations_l)
npz_update(data.outpath("All.R.within_subj_similarity.npz"), correlations_r)


############################################################################




# Compare  within subject similarity

import numpy as np
from variograd_utils import *
from itertools import combinations
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection


alpha = 0.05

data = dataset()
alg_pairs = combinations(algorithm, 2)

print("\n\nComparing the within-subject similarity of vertex distances in physical \nand latent space between algorithms.", 
      f"\n Compared algorighms: ", "\n\t".join(algorithm))

for h in ["L", "R"]:
    t_maps = {}
    p_maps = {}

        
    correlations = np.load(data.outpath(f"All.{h}.within_subj_similarity.npz"))

    param_pairs = []
    for alg_i, alg_j in alg_pairs:
        param_pairs.extend([ (i, j) for i in correlations.keys() for j in correlations.keys()
                            if i.startswith(alg_i) and j.startswith(alg_j)])


    print(f"\nHemisphere:", h, "\nN comparisons:", len(param_pairs))

    for param_i, param_j in param_pairs: 
        X = correlations[param_i]
        Y = correlations[param_j]
        results = ttest_rel(X, Y)
        pvalues = np.array([fdrcorrection(p, alpha)[1] for p in results.pvalue])

        t_maps[f"{param_i}-vs-{param_j}"] = results.statistic
        p_maps[f"{param_i}-vs-{param_j}"] = pvalues

    npz_update(data.outpath(f"{data.id}.{h}.wss_t_maps.npz"), t_maps)
    npz_update(data.outpath(f"{data.id}.{h}.wss_p_maps.npz"), p_maps)
    
