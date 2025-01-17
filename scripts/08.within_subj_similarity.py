"""
This script calculates and compares the within-subject similarity between geodesic 
distances in physical space and latent embeddings produced by JE and GCCA.

Steps:
1. Compute Within-Subject Similarity:
    - Load geodesic and latent distance matrices for each subject.
    - For each embedding algorithm:
        - Compute Euclidean distances in the latent space.
        - Compute correlations between geodesic and latent distances for each radius.
        - Save correlations in `.npz` files for left and right hemispheres.
2. Compare Within-Subject Similarity:
    - Load correlations for each hemisphere and embedding algorithm.
    - Perform paired t-tests for all parameter pairs across algorithms.
    - Apply FDR correction to control for multiple comparisons.
    - Save t-statistic and p-value maps in `.npz` files.

Dependencies:
    - `variograd_utils`: For dataset and subject handling, and embedding utilities.
    - `numpy`: For numerical computations and file handling.
    - `scipy`: For statistical testing.

Outputs:
    - Correlation maps for within-subject similarity:
        `<output_dir>/All.L.within_subj_similarity.npz`
        `<output_dir>/All.R.within_subj_similarity.npz`
    - T-statistic and p-value maps for algorithm comparisons:
        `<output_dir>/<data_id>.<H>.wss_t_maps.npz`
        `<output_dir>/<data_id>.<H>.wss_p_maps.npz`

"""


from os.path import exists
from itertools import combinations
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from sklearn.metrics.pairwise import euclidean_distances
from variograd_utils import dataset, subject, vector_wise_corr, npz_update
from variograd_utils.brain_utils import vertex_info_10k


# Parameters used by the script
overwrite = True
algorithm = ["JE", "GCCA"]
radius = [50, 100, 150, 200, 999]
size = 100


print("Computing within subject similarity")
print("\toverwrite previous data:", overwrite)
print("\tcompared algorithms:", algorithm)
print("\tradius of the comparisons:", radius)
print("\tN vertices sampled per radius:", size)

data =  dataset()

embed_l, embed_r = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))

n_comps = embed_l[list(embed_l.keys())[0]].shape[2]
n_ks = len(embed_l.keys())
vinfo = vertex_info_10k


# Check for the existence of previous computations and avoit overwriting if overwrite is False
skip_algs_l = skip_algs_r = []
if exists(data.outpath("All.L.within_subj_similarity.npz")) and not overwrite:
    skip_algs_l = np.load(data.outpath("All.L.within_subj_similarity.npz")).keys()
if exists(data.outpath("All.R.within_subj_similarity.npz")) and not overwrite:
    skip_algs_r = np.load(data.outpath("All.R.within_subj_similarity.npz")).keys()

if overwrite:
    correlations_l = {k: np.zeros([data.N, len(radius), vinfo.grayl.size])
                      for k in embed_l.keys()}
    correlations_r = {k: np.zeros([data.N, len(radius), vinfo.grayr.size])
                      for k in embed_r.keys()}
else:
    correlations_l = {k: np.zeros([data.N, len(radius), vinfo.grayl.size])
                      for k in embed_l.keys() if k not in skip_algs_l}
    correlations_r = {k: np.zeros([data.N, len(radius), vinfo.grayr.size])
                      for k in embed_r.keys() if k not in skip_algs_r}



# Randomly mask vertices within a given radius
def random_masking(x, rad=None, size=None):
    """Helper to randomly mask vertices within a given radius."""
    return np.random.choice(np.argwhere(x<=rad).squeeze(), size=size, replace=False)

radius_masks_l = {}
for r in radius:
    radius_masks_l[r] = np.apply_along_axis(random_masking, 0,
                                            data.load_gdist_matrix("L").astype("int32"),
                                            rad=r, size=size)

radius_masks_r = {}
for r in radius:
    radius_masks_r[r] = np.apply_along_axis(random_masking, 0,
                                            data.load_gdist_matrix("R").astype("int32"),
                                            rad=r, size=size)


print("Iterating over subjects:")
# Compute the correlation between the vertex distances in physical and latent space
for ID in data.subj_list:

    subj = subject(ID)

    gdist_matrix = subj.load_gdist_matrix("L").astype("float32")
    idx = np.arange(gdist_matrix.shape[0])
    for k in embed_l.keys():

        if k in skip_algs_l and not overwrite:
            continue

        edist_matrix = euclidean_distances(embed_l[k][subj.idx]).astype("float32")

        for i, r in enumerate(radius):

            mask = radius_masks_l[r]
            correlations_l[k][subj.idx, i] = vector_wise_corr(gdist_matrix.copy()[idx, mask],
                                                              edist_matrix.copy()[idx, mask]
                                                              ).astype("float32")


    gdist_matrix = subj.load_gdist_matrix("R").astype("float32")
    idx = np.arange(gdist_matrix.shape[0])
    for k in embed_r.keys():

        if k in skip_algs_r and not overwrite:
            continue

        edist_matrix = euclidean_distances(embed_r[k][subj.idx]).astype("float32")

        for i, r in enumerate(radius):

            mask = radius_masks_r[r]
            correlations_r[k][subj.idx, i] = vector_wise_corr(gdist_matrix.copy()[idx, mask],
                                                              edist_matrix.copy()[idx, mask]
                                                              ).astype("float32")


    print(f"\tSubject {subj.idx}/{data.N} completed")


npz_update(data.outpath("All.L.within_subj_similarity.npz"), correlations_l)
npz_update(data.outpath("All.R.within_subj_similarity.npz"), correlations_r)
####################################################################################################

# Compare  within subject similarity

alpha = 0.05

data = dataset()
alg_pairs = list(combinations(algorithm, 2))

print("\n\nComparing the within-subject similarity of vertex distances ",
      "\nin physicaland latent space between algorithms.", 
      "\n Compared algorighms:\n\t", "\n\t".join(algorithm))

for h in ["L", "R"]:
    t_maps = {}
    p_maps = {}

    # find pairs of parameters to compare between algorithms
    correlations = np.load(data.outpath(f"All.{h}.within_subj_similarity.npz"))
    param_pairs = []
    for alg_i, alg_j in alg_pairs:
        param_pairs.extend([ (i, j) for i in correlations.keys() for j in correlations.keys()
                            if i.startswith(alg_i) and j.startswith(alg_j) ])


    print("\nHemisphere:", h, "\nN comparisons:", len(param_pairs))

    # t-test for each pair of parameters
    for param_i, param_j in param_pairs: 
        X = correlations[param_i]
        Y = correlations[param_j]
        results = ttest_rel(X, Y)
        pvalues = np.array([fdrcorrection(p, alpha)[1] for p in results.pvalue])

        t_maps[f"{param_i}-vs-{param_j}"] = results.statistic
        p_maps[f"{param_i}-vs-{param_j}"] = pvalues

        print(f"\t{param_i} vs {param_j}:",
              f"\t\t Percent significant: \t{np.sum(pvalues<alpha) / pvalues.size}%",
              f"\t\t Average t-value: \t{np.nanmean(results.statistic[pvalues<alpha])}")

    npz_update(data.outpath(f"{data.id}.{h}.wss_t_maps.npz"), t_maps)
    npz_update(data.outpath(f"{data.id}.{h}.wss_p_maps.npz"), p_maps)
