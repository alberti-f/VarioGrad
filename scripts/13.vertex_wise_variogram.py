import psutil
import sys, os
from itertools import product
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from variograd_utils.core_utils import dataset, subject, npz_update
import variograd_utils.variography_utils as vu
from variograd_utils.brain_utils import vertex_info_10k

# Iterable parameters
alg_idx = int(sys.argv[1]) - 1
algorithms = ["JE_cauchy50", "JE_cauchy100", "JE_cauchy150", "JE_cauchy200"]
algorithms = [algorithms[alg_idx]]
ndims = [3, 8, 15]
nlagss = [20, 30]
hemispheres = ["L", "R"]
max_dist_fracts = [1, .50, .25]
variogram_models = ["spherical", "exponential", "gaussian"]

# Fixed parameters
grd = 0
overlap = 0
min_pairs = 30
alpha_time = "a05_t1"
detrend = False

# Create iterable parameter combinations
parameters = product(algorithms, ndims, nlagss, max_dist_fracts)
data =  dataset()

for h in hemispheres:
    print("\nHemisphere:", h)
    hemi = slice(vertex_info_10k.grayl.size) if h == "L" else slice(vertex_info_10k.grayl.size, None)
    
    # Loading coordinates in geometric embedding
    print(f"\n\tLoading coordinates in geometric embedding")
    LE_all = data.load_embeddings(h, "JE")

    # Loading coordinates in functional embedding
    print(f"\n\tLoading coordinates in functional embedding")
    load_gradient = lambda ID: np.load(subject(ID).outpath(f'{ID}.FC_embeddings.npz'))[alpha_time][hemi, grd]
    gradients = np.vstack([load_gradient(ID) for ID in data.subj_list], dtype="float32").T

    for algorithm, ndim, nlags, fract in parameters:
        dims = slice(None, ndim)
        LE =  LE_all[algorithm][:, :, dims].transpose(1, 0, 2)

        # Define lags
        max_dist = fract * np.median([np.percentile(euclidean_distances(v), 90) for v in LE])
        lags = np.linspace(0, max_dist, nlags)

        # Preallocate arrays
        exp_variograms = np.full([LE.shape[0], nlags], np.nan)
        lag_pairs =  np.full([LE.shape[0], nlags], np.nan)
        
        results = {"exp_variogram": None,
                   "lags": None,
                   "lag_pairs": None,
                   "spherical_model": np.full([LE.shape[0], 5], np.nan),
                   "exponential_model": np.full([LE.shape[0], 5], np.nan),
                   "gaussian_model": np.full([LE.shape[0], 5], np.nan)}

        # Calculate vertex-wise experimental variograms
        print("\n\tParameters:")
        print(f"\talgorithm: {algorithm}, N dimensions: {ndim}")
        print(f"\t max dist: {max_dist}, N lags: {nlags}, overlap: {overlap*100}%, min pairs: {min_pairs}\n")
        for vtx, (gembd_vtx, fembd_vtx) in enumerate(zip(LE, gradients)):
        
            if gembd_vtx.ndim == 1:
                gembd_vtx = gembd_vtx.reshape(-1, 1)
        
            if fembd_vtx.ndim == 1:
                fembd_vtx = fembd_vtx.reshape(-1, 1)
        
            VG = vu.Variogram()
            VG.omndir_variogram(gembd_vtx.copy(), fembd_vtx.copy() / fembd_vtx.std(),
                                lags, overlap=0, min_pairs=min_pairs)
        
            exp_variograms[vtx] = VG.exp_variogram
            lag_pairs[vtx] = VG.lag_pairs

            # Fit variogram models
            for model in variogram_models:
                bounds = (0, [np.nanmax(VG.exp_variogram), 1, np.nanmax(VG.lags)])
                p0 = [1e-16, 0.5 * np.nanmax(VG.exp_variogram), 0.5 * np.nanmax(VG.lags)]
                curve_fit_kwargs = {"bounds": bounds, "p0": p0, "method": "trf", "maxfev": 6000}
                try:
                    VG.fit_variogram_model(model=model, curve_fit_kwargs=curve_fit_kwargs)

                    results[f"{model}_model"][vtx] = np.array([VG.variogram_model["nugget"],
                                                               VG.variogram_model["contribution"],
                                                               VG.variogram_model["range"],
                                                               VG.variogram_model["sill"],
                                                               VG.variogram_model["r2"]])
                except:
                    results[f"{model}_model"][vtx] = np.full(5, np.nan)

        results["exp_variogram"] = exp_variograms
        results["lags"] = lags
        results["lag_pairs"] = lag_pairs

        # Sve out results
        if not os.path.exists(data.outpath("variograms")):
            os.mkdir(data.outpath("variograms"))
        filename = f"variograms/{data.id}.{h}.variogram_{algorithm}.nd{ndim}_nl{nlags}_mxl{int(fract * 100)}.npy"
        npz_update(data.outpath(filename), results)
