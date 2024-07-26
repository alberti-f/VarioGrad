import numpy as np
from variograd_utils import *
import psutil
import sys, os

algorithm = str(sys.argv[1])
dim = int(sys.argv[2])-1
grd = int(sys.argv[3])-1

process = psutil.Process()

nbins = 200
overlap = 0.25
min_pairs = 30

data =  dataset()

row, col = np.triu_indices(data.N, k=1)

for h in ["L", "R"]:

    # Calculate embedded geometric distances
    print(f"\nCalculating distances in embedded space")

    LE =  data.load_embeddings(h, algorithm)[algorithm][:, :, dim].T
    geo_dists = np.empty([LE.shape[0], row.size])
    for v, vertex in enumerate(LE):
        geo_dists[v] = abs(np.subtract(vertex[row], vertex[col], dtype="float32"))

    print("memory used:", process.memory_info().rss / 1e9)



    # Calculate functional distances
    print(f"\nCalculating distances in functional space")

    hemi = slice(vertex_info_10k.grayl.size) if h == "L" else slice(vertex_info_10k.grayl.size, None)
    gradients = np.vstack([np.load(subject(id).outpath(f"{id}.REST_FC_embedding.npy"))[hemi, grd] for id in data.subj_list], dtype="float32").T
    gradients -= gradients.mean(axis=1, keepdims=True) ############# 2DO: Replace with regression
    gradients = (gradients - gradients.mean(axis=1, keepdims=True)) / gradients.std(axis=1, keepdims=True)

    fun_dists = np.empty(geo_dists.shape)
    for v, vertex in enumerate(gradients):
        fun_dists[v] = abs(np.subtract(vertex[row], vertex[col], dtype="float32"))

    print("memory used:", process.memory_info().rss / 1e9)



    # Define geometric distance bins
    print(f"\nFiltering distance bins by minimum number of pairs per vertex.")
    print(f"N bins: {nbins}, overlap: {overlap*100}%, min pairs: {min_pairs}\n")

    bins = np.array(bins_ol(0, geo_dists.max(), nbins=nbins, overlap=overlap)).T
    bin_masks = {}
    lags = []
    for i, (lo, up) in enumerate(bins):
        mask = np.logical_and(geo_dists >= lo, geo_dists <= up)
        if mask.sum(axis=1).min() < min_pairs:
            break
        lags.append((lo + up) / 2)
        bin_masks[i] = mask
        print(f"{i+1}\th={lags[i]:.1E}\tbin=[{lo:.3f}, {up:.3f}]\tmin pairs={mask.sum(axis=1).min()}\tmax pairs={mask.sum(axis=1).max()} \tmean % pairs={mask.mean(axis=1).mean() * 100:.0f}")
    lags = np.array(lags)

    print(f"Bins used: {len(bin_masks.keys())}")
    print(f"Included pairs: {int(np.any([m for m in bin_masks.values()], axis=0).mean() * 100)}%")
    print("memory used:", process.memory_info().rss / 1e9)



    # Calculate experimental variograms 
    print("\nCalculating vertex-wise variograms")

    variograd = np.empty([fun_dists.shape[0], len(bin_masks.keys())])
    for bin, mask in bin_masks.items():
        s = 0.25 * (bins[bin, 1] - bins[bin, 0])
        W = np.subtract(geo_dists, lags[bin], dtype="float32")
        W = np.exp(-0.5 * (W ** 2) / (s ** 2)) * mask
        W = W / W.sum(axis=1, keepdims=True)

        variograd[:, bin] =  0.5 * np.sum(np.square(fun_dists) * W , axis=1)

        print(f"Lag {bin+1}\t\u03B3(h) = {variograd[:, bin].mean():.2f}({variograd[:, bin].std():.2f})")

    print("memory used:", process.memory_info().rss / 1e9)



    if not os.path.exists(data.outpath(f"/variograms")):
        os.mkdir(data.outpath(f"/variograms"))

    np.save(data.outpath(f"variograms/variogram.{h}.{algorithm}.ax{dim}_G{grd}.npy"), variograd)