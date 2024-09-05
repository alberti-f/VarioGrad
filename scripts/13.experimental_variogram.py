import numpy as np
from variograd_utils import *
import psutil
import sys, os

algorithm = str(sys.argv[1])
dim = int(sys.argv[2])-1
grd = int(sys.argv[3])-1

process = psutil.Process()

nbins = 100
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
    geo_dists /= geo_dists.max(axis=1, keepdims=True)

    print("memory used:", process.memory_info().rss / 1e9)



    # Calculate functional distances
    print(f"\nCalculating distances in functional space")

    hemi = slice(vertex_info_10k.grayl.size) if h == "L" else slice(vertex_info_10k.grayl.size, None)
    gradients = np.vstack([np.load(subject(id).outpath(f"{id}.REST_FC_embedding.npy"))[hemi, grd] for id in data.subj_list], dtype="float32").T

    ############################################################################################################
    gradients = 2 * (gradients - gradients.min(axis=1, keepdims=True)) / (gradients.max(axis=1, keepdims=True) - gradients.min(axis=1, keepdims=True)) - 1

    gradients -= gradients.mean(axis=0, keepdims=True) ############# 2DO: Replace with regression
    gradients = (gradients - gradients.mean(axis=0, keepdims=True)) / gradients.std(axis=0, keepdims=True)

    print("\nRow- column-wise stats:\n", gradients.shape, 
          gradients.mean(axis=0), gradients.std(axis=0), gradients.mean(axis=0).shape,
          gradients.mean(axis=1), gradients.std(axis=1), gradients.mean(axis=1).shape)
    ############################################################################################################

    fun_dists = np.empty(geo_dists.shape)
    for v, vertex in enumerate(gradients):
        fun_dists[v] = abs(np.subtract(vertex[row], vertex[col], dtype="float32"))

    print("memory used:", process.memory_info().rss / 1e9)



    # Calculate experimental variograms 
    print("\nCalculating vertex-wise variograms")
    print(f"N bins: {nbins}, overlap: {overlap*100}%, min pairs: {min_pairs}\n")

    ############################################################################################################
    bins = np.array(bins_ol(0, np.percentile(geo_dists, 90), nbins=nbins, overlap=overlap)).T
    # bins = np.array(bins_ol(0, geo_dists.max(), nbins=nbins, overlap=overlap)).T
    ############################################################################################################

    variograd = np.empty([fun_dists.shape[0], bins.shape[0]])
    lags = []
    for bin, (lo, up) in enumerate(bins):

        mask = np.logical_and(geo_dists >= lo, geo_dists <= up)
        if mask.sum(axis=1).min() < min_pairs:
            variograd = variograd[:, :bin]
            break
        lags.append((lo + up) / 2)

        s = 0.25 * (bins[bin, 1] - bins[bin, 0])
        W = np.subtract(geo_dists, lags[bin], dtype="float32")
        W = np.exp(-0.5 * (W ** 2) / (s ** 2)) * mask
        W = W / W.sum(axis=1, keepdims=True)

        variograd[:, bin] =  0.5 * np.sum(np.square(fun_dists) * W , axis=1)

        print(f"Bin {bin+1}\th={lags[bin]:.1E}\t\u03B3(h) = {variograd[:, bin].mean():.2f}({variograd[:, bin].std():.2f})")
        print(f"\tbin=[{lo:.3f}, {up:.3f}]\tmin pairs={mask.sum(axis=1).min()}\tmax pairs={mask.sum(axis=1).max()}\tmean % pairs={mask.mean(axis=1).mean() * 100:.0f}\n")

    print(f"Bins used: {len(lags)}")
    print("memory used:", process.memory_info().rss / 1e9)


    print("memory used:", process.memory_info().rss / 1e9)



    if not os.path.exists(data.outpath(f"/variograms")):
        os.mkdir(data.outpath(f"/variograms"))

    np.save(data.outpath(f"variograms/variogram.{h}.{algorithm}.ax{dim}_G{grd}.npy"), variograd)