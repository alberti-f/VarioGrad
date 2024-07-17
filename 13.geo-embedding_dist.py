# LEr = data.load_embeddings("L", algorithm)

import numpy as np
from variograd_utils import *
import psutil

process = psutil.Process()

algorithm = "JE_cauchy50"
h = "L"

dim = 0
grd = 0
nbins = 100
overlap = 0.25
trim = 1000

data =  dataset()

row, col = np.triu_indices(500, k=1)

# Calculate embedded geometric distances
print(f"\nCalculating distances in embedded space")
LE =  data.load_embeddings(h, algorithm)[algorithm][:, :, dim].T
geo_dists = np.empty([LE.shape[0], row.size])
for v, vertex in enumerate(LE):
    geo_dists[v] = abs(np.subtract(vertex[row], vertex[col], dtype="float32"))
print("memory used:", process.memory_info().rss / 1e9)

# Calculate functional distances
print(f"\nCalculating distances in functional space")
offset = 0 if h == "L" else vertex_info_10k.grayl.size
hemi = offset + vertex_info_10k[f"gray{h.lower()}"]
gradients = np.vstack([np.load(subject(id).outpath(f"{id}.REST_FC_embedding.npy"))[hemi, grd] for id in data.subj_list], dtype="float32").T
fun_dists = np.empty(geo_dists.shape)
for v, vertex in enumerate(gradients):
    fun_dists[v] = abs(np.subtract(vertex[row], vertex[col], dtype="float32"))
print("memory used:", process.memory_info().rss / 1e9)


# Define distance bins
print(f"\nFiltering distance bins by minimum number of pairs per vertex.")
print(f"N bins: {nbins}, overlap: {overlap*100}%, min pairs: {trim}")    
bins = np.array(bins_ol(geo_dists.min(), geo_dists.max(), nbins=nbins, overlap=overlap)).T
bin_masks = {}
for i, (lo, up) in enumerate(bins):
    mask = np.logical_and(geo_dists >= lo, geo_dists < up)
    if mask.sum(axis=1).min() < 1000:
        trim_idx = i-1
        break
    print(f"{i}\t[{lo:.5f}, {up:.5f})\tmin pairs={mask.sum(axis=1).min()}")
    bin_masks[i] = mask
print(f"{trim_idx + 1 } bins included")
print("memory used:", process.memory_info().rss / 1e9)


# for bin, mask in bin_masks.items():
#     print(f"Bin {bin}")
#     fun = fun_dists[mask].mean(axis=1)
#     print(f"Correlation: {np.corrcoef(geo, fun)[0,1]}")