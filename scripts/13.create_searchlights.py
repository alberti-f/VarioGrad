import sys
from itertools import product
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from variograd_utils.variography_utils import digitizedd, bin_centers
from variograd_utils import dataset, vertex_info_10k as vinfo

side = np.int32(sys.argv[1])
dataset1_id = "train" 
dataset2_id = "test"
min_vtx = 1000

H_scale = product(["L", "R"], [10, 50, 200])

for H, s in H_scale:

    nvtx = vinfo.grayl.size if H=="L" else vinfo.grayr.size
    algorithm = f"JE_cauchy{s}"
    
    # Load train data
    locations_train = dataset(dataset1_id).load_embeddings(H, algorithm)[algorithm][:,:,:3]
    locations_avg = locations_train.mean(axis=0)
    locations_train = locations_train.reshape(-1,3)
    
    # Compute median neighbors distance
    avg_dists = euclidean_distances(locations_avg)
    avg_dists[avg_dists==0] = avg_dists.max()
    
    # Set searchlight size (bin size)
    sl_side = np.median(avg_dists.min(axis=0)) * side
    
    # Set bin edges along each dimension
    bins = []
    for x in locations_train.T:
        nbins = np.ceil(x.ptp() / sl_side).astype("int32")
        xmin = x.min() - ((sl_side * nbins) - x.ptp()) / 2
        xmax = x.max() + ((sl_side * nbins) - x.ptp()) / 2
        bins.append(np.linspace(xmin, xmax, nbins+1))
    
    # Assign vertices to bins
    srchlight_train = digitizedd(locations_train, bins)
    srchlight_train_id, srchlight_train_n = np.unique(srchlight_train, return_counts=True)
    low_occupancy = srchlight_train_n < min_vtx
    
    
    # Remove scarcely populated bins
    if np.any(low_occupancy):
        # Mark scarcely occupied bins
        srchlight_train_id = srchlight_train_id[~low_occupancy]
        srchlight_train[~np.isin(srchlight_train, srchlight_train_id)] = 0
    
        # Find searchlight center
        srchlight_center = bin_centers(bins)[srchlight_train_id - 1]
    
        # Find vertices out of bounts
        reassigned_vtx = locations_train[srchlight_train == 0]
    
        # Assign vertices to closest searchlight
        dist_from_sl = euclidean_distances(reassigned_vtx, srchlight_center)
        closest_sl = srchlight_train_id[np.argmin(dist_from_sl, axis=1)]
        srchlight_train[srchlight_train == 0] = closest_sl
    
    
    # Load test data
    locations_test = dataset(dataset2_id).load_embeddings(H, algorithm)[algorithm][:,:,:3]
    locations_test = locations_test.reshape(-1,3)
    
    # Assign test data to bins
    srchlight_test = digitizedd(locations_test, bins)
    srchlight_test_ob = (~np.isin(srchlight_test, srchlight_train_id)) | (srchlight_test == 0)
    
    # Assign out-of-bounds test data to closest searchlight
    if np.any(srchlight_test_ob):
        # Find searchlight center
        srchlight_center = bin_centers(bins)[srchlight_train_id - 1]
    
        # Find vertices out of bounts
        outbound_vtx = locations_test[srchlight_test_ob]
    
        # Assign vertices to closest searchlight
        dist_from_sl = euclidean_distances(outbound_vtx, srchlight_center)
        closest_sl = srchlight_train_id[np.argmin(dist_from_sl, axis=1)]
        srchlight_test[srchlight_test_ob] = closest_sl

    # Save out searchlight indices
    srchlight_train = srchlight_train.reshape(-1, nvtx)
    fname = dataset(dataset1_id).outpath(f"{dataset1_id}.{H}.SL_IDs.{algorithm}_l{side}.npy")
    np.save(fname, srchlight_train)

    srchlight_test = srchlight_test.reshape(-1, nvtx)
    fname = dataset(dataset2_id).outpath(f"{dataset2_id}.{H}.SL_IDs.{algorithm}_l{side}.npy")
    np.save(fname, srchlight_test)
