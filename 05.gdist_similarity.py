# inter-subject similarity of vertex geodesic distance profile

import numpy as np
from variograd_utils import *
from itertools import combinations
from sklearn.metrics.pairwise import euclidean_distances
from joblib import Parallel, delayed

data =  dataset()
data.subj_list = data.subj_list
n_subj = data.N


subj_pairs = list(combinations(range(n_subj), 2))
subj_pairs = data.subj_list[subj_pairs]

np.print(f"\n\nComputing the inter-subject similarity of vertex geodesic 
         distance profile for {len(subj_pairs)} subject pairs \n")


gdist_L2_l = Parallel(n_jobs=-1)(delayed(np.diag)(euclidean_distances(subject(i).load_gdist_matrix("L")),
                                                  euclidean_distances(subject(j).load_gdist_matrix("L"))) 
                                                  for i, j in subj_pairs)
filename = data.outpath("AllToAll.L.gdist_L2.npy")
np.save(filename, np.asarray(gdist_L2_l).T)
print(f"Left hemisphere output saved at {filename} \n", 
      f"Matrix size: {np.asarray(gdist_L2_l).T.shape} \n")


gdist_L2_r = Parallel(n_jobs=-1)(delayed(np.diag)(euclidean_distances(subject(i).load_gdist_matrix("R")),
                                                  euclidean_distances(subject(j).load_gdist_matrix("R")))
                                                  for i, j in subj_pairs)
filename = data.outpath("AllToAll.R.gdist_L2.npy")
np.save(filename, np.asarray(gdist_L2_r).T)
print(f"Right hemisphere output saved at {filename} \n", 
      f"Matrix size: {np.asarray(gdist_L2_r).T.shape} \n")
