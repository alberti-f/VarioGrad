# inter-subject similarity of vertex geodesic distance profile

import numpy as np
from variograd_utils import *
from itertools import combinations
from sklearn.metrics.pairwise import euclidean_distances
from os.path import exists
import sys
# import portalocker

# ###############################################
# def edit_npy_file_row(filename, row_index, new_data):
#     with open(filename, 'r+b') as f:
#         # Lock the file for writing
#         portalocker.lock(f, portalocker.LOCK_EX)

#         try:
#             # Memory-map the .npy file
#             mmapped_array = np.load(f, mmap_mode='r+')

#             # Modify the specific row
#             mmapped_array[row_index] = new_data

#             # Ensure changes are written (optional, usually handled by the OS)
#             mmapped_array.flush()

#         finally:
#             # Unlock the file
#             portalocker.unlock(f)
# ###############################################




index = int(sys.argv[1]) - 1

data =  dataset()

if not exists(data.outpath("subject_pairs.txt")):
    subj_pairs = list(combinations(range(data.N), 2))
    np.savetxt(data.outpath("subject_pairs.txt"), subj_pairs, fmt="%d")


subj_pairs = np.loadtxt(data.outpath("subject_pairs.txt")).astype(int)
n_pairs = len(list(subj_pairs))


if not exists(data.outpath("AllToAll.L.gdist_L2.npy")):
    gdist_L2_l = np.zeros([vertex_info_10k.grayl.size, n_pairs])
    np.save(data.outpath("AllToAll.L.gdist_L2.npy"), gdist_L2_l)

if not exists(data.outpath("AllToAll.R.gdist_L2.npy")):
    gdist_L2_r = np.zeros([vertex_info_10k.grayr.size, n_pairs])
    np.save(data.outpath("AllToAll.R.gdist_L2.npy"), gdist_L2_r)




gdist_L2_l = np.load(data.outpath("AllToAll.L.gdist_L2.npy"), mmap_mode='r+')
gdist_L2_r = np.load(data.outpath("AllToAll.R.gdist_L2.npy"), mmap_mode='r+')


i, j = subj_pairs[index]

# Left
Di = subject(data.subj_list[i]).load_gdist_triu("L")
Dj = subject(data.subj_list[j]).load_gdist_triu("L")
n_vtx = vertex_info_10k.grayl.size 
for vtx in range(n_vtx):
    L2 = euclidean_distances(row_from_triu(vtx, k=1, triu=Di, include_diag=True).reshape(1,-1), 
                                row_from_triu(vtx, k=1, triu=Dj, include_diag=True).reshape(1,-1)
                                ).squeeze()
    gdist_L2_l[vtx, index] = L2
    gdist_L2_l.flush()


# Right
print
Di = subject(data.subj_list[i]).load_gdist_triu("R")
Dj = subject(data.subj_list[j]).load_gdist_triu("R")
n_vtx = vertex_info_10k.grayr.size
for vtx in range(n_vtx):
    L2 = euclidean_distances(row_from_triu(vtx, k=1, triu=Di, include_diag=True).reshape(1,-1), 
                                row_from_triu(vtx, k=1, triu=Dj, include_diag=True).reshape(1,-1)
                                ).squeeze()
    gdist_L2_r[vtx, index] = L2
    gdist_L2_r.flush()

print(f"Completed comparison {index+1} of {n_pairs}")

