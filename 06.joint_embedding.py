# Joint Embedding in parallel
import psutil
from variograd_utils import *
from nighres.shape import spectral_matrix_embedding
import numpy as np
from joblib import Parallel, delayed
import gc

n_components = 10
data = dataset()
msize = [500]
affinity = ["Gauss", "linear"]
scale = np.arange(50, 201, 50, dtype="float32")
hemi = ["L", "R"]

params = np.array(np.meshgrid(hemi, msize, scale, affinity), dtype="object").T.reshape(-1, 4)

def joint_embedding(id, h, m, s, a):

    print("\n\n\t\t\tSubject ID: ", id, "\n\n")    
    subj = subject(id)
    subj_points = subj.load_surf(h, 10, type="cortex_midthickness").darrays[0].data
    avg_points = data.load_surf(h, 10, type="cortex_midthickness").darrays[0].data
    C = np.sqrt(np.sum((subj_points - avg_points) ** 2, axis=1), dtype="float32")
    C = C + np.tile(C, [C.size, 1]).T
    R = data.load_gdist_matrix(h).astype("float32")
    M = subj.load_gdist_matrix(h).astype("float32")    
    C += (M + R) / 2

    embedding = spectral_matrix_embedding(M, 
                                            reference_matrix = R,
                                            correspondence_matrix=C,
                                            ref_correspondence_matrix=R,
                                            surface_mesh = getattr(subj, f"{h}_cortex_midthickness_10k_T1w"), 
                                            reference_mesh = getattr(data, f"{h}_cortex_midthickness_10k"), 
                                            dims=n_components, msize=m, scale=s, space=s, affinity=a, 
                                            rotate=True, save_data=False, overwrite=True)
    
    del subj_points, avg_points, C, R, M
    gc.collect()

    return embedding["result"]


for n, args in enumerate(params):
    process = psutil.Process()

    h, m, s, a  = args

    print(f"\n\n\t\t\tParameter combinaiton {n+1}/{len(params)}\n", h, m, s, a)

    n_vertices = vertex_info_10k[f"gray{h.lower()}"].size

    all_embeddings = Parallel(n_jobs=3)(delayed(joint_embedding)(id, h, m, s, a) for id in data.subj_list)

    filename = data.outpath(f'All.{h}.embeddings.npz')
    npz_update(filename, {f"JE_m{int(m)}_{a}_s{int(s)}": all_embeddings})

    print(f"Output saved in archive {filename} \nFile name: JE_m{int(m)}_{a}_s{int(s)} \n")

    print("Peak memory usage: ", process.memory_info().rss / 1e+9, "GB")
    gc.collect()
    print("Memory usage after garbage collection: ", process.memory_info().rss / 1e+9, "GB")
    print("\n\n")
    
    del all_embeddings