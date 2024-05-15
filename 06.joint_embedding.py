# Joint Embedding

from nighres.shape import spectral_matrix_embedding
from sklearn.metrics import pairwise_distances
from variograd_utils import *
import numpy as np
import gc

n_components = 10
data = dataset()
msize = [1500]
affinity = ["Gauss", "linear"]
scale = np.arange(50, 201, 50, dtype="float32")
hemi = ["L", "R"]



params = np.array(np.meshgrid(hemi, msize, scale, affinity), dtype="object").T.reshape(-1, 4)

def joint_embedding(params):
    id, h, m, s, a  = params

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
    return embedding["result"]


for n, args in enumerate(params):
    h, m, s, a  = args
    print("-" * 80, f"\n\t\t\tParameter combinaiton {n+1}/{len(params)}\n", h, m, s, a)

    n_vertices = vertex_info_10k[f"gray{h.lower()}"].size
    all_embeddings = np.zeros([data.N, n_vertices, n_components])

    for i in range(data.N):
        print(f"subject {i+1}/{data.N}")
        embedding = joint_embedding([data.subj_list[i], h, m, s, a])
        all_embeddings[i, :, :] += embedding

        embedding = None
        gc.collect()

        filename = data.outpath(f'All.{h}.embeddings.npz')
        npz_update(filename, {f"JE_m{int(m)}_{a}_s{int(s)}": all_embeddings})
 
