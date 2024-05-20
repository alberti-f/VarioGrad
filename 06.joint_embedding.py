# Joint Embedding in parallel

from variograd_utils import *
from nighres.shape import spectral_matrix_embedding
import numpy as np
import gc
from joblib import Parallel, delayed

n_components = 10
data = dataset()
msize = [500]
affinity = ["Gauss", "linear"]
scale = np.arange(50, 201, 50, dtype="float32")
hemi = ["L", "R"]

params = np.array(np.meshgrid(hemi, msize, scale, affinity), dtype="object").T.reshape(-1, 4)

def joint_embedding(id, h, m, s, a):

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


for n, args in enumerate(params[:1]):
    h, m, s, a  = args

    print(f"\n\n\t\t\tParameter combinaiton {n+1}/{len(params)}\n", h, m, s, a)

    n_vertices = vertex_info_10k[f"gray{h.lower()}"].size

    all_embeddings = Parallel(n_jobs=-1)(delayed(joint_embedding)(id, h, m, s, a) for id in data.subj_list[:1])

    filename = data.outpath(f'All.{h}.embeddings.npz')
    # npz_update(filename, {f"JE_m{int(m)}_{a}_s{int(s)}": all_embeddings})

    print(f"Output saved in archive {filename} \nFile name: JE_m{int(m)}_{a}_s{int(s)} \n")
