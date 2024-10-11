# Joint Embedding of geodesic distances

import sys
import psutil
import numpy as np
from variograd_utils.core_utils import dataset, subject, npz_update
from variograd_utils.embed_utils import JointEmbedding, kernelize

index = int(sys.argv[1])-1

data = dataset()
ID = data.subj_list[index]

n_components = 20
data = dataset()
kernel = ["cauchy", "gauss", "linear", None]
scale = np.arange(50, 201, 50, dtype="float32")
alignment = "rotation"
affinity = "precomputed"
je_method = "dme"
hemi = ["L", "R"]

params = np.array(np.meshgrid(scale, kernel), dtype="object").T.reshape(-1, 2)

for h in hemi:
    subj = subject(ID)
    subj_points = subj.load_surf(h, 10, type="cortex_midthickness").darrays[0].data
    avg_points = data.load_surf(h, 10, type="cortex_midthickness").darrays[0].data

    C = np.sqrt(np.sum((subj_points - avg_points) ** 2, axis=1), dtype="float32")
    C = C + np.tile(C, [C.size, 1]).T
    R = data.load_gdist_matrix(h).astype("float32")
    M = subj.load_gdist_matrix(h).astype("float32")    
    C += (M + R) / 2

    all_embeddings = {}
    for n, (s, k) in enumerate(params):
        key = f"JE_{k}{int(s)}"

        print(f"\n\nParameter combinaiton {n+1}/{len(params)}\tkey:", key, "\n")
        process = psutil.Process()

        Radj = R.copy() if k is None else kernelize(R.copy(), kernel=k, scale=s)
        Madj = M.copy() if k is None else kernelize(M.copy(), kernel=k, scale=s)
        Cadj = C.copy() if k is None else kernelize(C.copy(), kernel=k, scale=s)

        je = JointEmbedding(method=je_method,
                            n_components=n_components,
                            alignment=alignment,
                            random_state=0,
                            copy=True)
        all_embeddings[key], _ = je.fit_transform(Madj, Radj, C=Cadj, affinity=affinity)

        print(f"\n\tMemory used: {process.memory_info().rss / (1024 ** 3):.2f} GB\n")
        del process

    filename = subj.outpath(f'{ID}.{h}.embeddings.npz')
    npz_update(filename,  all_embeddings)

    print(f"Output saved in archive {filename} \n")
