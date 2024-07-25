# Joint Embedding in parallel
from variograd_utils import *
import numpy as np
import sys
import psutil

index = int(sys.argv[1])-1

data = dataset()
id = data.subj_list[index]

n_components = 10
data = dataset()
kernel = ["cauchy", "gauss", "linear", None]
scale = np.arange(50, 201, 50, dtype="float32")
hemi = ["L", "R"]

params = np.array(np.meshgrid(scale, kernel), dtype="object").T.reshape(-1, 2)

for h in hemi:
    subj = subject(id)
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

        print(f"\n\nParameter combinaiton {n+1}/{len(params)}\tkey:", key, "\n",)
        process = psutil.Process()

        all_embeddings[key] = joint_embedding(M, R, C=C, n_components=n_components, kernel=k, scale=s, alignment="rotation")

        print(f"\n\tMemory used: {process.memory_info().rss / (1024 ** 3):.2f} GB\n")
        del process

    filename = subj.outpath(f'{id}.{h}.embeddings.npz')
    npz_update(filename,  all_embeddings)

    print(f"Output saved in archive {filename} \n")