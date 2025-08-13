"""
This script performs joint embedding of geodesic distance matrices for individual 
subjects and group-average surfaces.

Steps:
    1. Parse the subject index and retrieve the corresponding subject ID.
    2. Load geodesic distance matrices and cortical surfaces for the subject and 
       group-average surfaces.
    3. Compute a combined geodesic distance matrix by averaging subject and group 
       distances, and adding Euclidean distances between vertices.
    4. For each combination of kernel and scale parameters:
        - Adjust the geodesic distance matrices using the specified kernel and scale.
        - Perform joint embedding with the adjusted matrices.
        - Save the resulting embeddings for the parameter combination.
    5. Save all embeddings in a compressed `.npz` archive for each hemisphere.

Parameters:
    <index>: Integer
        The index (1-based) of the subject in the subject list file specified in `directories.txt`. 

Dependencies:
    - `variograd_utils`: For dataset and subject handling, embedding, and kernel utilities.
    - `psutil`: For monitoring memory usage.
    - `numpy`: For numerical computations and file handling.

Inputs:
    - Geodesic distance matrices for left and right hemispheres:
        `<output_dir>/<subject>.<H>.gdist_triu.10k_fs_LR.npy`
    - Group-average cortical surfaces:
        `<group_dir>/T1w/fsaverage_LR10k/<subject>.<H>.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii`

Outputs:
    - Joint embeddings for all parameter combinations:
        `<output_dir>/<subject>.<H>.embeddings.npz`

Notes:
    - The script computes embeddings for a range of scales (50, 100, 150, 200) and kernels 
      (`cauchy`, `gauss`, `linear`, `None`).
    - The embedding results are stored as key-value pairs in the `.npz` file, where the 
      key corresponds to the parameter combination.
    - Ensure sufficient memory is available for processing large matrices.

"""


import sys
import psutil
import numpy as np
from variograd_utils.core_utils import dataset, subject, npz_update
from variograd_utils.embed_utils import JointEmbedding, kernel_affinity, pseudo_sqrt

dataset_id = str(sys.argv[1])
index = int(sys.argv[2])-1

data = dataset(dataset_id)
ID = data.subj_list[index]

n_components = 20
kernel = ["cauchy", "gauss", "linear"]
scale = np.array([10, 50, 100, 200], dtype="float32")
alignment = "sort_flip"
affinity = "precomputed"
je_method = "le"
hemi = ["L", "R"]

params = np.array(np.meshgrid(scale, kernel), dtype="object").T.reshape(-1, 2)

for h in hemi:
    subj = subject(ID, data.id)
    subj_points = subj.load_surf(h, 10, type="cortex_midthickness").darrays[0].data
    avg_points = data.load_surf(h, 10, type="cortex_midthickness").darrays[0].data

    R = data.load_gdist_matrix(h).astype("float32")
    R = (R + R.T) / 2
    M = subj.load_gdist_matrix(h).astype("float32")
    M = (M + M.T / 2)

    all_embeddings = {}
    for n, (s, k) in enumerate(params):
        key = f"JE_{k}{int(s)}"

        print(f"\n\nParameter combinaiton {n+1}/{len(params)}\tkey:", key, "\n")
        process = psutil.Process()

        Radj = R.copy() if k is None else kernel_affinity(R.copy(), kernel=k, scale=s)
        Madj = M.copy() if k is None else kernel_affinity(M.copy(), kernel=k, scale=s)
        Cadj = pseudo_sqrt(np.dot(Madj, Radj), n_components=100)

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
