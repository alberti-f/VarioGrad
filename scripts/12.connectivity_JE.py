"""
Compute Joint Embedding of Functional Connectivity Matrices for Each Hemisphere.

This script computes joint diffusion map embeddings for individual and group-average 
functional connectivity (FC) matrices, separately for the left and right cortical hemispheres.

Parameters:
    <idx>: Integer
        The index of the subject in the subject list file.

Steps:
1. Preallocate Memory:
    - Allocate memory for storing embeddings and reference embeddings for both hemispheres.
    - Prepare parameter combinations of `alpha` and `diffusion_time`.

2. Load and Threshold FC Matrices:
    - Load the group-average FC matrix and threshold to the top n% values for each hemisphere.
    - Compute the subject's individual FC matrix from their timeseries, and threshold it.

3. Compute Correspondence Matrix:
    - Compute a cosine similarity matrix between the group and individual FC.

4. Compute Joint Embeddings:
    - Perform a joint diffusion map embedding for each hemisphere using the
      thresholded FC matrices and correspondence matrix.
    - Compute embeddings for all combinations of `alpha` and `diffusion_time`.

5. Save Results:
    - Save embeddings and their reference versions in `.npz` files.

Dependencies:
    - `variograd_utils`: For dataset and subject handling, and embedding utilities.
    - `numpy`: For numerical computations and matrix operations.
    - `nibabel`: For loading neuroimaging timeseries data.

Outputs:
    - Joint embeddings for both hemispheres:
        `<output_dir>/<subject>.FC_embeddings_flip_hemi_refs.npz`
    - Reference embeddings for both hemispheres:
        `<output_dir>/<subject>.FC_embeddings_flip_hemi_refs.npz`

Notes:
    - The `threshold` parameter retains the top percentile values in the FC matrices for sparsity.
    - Embeddings are computed for combinations of `alpha` and `diffusion_time`.

"""

import sys
from itertools import product
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nibabel as nib
from variograd_utils import dataset, subject, npz_update
from variograd_utils.brain_utils import vertex_info_10k as vinfo
from variograd_utils.embed_utils import JointEmbedding

dataset_id = sys.argv[1]
idx = int(sys.argv[2])-1 
data = dataset(dataset_id)
ID = data.subj_list[idx]
subj = subject(ID, data.id)
cortex = np.hstack([vinfo.grayl, vinfo.grayr + vinfo.num_meshl])
threshold = 95
n_components = 100
alphas = [0.5, 1]
diffusion_times = [0, 1]

print(f"Processing subject {ID}")

# Preallocate memory
kwarg_dict = {f"a{str(a).replace('.', '')}_t{t}": {"alpha": a, "diffusion_time": t} 
              for a, t in product(alphas, diffusion_times)}
n_vtx_LR = vinfo.grayl.size + vinfo.grayr.size
embedding_dict = {k: np.zeros([n_vtx_LR, n_components])
                  for k in kwarg_dict.keys()}
reference_dict = {k: np.zeros([n_vtx_LR, n_components])
                  for k in kwarg_dict.keys()}

for h in ["L", "R"]:
    cortex = vinfo.grayl if h=="L" else vinfo.grayr + vinfo.num_meshl
    cortex_slice = slice(0, vinfo.grayl.size) if h == "L" else slice(vinfo.grayl.size, None)

    # Load and threshold group average FC matrix
    R = np.load(dataset("train").outpath(f"train.REST_FC.10k_fs_LR.npy")
               ).astype("float32")[:, cortex][cortex, :]
    R = (R + R.T) / 2
    R[R < np.percentile(R, threshold, axis=0, keepdims=True)] = 0
    R[R < 0] = 0
    
    # Load individual timeseries, compute FC, and threshold
    M = nib.load(subject(subj.id, data.id).outpath(
        f"{subj.id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
        ).get_fdata().astype("float32")[:, cortex]
    M = np.corrcoef(M.T)
    M = (M + M.T) / 2
    M[M < np.percentile(M, threshold, axis=0, keepdims=True)] = 0
    M[M < 0] = 0

    # Compute correspondance matrix
    C = cosine_similarity(M.T, R.T)

    # Compute joint diffusion map embedding
    print("Diffusion map embedding:")
    print(f"\tFC threshold: {threshold}%")
    print(f"\talphas: {alphas}")
    print(f"\tdiffusion times: {diffusion_times}\n\n")

    je = JointEmbedding(method="dme",
                        n_components=n_components,
                        alignment="sort_flip",
                        random_state=0,
                        copy=True)    

    # Compute gradients with all combinations of alpha and time
    for key, kwargs in kwarg_dict.items():

        embedding, reference = je.fit_transform(M.T, R.T, C=C,
                                                affinity="cosine",
                                                method_kwargs=kwargs)
       
        embedding_dict[key][cortex_slice] = embedding
        reference_dict[key][cortex_slice] = reference
        print(f"\t\N{GREEK SMALL LETTER ALPHA}={kwargs['alpha']} "
              + f"t={kwargs['diffusion_time']} :\t done")

# Save output
filename = subj.outpath(f'{ID}.FC_embeddings.npz')
npz_update(filename,  embedding_dict)
print(f"Subject embeddings saved in archive {filename} \n")

filename = subj.outpath(f'{ID}.FC_embeddings_refs.npz')
npz_update(filename,  reference_dict)
print(f"Reference embeddings saved in archive {filename} \n")
