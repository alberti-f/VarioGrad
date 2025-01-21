"""
Compute Joint Embedding of Functional Connectivity Matrices.

This script computes a joint diffusion map embedding of individual and group-average 
functional connectivity (FC) matrices for a single subject. It aligns the embeddings 
across subjects using cosine similarity and saves the results as `.npz` files.

Parameters:
    <idx>: Integer
        Index of the subject in the subject list file.

Steps:
1. **Load and Threshold FC Matrices**:
    - Load the group-average FC matrix and threshold it by retaining the top n% values.
    - Compute the individual subject's FC matrix from their timeseries, and threshold it.

2. **Compute Correspondence Matrix**:
    - Calculate a cosine similarity matrix between the thresholded group and individual FC matrices.

3. **Compute Joint Embeddings**:
    - Perform a joint diffusion map embedding using the individual and group FC matrices,
      along with the correspondence matrix.
    - Use combinations of `alpha` and `diffusion_time` to compute multiple embeddings.

4. **Save Results**:
    - Save the computed embeddings and their reference versions to `.npz` files.

Outputs:
    - Joint embeddings:
        `<output_dir>/<subject>.FC_embeddings_flip.npz`
    - Reference embeddings:
        `<output_dir>/<subject>.FC_embeddings_flip_refs.npz`

Notes:
    - The `threshold` parameter retains the top percentile values in the FC matrices for sparsity.
    - Embeddings are computed for all combinations of `alpha` and `diffusion_time`.

"""


import sys
from itertools import product
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nibabel as nib
from variograd_utils.core_utils import dataset, subject, npz_update
from variograd_utils.brain_utils import vertex_info_10k
from variograd_utils.embed_utils import JointEmbedding

dataset_id = sys.argv[1]
idx = int(sys.argv[2])-1

data = dataset(dataset_id)
ID = data.subj_list[idx]
subj = subject(ID, data.id)
cortex = np.hstack([vertex_info_10k.grayl, vertex_info_10k.grayr + vertex_info_10k.num_meshl])
threshold = 95
n_components = 100
alphas = [0.5, 1]
diffusion_times = [0, 1]

print(f"Processing subject {ID}")

# Load and threshold group average FC matrix
R = np.load(data.outpath(f"{data.id}.REST_FC.10k_fs_LR.npy")
           ).astype("float32")[:, cortex][cortex, :]
R[R < np.percentile(R, threshold, axis=0)] = 0

# Load individual timeseries, compute FC, and threshold
M = nib.load(subj.outpath(f"{subj.id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
            ).get_fdata().astype("float32")[:, cortex]
M = np.corrcoef(M.T)
M[M < np.percentile(M, threshold, axis=0)] = 0

# Compute correspondance matrix
C = cosine_similarity(M.T, R.T)

# Compute joint diffusion map embedding
print("Diffusion map embedding:")
print(f"\tFC threshold: {threshold}%")
print(f"\talphas: {alphas}")
print(f"\tdiffusion times: {diffusion_times}\n\n")

je = JointEmbedding(method="dme",
                    n_components=n_components,
                    alignment="sign_flip",
                    random_state=0,
                    copy=True)

# Preallocate memory
kwarg_dict = {f"a{str(a).replace('.', '')}_t{t}": {"alpha": a, "diffusion_time": t}
              for a, t in product(alphas, diffusion_times)}

embedding_dict = {k: np.zeros([M.shape[0], n_components])
                  for k in kwarg_dict.keys()}

reference_dict = {k: np.zeros([M.shape[0], n_components])
                  for k in kwarg_dict.keys()}

# Compute gradients with all combinations of alpha and time
for key, kwargs in kwarg_dict.items():

    embedding_dict[key], reference_dict[key] = je.fit_transform(M.T, R.T, C=C,
                                                                 affinity="cosine",
                                                                 method_kwargs=kwargs)
    print(f"\t\N{GREEK SMALL LETTER ALPHA}={kwargs['alpha']} t={kwargs['diffusion_time']} :\t done")

# Save outut
filename = subj.outpath(f'{ID}.FC_embeddings_flip.npz')
npz_update(filename,  embedding_dict)
print(f"Subject embeddings saved in archive {filename} \n")

filename = subj.outpath(f'{ID}.FC_embeddings_flip_refs.npz')
npz_update(filename,  reference_dict)
print(f"Reference embeddings saved in archive {filename} \n")
