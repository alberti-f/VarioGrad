from variograd_utils import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nibabel as nib
import sys

idx = int(sys.argv[1])-1
data = dataset()
id = data.subj_list[idx]
subj = subject(id)
cortex = np.hstack([vertex_info_10k.grayl, vertex_info_10k.grayr + vertex_info_10k.num_meshl])
threshold = 95
alpha = 0.5
diffusion_time = 0

print(f"Processing subject {id}")

# Loand and threshold group average FC matrix
W = np.load(data.outpath(f"{data.id}.REST_FC.10k_fs_LR.npy")).astype("float32")[:, cortex][cortex, :]
W[W < 0] = 0
W[W < np.percentile(W, threshold, axis=0)] = 0

# Load individual timeseries and threshold
M = nib.load(subject(subj.id).outpath(f"{subj.id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")).get_fdata().astype("float32")[:, cortex]
print(np.any(np.isnan(M)))
M = np.corrcoef(M.T)
M[M < 0] = 0
M[M < np.percentile(M, threshold, axis=0)] = 0

if np.any(np.isnan(M)) or np.any(np.isnan(W)):
    raise ValueError(f"NaN values in FC matrices. Subject {id} not processed.")

# Compute adjacency matrices
C = cosine_similarity(M.T, W.T)
W = cosine_similarity(W.T)
M = cosine_similarity(M.T)

print("Diffusion map embedding:\n")
print(f"\tFC threshold: {threshold}%")
print(f"\talpha: {alpha}")
print(f"\tdiffusion time: {diffusion_time}\n\n")


# Compute joint diffusion map embedding
kws = {"alpha": alpha, "diffusion_time": diffusion_time}
embedding = joint_embedding(M, W, C=C, n_components=10, method="diffusion", method_kws=kws, 
                            rotate=False, normalize=False, overwrite=True)


# Save embedding
np.save(subj.outpath(f"{subj.id}.REST_FC_embedding.npy"), embedding)

print(f"\n\nSubject {id} embedding saved to {subj.outpath(f'{subj.id}.REST_FC_embedding.npy')}")