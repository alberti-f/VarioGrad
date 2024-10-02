import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nibabel as nib
from variograd_utils.core_utils import dataset, subject
from variograd_utils.brain_utils import vertex_info_10k
from variograd_utils.embed_utils import JointEmbedding


idx = int(sys.argv[1])-1
data = dataset()
ID = data.subj_list[idx]
subj = subject(ID)
cortex = np.hstack([vertex_info_10k.grayl, vertex_info_10k.grayr + vertex_info_10k.num_meshl])
threshold = 95
alpha = 0.5
diffusion_time = 1

print(f"Processing subject {ID}")

# Loand and threshold group average FC matrix
W = np.load(data.outpath(f"{data.id}.REST_FC.10k_fs_LR.npy")).astype("float32")[:, cortex][cortex, :]
W[W < 0] = 0
W[W < np.percentile(W, threshold, axis=0)] = 0

# Load individual timeseries and threshold
M = nib.load(subject(subj.id).outpath(f"{subj.id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")).get_fdata().astype("float32")[:, cortex]
M = np.corrcoef(M.T)
M[M < 0] = 0
M[M < np.percentile(M, threshold, axis=0)] = 0

if np.any(np.isnan(M)) or np.any(np.isnan(W)):
    raise ValueError(f"NaN values in FC matrices. Subject {ID} not processed.")

# Compute adjacency matrices
C = cosine_similarity(M.T, W.T)
W = cosine_similarity(W.T)
M = cosine_similarity(M.T)

print("Diffusion map embedding:")
print(f"\tFC threshold: {threshold}%")
print(f"\talpha: {alpha}")
print(f"\tdiffusion time: {diffusion_time}\n\n")


# Compute joint diffusion map embedding
kws = {"alpha": alpha, "diffusion_time": diffusion_time}
embedding = joint_embedding(M, W, C=C, n_components=10, method="diffusion", method_kws=kws, 
                            alignment="rotation", normalized=False, overwrite=True)



# Save embedding
np.save(subj.outpath(f"{subj.id}.REST_FC_embedding.npy"), embedding)

print(f"\n\nSubject {ID} embedding saved to {subj.outpath(f'{subj.id}.REST_FC_embedding.npy')}")
