import sys
import psutil#############################################################################
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nibabel as nib
from variograd_utils.core_utils import dataset, subject
from variograd_utils.brain_utils import vertex_info_10k
from variograd_utils.embed_utils import JointEmbedding


idx = 1 #int(sys.argv[1])-1
data = dataset()
ID = data.subj_list[idx]
subj = subject(ID)
cortex = np.hstack([vertex_info_10k.grayl, vertex_info_10k.grayr + vertex_info_10k.num_meshl])
threshold = 95
n_components = 20
alpha = 0.5
diffusion_time = 0

print(f"Processing subject {ID}")

process = psutil.Process()############################################################

# Load and threshold group average FC matrix
R = np.load(data.outpath(f"{data.id}.REST_FC.10k_fs_LR.npy")
           ).astype("float32")[:, cortex][cortex, :]
R[R < 0] = 0
R[R < np.percentile(R, threshold, axis=0)] = 0
print(f"\n\tMemory used: {process.memory_info().rss / (1024 ** 3):.2f} GB\n")#############################################

# Load individual timeseries, compute FC, and threshold
M = nib.load(subject(subj.id).outpath(f"{subj.id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
            ).get_fdata().astype("float32")[:, cortex]
M = np.corrcoef(M.T)
M[M < 0] = 0
M[M < np.percentile(M, threshold, axis=0)] = 0
print(f"\n\tMemory used: {process.memory_info().rss / (1024 ** 3):.2f} GB\n")#############################################


# Check for NaNs in the matrices
if np.any(np.isnan(M)) or np.any(np.isnan(R)):
    raise ValueError(f"NaN values in FC matrices. Subject {ID} not processed.")


# # Compute adjacency matrices
C = cosine_similarity(M.T, R.T)
# W = cosine_similarity(W.T)
# M = cosine_similarity(M.T)

print("Diffusion map embedding:")
print(f"\tFC threshold: {threshold}%")
print(f"\talpha: {alpha}")
print(f"\tdiffusion time: {diffusion_time}\n\n")


# Compute joint diffusion map embedding
kwargs = {"alpha": alpha, "diffusion_time": diffusion_time}
je = JointEmbedding(method="dme",
                    n_components=n_components,
                    alignment="rotation",
                    random_state=0,
                    copy=True)
print(f"\n\tMemory used: {process.memory_info().rss / (1024 ** 3):.2f} GB\n")#############################################

embedding, _ = je.fit_transform(M, R, C=C,
                                affinity="cosine",
                                method_kwargs=kwargs)
print(f"\n\tMemory used: {process.memory_info().rss / (1024 ** 3):.2f} GB\n")#############################################

# joint_embedding(M, W, C=C, n_components=n_components, method="diffusion", method_kws=kws, 
#                             alignment="rotation", normalized=False, overwrite=True)



#Save embedding
np.save(subj.outpath(f"{subj.id}.REST_FC_embedding.npy"), embedding)

print(f"\n\nSubject {ID} embedding saved to {subj.outpath(f'{subj.id}.REST_FC_embedding.npy')}")
