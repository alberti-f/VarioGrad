import sys
from itertools import product
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
n_components = 100
alphas = [0.5, 1]
diffusion_times = [0, 1]

print(f"Processing subject {ID}")

# Load and threshold group average FC matrix
R = np.load(data.outpath(f"{data.id}.REST_FC.10k_fs_LR.npy")
           ).astype("float32")[:, cortex][cortex, :]
R[R < np.percentile(R, threshold, axis=0)] = 0

# Load individual timeseries, compute FC, and threshold
M = nib.load(subject(subj.id).outpath(f"{subj.id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
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
                    alignment="rotation",
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
filename = subj.outpath(f'{ID}.FC_embeddings.npz')
npz_update(filename,  embedding_dict)
print(f"Subject embeddings saved in archive {filename} \n")

filename = subj.outpath(f'{ID}.FC_embeddings_refs.npz')
npz_update(filename,  reference_dict)
print(f"Reference embeddings saved in archive {filename} \n")