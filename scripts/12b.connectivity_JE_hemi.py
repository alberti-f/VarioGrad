import sys
from itertools import product
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nibabel as nib
from variograd_utils.core_utils import dataset, subject, npz_update, vector_wise_corr
from variograd_utils.brain_utils import vertex_info_10k as vinfo
from variograd_utils.brain_utils import left_cortex_data_10k, right_cortex_data_10k
from variograd_utils.embed_utils import JointEmbedding


idx = int(sys.argv[1])-1 
data = dataset()
ID = data.subj_list[idx]
subj = subject(ID)
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
    
    # Compute gradients with all combinations of alpha and time
    for key, kwargs in kwarg_dict.items():

        embedding, reference = je.fit_transform(M.T, R.T, C=C, affinity="cosine", method_kwargs=kwargs)
       
        embedding_dict[key][cortex_slice], reference_dict[key][cortex_slice] = (embedding, reference)
        print(f"\t\N{GREEK SMALL LETTER ALPHA}={kwargs['alpha']} t={kwargs['diffusion_time']} :\t done")

# Save output
filename = subj.outpath(f'{ID}.FC_embeddings_hemi.npz')
npz_update(filename,  embedding_dict)
print(f"Subject embeddings saved in archive {filename} \n")

filename = subj.outpath(f'{ID}.FC_embeddings_refs_hemi.npz')
npz_update(filename,  reference_dict)
print(f"Reference embeddings saved in archive {filename} \n")