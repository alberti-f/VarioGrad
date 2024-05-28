# joint embedding simplified


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from mapalign.embed import compute_diffusion_map
import psutil
from variograd_utils import *
import numpy as np
import gc
from joblib import Parallel, delayed
import sys



def joint_embedding(M, R, C=None, n_components=10, kernel=None, affinity="cosine", scale=50, overwrite=False, embedding="diffusion", embedding_kws=None):

    """
    Perform joint embedding of two distance matrices using spectral embedding
    
    Parameters
    ----------
    M : array-like, shape (n_samples, n_samples)    
        The first distance matrix to be embedded.
        
    R : array-like, shape (n_samples, n_samples)
        The reference matrix to use for the embedding.
        
    C : array-like, shape (n_samples, n_samples), default=None
        The correspondence matrix to use for the embedding. If None, the correspondence matrix
        will be computed as the cosine similarity between M and R.
    
    n_components : int, default=10
        The number of components to use for the embedding.
        
    kernel : str, default=None
        The kernel to use for the embedding. If None, the kernel will be computed as the cosine similarity.
        
    affinity : str, default='cosine'
        The affinity metric to use for converting M, R, and C to similarity matrices.

    scale : float, default=50
        The scaling factor to use for the kernel.

    overwrite : bool, default=False
        Whether to overwrite the input matrices or not.

    Returns
    -------
    embedding : array-like, shape (n_samples, n_components)
        The joint embedding of the two distance matrices. 
    """

    N = M.shape[0]
    if M.shape != R.shape:
        raise ValueError("The input matrices must have the same shape")

    if not overwrite:
        M = M.copy()
        R = R.copy()
        if C is not None:
            C = C.copy()

    # Apply kernel to matrices
    if kernel == 'cauchy':
        M = 1.0 / (1.0 + (M ** 2) / (scale ** 2))
        R = 1.0 / (1.0 + (R ** 2) / (scale ** 2))
    elif kernel == 'gauss':
        M = np.exp(-0.5 * (M ** 2) / (scale ** 2))
        R = np.exp(-0.5 * (R ** 2) / (scale ** 2))
    elif kernel == 'cosine':
        M = max(1.0 - M / scale, 0.0)
        R = max(1.0 - R / scale, 0.0)
    elif kernel == 'precomputed' or kernel is None:
        pass
    else:
        raise ValueError("Unknown kernel type")



    # Convert to affinity and compute correspondence matrix
    M = cosine_similarity(M)
    R = cosine_similarity(R)
    if C is None:
        C = cosine_similarity(R, M)



    # Build the joint affinity matrix
    A = np.vstack([np.hstack([R, C]), 
                   np.hstack([C.T, M])])


    
    if embedding == "diffusion":

        if embedding_kws is None:
            embedding_kws = {"n_components": n_components, "overwrite": overwrite}

        # # Compute reference embedding
        # embedding_kws["return_result"] = False
        # ref_embedding = compute_diffusion_map(R, **embedding_kws)

        # Compute the joint embedding
        embedding_kws["return_result"] = False
        joint_embedding = compute_diffusion_map(A, **embedding_kws, )
        # lambdas = result["lambdas"]
        
    else:
        raise ValueError("Unknown embedding method")

    
    # Not working as expected
    # # Rotate the joint embedding
    # ref_norm = normalize(ref_embedding, axis=0)
    # joint_norm = normalize(joint_embedding[:N], axis=0)
    # rotation_mat = np.dot(joint_norm.T, ref_norm)

    # ref_out = np.dot(joint_embedding[:N], rotation_mat)
    # subj_out = np.dot(joint_embedding[N:], rotation_mat)

    # i = 0
    # print(np.corrcoef(ref_embedding[:, i], joint_embedding[N:, i])[0, 1], 
    #       np.corrcoef(ref_embedding[:, i], subj_out[:, i])[0, 1],
    #       np.corrcoef(joint_embedding[:N, i], subj_out[:, i])[0, 1], 
    #       np.corrcoef(ref_embedding[:, i], ref_out[:, i])[0, 1])

    print(joint_embedding.shape)
    # Return the embedding
    return joint_embedding[N:], joint_embedding[:N]

def embed_mesh(id, s, k, a, t, h):
    subj = subject(id)
    R = data.load_gdist_matrix(h).astype("float32")
    M = subj.load_gdist_matrix(h).astype("float32")    

    embedding, _ = joint_embedding(M, R, n_components=n_components, kernel=k, 
                                scale=s, overwrite=True, 
                                embedding_kws={"n_components": n_components, "overwrite": True,
                                               "alpha": a, "diffusion_time": t})
    
    print(embedding.shape)
    return embedding

#------------------------------------------------------------


index = int(sys.argv[1])-1

data = dataset()
id = data.subj_list[index]

n_components = 10
data = dataset()
kernel = ["cauchy", "gauss", "linear"]
scale = np.arange(50, 201, 50, dtype="float32")
alpha = [0.5]
time = [1, 2]
hemi = ["L", "R"]

params = np.array(np.meshgrid(scale, hemi, alpha, time, kernel), dtype="object").T.reshape(-1, 5)


all_embeddings = {}
for n, args in enumerate(params):
    s, h, a, t, k  = args
    key = f"JDE_{k}{int(s)}_a{str(0.5).replace('.', '')}_t{t}"

    print(f"\n\nParameter combinaiton {n+1}/{len(params)}\n", "\t\tkey:", key, "\n",)
    
    n_vertices = vertex_info_10k[f"gray{h.lower()}"].size

    # all_embeddings = Parallel(n_jobs=-1)(delayed(embed_mesh)(id, s, k, a, t, h) for id in data.subj_list)
    all_embeddings[key] = embed_mesh(id, s, k, a, t, h) 

filename = data.outpath(f'{id}.{h}.embeddings.npz')
npz_update(filename,  all_embeddings)

print(f"Output saved in archive {filename} \n")

# data.allign_embeddings(alg="JDE")
