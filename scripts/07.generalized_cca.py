"""
This script performs GCCA to generate group-level embeddings for geodesic distance 
matrices across subjects. It first computes individual embeddings using Singular 
Value Decomposition (SVD) for each subject and hemisphere. Once all subjects are 
processed, it concatenates individual embeddings and performs group-level SVD to
project individual embeddings onto a common space.

Parameters:
    <index>: Integer
        The index (1-based) of the subject in the subject list file specified in `directories.txt`. 

Steps:
    1. Parse the subject index and set fixed parameters.
    2. For each hemisphere (`L` and `R`):
        - Load and preprocess the geodesic distance matrix.
        - Compute SVD embeddings up to the maximum rank specified in `rank_svd`.
        - Save the individual embeddings (`U`, `S`, and `V`) in `.npz` format.
    3. After all subjects are processed:
        - Conctenate individual embeddings for each hemisphere.
        - Perform group-level SVD.
        - Project individual embeddings onto the group SVD space.
        - Save GCCA embeddings in `.npz` format for both hemispheres.
    4. Flip the sign of non-matching embeddings across subjects.
    5. Remove intermediate files (e.g., individual SVD embeddings).

Dependencies:
    - `variograd_utils`: For dataset and subject handling, file utilities, and embedding alignment.
    - `scipy`: For SVD computations and matrix operations.
    - `numpy`: For numerical computations and file handling.
    - `sklearn`: For normalization of projection matrices.

Outputs:
    - Individual SVD embeddings:
        `<output_dir>/<subject>.<H>.svd.npz`
    - Group GCCA embeddings:
        `<output_dir>/All.<H>.embeddings.npz`

Notes:
    - The script adapts mvlearn's GCCA class.
      (see https://github.com/mvlearn/mvlearn/blob/main/mvlearn/embed/gcca.py)
    - The script individual embeddings for a range of ranks (`rank_svd=[10, 50, 100]`) and 
      generates group embeddings of rank `rank_gcca=10`.
    - Individual SVD embeddings are removed after the group-level embeddings are generated.

"""


import sys
import os
from scipy import stats, linalg
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.preprocessing import normalize
from variograd_utils import dataset, subject, npz_update
from variograd_utils.brain_utils import vertex_info_10k


index = int(sys.argv[1])-1

rank_gcca = 10
rank_svd = [10, 50, 100]
hemi = ["L", "R"]


data = dataset()
ID = data.subj_list[index]
subj = subject(ID)
vinfo = vertex_info_10k

print("\n\nGeneralized Canonical Correlation Analysis\n\n")
print(f"Subject {ID} ({index+1}/{data.N})")

# Individual SVD
for h in hemi:
    X = subj.load_gdist_matrix(h)
    X = stats.zscore(X, axis=1, ddof=1)
    mu = np.mean(X, axis=0)
    X -= mu

    # Compute individual embeddings
    SVDs = {}
    r_svd = np.max(rank_svd)

    print("\t Computing individual embedding for hemisphere ", h)

    u, s, vt = svds(X, k=r_svd, which="LM", random_state=0)

    sorter = np.argsort(-s)
    v = vt.T
    v = v[:, sorter]
    ut = u.T
    u = ut.T[:, sorter]
    s = s[sorter]

    SVDs["SVD_U"] = u
    SVDs["SVD_S"] = s
    SVDs["SVD_V"] = v


    npz_update(subj.outpath(f'{subj.id}.{h}.svd.npz'), SVDs)

################# NOT ADAPTED TO PARALLELIZATION #################
# Group SVD and projections take place only if all subjects have been processed
params = np.array(np.meshgrid(rank_svd, hemi), dtype="object").T.reshape(-1, 2)
if index == data.N-1:

    # Generate projection matrices
    for r_svd, h in params:
        print(f"\n\nComputing GCCA with r_svd={r_svd} and r_gcca={rank_gcca} for hemisphere {h}")
        print("\tComputing group SVD")

        svd_paths = [subject(ID).outpath(f"{ID}.{h}.svd.npz") for ID in data.subj_list]
        Uall = np.hstack([np.load(path)["SVD_U"][:, :r_svd] for path in svd_paths])

        _, _, VV = svds(Uall, k=rank_gcca)
        VV = np.flip(VV.T, axis=1)
        VV = VV[:, : rank_gcca]

        print("\tProjecting individual embeddings onto group SVDs")
        GCCA = {f"GCCA_r{r_svd}": np.zeros([data.N, vinfo[f"gray{h.lower()}"].size, rank_gcca])}
        for ID in data.subj_list: 
            subj = subject(ID)

            idx_start = subj.idx * r_svd
            idx_end = idx_start + r_svd

            Vi = np.load(svd_paths[subj.idx])["SVD_V"]
            Si = np.load(svd_paths[subj.idx])["SVD_S"]
            VVi = normalize(VV[idx_start:idx_end, :], "l2", axis=0)


            A = np.sqrt(data.N - 1) * Vi[:, : r_svd]
            A = A @ (linalg.solve(
                np.diag(Si[: r_svd]), VVi
                ))

            X = subj.load_gdist_matrix(h)
            X = stats.zscore(X, axis=1, ddof=1)
            mu = np.mean(X, axis=0)
            X -= mu

            GCCA[f"GCCA_r{r_svd}"][subj.idx] = Xfit = X @ A

        npz_update(data.outpath(f'All.{h}.embeddings.npz'), GCCA)

    data.allign_embeddings("L", alg="GCCA")
    data.allign_embeddings("R", alg="GCCA")

    for ID in data.subj_list:
        os.remove(subject(ID).outpath(f"{ID}.L.svd.npz"))
        os.remove(subject(ID).outpath(f"{ID}.R.svd.npz"))

    print("\t\t Done")
