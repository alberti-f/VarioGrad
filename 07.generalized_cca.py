# GCCA in parallel

# Individual SVD

from variograd_utils import *
import numpy as np
from scipy import stats, linalg
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import os, sys

index = int(sys.argv[1])-1

rank_gcca = 10
rank_svd = [10, 50, 100]
hemi = ["L", "R"]


data = dataset()
id = data.subj_list[index]
subj = subject(id)
vinfo = vertex_info_10k

print("\n\nGeneralized Canonical Correlation Analysis\n\n")
print(f"Subject {id} ({index+1}/{data.N})")

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

    SVDs[f"SVD_U"] = u
    SVDs[f"SVD_S"] = s
    SVDs[f"SVD_V"] = v


    npz_update(subj.outpath(f'{subj.id}.{h}.svd.npz'), SVDs)

################# NOT ADAPTED TO PARALLELIZATION #################
# Group SVD and projections take place only if all subjects have been processed
params = np.array(np.meshgrid(rank_svd, hemi), dtype="object").T.reshape(-1, 2)
if index == data.N-1:

    # Generate projection matrices
    for r_svd, h in params:
        print(f"\n\nComputing GCCA with r_svd={r_svd} and r_gcca={rank_gcca} for hemisphere {h}")
        print("\tComputing group SVD")

        svd_paths = [subject(id).outpath(f"{id}.{h}.svd.npz") for id in data.subj_list]
        Uall = np.hstack([np.load(path)[f"SVD_U"][:, :r_svd] for path in svd_paths])

        _, _, VV = svds(Uall, k=rank_gcca)
        VV = np.flip(VV.T, axis=1)
        VV = VV[:, : rank_gcca]

        print("\tProjecting individual embeddings onto group SVDs")
        GCCA = {f"GCCA_r{r_svd}": np.zeros([data.N, vinfo[f"gray{h.lower()}"].size, rank_gcca])}
        for id in data.subj_list: 
            subj = subject(id)

            idx_start = subj.idx * r_svd
            idx_end = idx_start + r_svd

            Vi = np.load(svd_paths[subj.idx])[f"SVD_V"]
            Si = np.load(svd_paths[subj.idx])[f"SVD_S"]
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

    print("\t\t Done")