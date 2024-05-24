# GCCA in parallel

# Individual SVD

from variograd_utils import *
import numpy as np
from scipy import stats, linalg
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed


def subject_svd(id, rank):
    X = subject(id).load_gdist_matrix(h)
    X = stats.zscore(X, axis=1, ddof=1)
    mu = np.mean(X, axis=0)
    X -= mu

    u, s, vt = svds(X, k=rank, which="LM", random_state=0)
    sorter = np.argsort(-s)
    v = vt.T
    v = v[:, sorter]
    ut = u.T
    u = ut.T[:, sorter]
    
    return u, s, v




data = dataset()
rank_gcca = 10
rank_svd = [10, 50, 100]
hemi = ["L", "R"]

params = np.array(np.meshgrid(hemi, rank_gcca, rank_svd), dtype="object").T.reshape(-1, 3)


for h, r_gcca, r_svd in params[:1]:
    print("\n\nGeneralized Canonical Correlation Analysis\n\n")


    # Compute individual embeddings
    print("\t Computing individual embeddings")

    USV = Parallel(n_jobs=-1)(delayed(subject_svd)(id, r_svd) for id in data.subj_list)
    Uall = [usv[0] for usv in USV]
    Sall = [usv[1] for usv in USV]
    Vall = [usv[2] for usv in USV]
    

    npz_update(data.outpath(f'All.{h}.embeddings.npz'), {f"SVD_r{r_svd}_U": np.array(Uall)})
    npz_update(data.outpath(f'All.{h}.embeddings.npz'), {f"SVD_r{r_svd}_S": np.array(Sall)})
    npz_update(data.outpath(f'All.{h}.embeddings.npz'), {f"SVD_r{r_svd}_V": np.array(Vall)})


    # Generate projection matrices
    print("\t Generating projection matrices")

    Uall = data.load_embeddings(h, "SVD")[f"SVD_r{r_gcca}_U"]
    Uall =  Uall.transpose(1, 0, 2).reshape(Uall.shape[1], -1)

    _, _, VV = svds(Uall, k=r_gcca)
    VV = np.flip(VV.T, axis=1)
    VV = VV[:, : r_gcca]


    projection_mats = []
    idx_end = 0
    for i in range(data.N):
        idx_start = idx_end
        idx_end = idx_start + r_svd
        VVi = normalize(VV[idx_start:idx_end, :], "l2", axis=0)

        A = np.sqrt(data.N - 1) * Vall[i][:, : r_svd]
        A = A @ (linalg.solve(
            np.diag(Sall[i][: r_svd]), VVi
            ))
        projection_mats.append(A)

    np.save(data.outpath(f"All.{h}.GCCA_proj_mats.npy"), projection_mats)


    # Canonical projections
    print("\t Projecting individual views")
    vinfo = vertex_info_10k
    GCCA = {f"GCCA_r{r_svd}": np.zeros([data.N, vinfo[f"gray{h.lower()}"].size, r_gcca])}

    for id in data.subj_list:
        subj = subject(id)

        X = subj.load_gdist_matrix(h)
        X = stats.zscore(X, axis=1, ddof=1)
        mu = np.mean(X, axis=0)
        X -= mu

        Xfit = X @ projection_mats[subj.idx]
        GCCA[f"GCCA_r{r_svd}"][subj.idx] = Xfit

    npz_update(data.outpath(f'All.{h}.embeddings.npz'), GCCA)

    print("\t\t Done")