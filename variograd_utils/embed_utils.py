# # embed_utils.py

# from scipy.stats import rankdata
# from scipy.sparse import csr_matrix, linalg
# import numpy as np


# def kneighbors_graph(M, k, axis=1, mode="distance"):

#     if mode not in ["distance", "connectivity"]:
#         raise ValueError("Please specify a valid mode batween: 'distance', 'connectivity'")

#     M = M.copy()
#     ranks = rankdata(M, axis=axis, method="ordinal")
#     ranks[M==0] = 0
#     ranks = rankdata(ranks, axis=axis, method="dense") > k+1
#     M[ranks] = 0

#     if mode=="connectivity":
#         M[M!=0] = 1

#     return M


# def laplacian(M, k, norm=None, mode="distance", sparse=True):

#     A = kneighbors_graph(M, k, mode=mode)
#     D = np.diag(A.sum(axis=1))

#     if norm is None:
#         L = D - A

#     elif norm == "sym":
#         D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(D, axis=1)))
#         L = np.identity(A.shape[0]) - np.dot(np.dot(D_sqrt_inv, A), D_sqrt_inv)
    
#     elif norm == "rw":
#         D_inv = np.linalg.inv(D)
#         L = np.identity(D.shape[0]) - np.dot(D_inv, A)
    
#     if sparse:
#         return csr_matrix(L)
    
#     return A, L


# def laplacian_eigenmaps(M, n_components=None, n_neighbors=None, norm=None, mode="distance", sparse=True):

#     n_components = [M.shape[0] // 2 if n_components is None else n_components][0]
#     n_neighbors = [M.shape[0] - 1 if n_neighbors is None else n_neighbors][0]

#     L = laplacian(M, n_neighbors, norm=norm, mode=mode, sparse=sparse)

#     l, Q = linalg.eigsh(L, k=n_components, which="SM")

#     sorted_indices = np.argsort(l)[::-1]
#     l = l[sorted_indices]
#     Q = Q[:, sorted_indices]

#     return l, Q
