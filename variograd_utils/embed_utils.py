#embed_utils.py

from variograd_utils import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def kernelize(A, kernel="linear", scale=50):
    '''
    Apply kernel to a matrix A
    
    Parameters
    ----------
    A : array-like
        The input matrix
    kernel : str
        The kernel to apply. Options are "cauchy", "log", "gauss", "linear"
    scale : float
        The scaling parameter for the kernel
        
    Returns
    -------
    A : array-like
        The kernelized matrix
    '''

    if kernel == 'cauchy':
        A = 1.0 / (1.0 + (A ** 2) / (scale ** 2))
    elif kernel == 'log':
        A = np.log(1.0 + (A ** 2) / (scale ** 2))
    elif kernel == 'gauss':
        A = np.exp(-0.5 * (A ** 2) / (scale ** 2))
    elif kernel == "linear":
        A = A / scale
    else:
        raise ValueError("Unknown kernel type")
    
    return A


def affinity(A, B=None, method="cosine"):
    '''
    Compute the affinity matrix between two matrices A and B
    
    Parameters
    ----------
    A : array-like
        The first matrix
    B : array-like
        The second matrix
    method : str
        The method to compute the affinity. Options are "cosine", "correlation"
    
    Returns
    -------
    M : array-like
        The affinity matrix
    '''

    B = A if B is None else B

    if method == 'cosine':
        M = cosine_similarity(A, B)
    elif method == "correlation":
        M = np.corrcoeff(A, B)[:A.shape[0], A.shape[1]:]
    else:
        raise ValueError("Unknown affinity method")
    
    return M


def joint_laplacian(M, R, C=None, kernel=None, similarity=None, scale=50, space=None, laplacian="normalized"):
    '''
    Compute the Laplacian matrix for the joint embedding
    
    Parameters
    ----------
    M : array-like
        The target matrix
    R : array-like
        The reference matrix
    C : array-like, optional
        The correspondence matrix (default=None)
    kernel : str, optional
        The kernel to apply. Options are "cauchy", "log", "gauss", "linear" (default=None)
    affinity : str, optional
        The method to compute the affinity. Options are "cosine", "correlation" (default=None)
    scale : float, optional
        The scaling parameter for the kernel (default=50)
    space : float, optional
        The scaling parameter for the correspondence matrix (default=None)
    laplacian : str, optional
        The type of Laplacian to compute. Options are "normalized", "unnormalized" (default="normalized")
    
    Returns
    -------
    L : array-like
        The joint Laplacian matrix
    '''

    # Apply kernel to matrices
    space = scale if space is None else space
    if kernel is not None:
        R = kernelize(R, kernel=kernel, scale=scale)
        M = kernelize(M, kernel=kernel, scale=scale)
        if isinstance(C, np.ndarray):
            C = kernelize(C, kernel=kernel, scale=space)
    
    # Convert to affinity and compute correspondence matrix
    if similarity is not None:
        R = affinity(R, method=similarity)
        M = affinity(M, method=similarity)
        if isinstance(C, np.ndarray):
            C = affinity(C, method=similarity)
        else:
            C = affinity(R, M, method=similarity)
    
    # Build the joint affinity matrix and degree matrix
    A = np.vstack([np.hstack([R, C]), 
                   np.hstack([C, M])])
    D = np.sum(A, axis=1).reshape(-1, 1)

    # Compute the Laplacian matrix
    if laplacian == "unnormalized":
        L = np.diag(D.squeeeze()) - A
    else:
        L = - A / np.sqrt(D @ D.T)

    return L



def reference_laplacian(R, kernel=None, similarity=None, scale=50, laplacian="normalized"):
    '''
    Compute the Laplacian matrix for the reference embedding
    
    Parameters
    ----------
    R : array-like
        The reference matrix
    kernel : str, optional
        The kernel to apply. Options are "cauchy", "log", "gauss", "linear" (default=None)
    affinity : str, optional
        The method to compute the affinity. Options are "cosine", "correlation" (default=None)
    scale : float, optional
        The scaling parameter for the kernel (default=50)
    laplacian : str, optional
        The type of Laplacian to compute. Options are "normalized", "unnormalized" (default="normalized")
        
    Returns
    -------
    L : array-like
        The reference Laplacian matrix
    
    '''
    # Apply kernel to matrices
    if kernel is not None:
        R = kernelize(R, kernel=kernel, scale=scale)
    
    # Convert to affinity and compute correspondence matrix
    if similarity is not None:
        R = affinity(R, method=similarity)
    
    # Calculate degree
    D = np.sum(R, axis=1).reshape(-1, 1)

    # Compute the Laplacian matrix
    if laplacian == "unnormalized":
        L = np.diag(D.squeeeze()) - R
    else:
        L = - R / np.sqrt(D @ D.T)

    return L



def joint_embedding(M, R, C=None, n_components=2, kernel=None, similarity=None, scale=50, space=None, 
                    laplacian="normalized", overwrite=False, svd_kws=None, return_ref=False, rotate=True):
    '''
    Compute the joint embedding of two matrices M and R
    
    Parameters
    ----------
    M : array-like
        The target matrix   
    R : array-like
        The reference matrix
    C : array-like, optional
        The correspondence matrix (default=None)
    n_components : int, optional
        The number of components to compute (default=2)
    kernel : str, optional
        The kernel to apply. Options are "cauchy", "log", "gauss", "linear" (default=None)
    
    similarity : str, optional
        The method to compute the affinity. Options are "cosine", "correlation" (default=None)
    scale : float, optional
        The scaling parameter for the kernel (default=50)
    space : float, optional
        The scaling parameter for the correspondence matrix (default=None)
    laplacian : str, optional
        The type of Laplacian to compute. Options are "normalized", "unnormalized" (default="normalized")
        overwrite : bool, optional
        Whether to overwrite the input matrices (default=False)
    svd_kws : dict, optional
        Additional keyword arguments for TruncatedSVD (default=None)
    return_ref : bool, optional
        Whether to return the reference embedding (default=False)
    rotate : bool, optional
        Whether to rotate the joint embedding (default=True)
    
    Returns
    -------
    B : array-like
        The joint embedding
    A : array-like
        The reference embedding'''

    N = M.shape[0]
    if M.shape != R.shape:
        raise ValueError("The input matrices must have the same shape")

    # avoid overwriting the input matrices
    if not overwrite:
        M = M.copy()
        R = R.copy()
        if isinstance(C, np.ndarray):
            C = C.copy()


    # Compute the joint Laplacian matrix
    L = joint_laplacian(M, R, C, kernel=kernel, similarity=similarity, 
                        scale=scale, space=space, laplacian=laplacian)

    # Compute the eigenvectors and eigenvalues
    n_components += 1
    if isinstance(svd_kws, dict) :
        if "random_state" in svd_kws.keys():
            print("Truncated SVD's 'random_state' set to 0 for reproducibility")        
        svd_kws["random_state"] = 0
        svd_kws["n_components"] = n_components
    else:
        svd_kws = {"n_components": n_components, "random_state": 0}
    
    SVD = TruncatedSVD(**svd_kws).fit(L)
    A, B = SVD.components_[:, :N], SVD.components_[:, N:]


    # orthogonality check
    orthogonality = np.dot(SVD.components_, SVD.components_.T)
    np.fill_diagonal(orthogonality, np.nan)
    print("Orthogonality check:",
          "\n\tMax:", np.nanmax(orthogonality),
          "\n\tMin:", np.nanmin(orthogonality),
          "\n\tMean:", np.nanmean(orthogonality))


    # Rotate the joint embedding
    if rotate:

        SVD_ref = TruncatedSVD(**svd_kws).fit(reference_laplacian(R, kernel=kernel, similarity=similarity, 
                                                                  scale=scale, laplacian=laplacian))
        ref_norm = normalize(SVD_ref.components_, axis=0)
        
        A_norm = normalize(A, axis=0)
        rotation_mat = np.dot(A_norm, ref_norm.T)

        B = np.dot(B.T, rotation_mat)

    # Normalize the embeddings
    B = normalize(B, axis=1)
    A = normalize(A.T, axis=1)

    if return_ref:
        return B[:, 1:], A[:, 1:]
    else:
        return B[:, 1:]



