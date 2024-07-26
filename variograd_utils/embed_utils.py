#embed_utils.py

from variograd_utils import *
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import diags, linalg
from scipy.linalg import orthogonal_procrustes


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



def diffusion_map(W, n_components=2, alpha=0.5, diffusion_time=0, random_state=None):
    """
    Performs diffusion map embedding on the input matrix W.

    Parameters:
    ----------
        W (np.ndarray): Input matrix.
        n_components (int): Number of embedding dimensions.
        alpha (float): Diffusion parameter.
        diffusion_time (int): Diffusion time.
        random_state (int): Random seed.

    Returns:
    -------
        embedding (np.ndarray): Diffusion map embedding.
        vectors (np.ndarray): Eigenvectors of the diffusion map.
        lambdas (np.ndarray): Eigenvalues of the diffusion map.
    """

    # Get the number of samples
    n = W.shape[0]

    # Compute the normalized Laplacian
    degree = np.sum(W, axis=1)
    d_alpha = diags(np.power(degree, -alpha))
    L_alpha = d_alpha @ W @ d_alpha

    # Compute the random walk Laplacian
    degree = np.sum(L_alpha, axis=1)
    d_alpha = diags(np.power(degree, -1))
    M = d_alpha @ L_alpha
    
    # Compute the eigendecomposition of the random walk Laplacian
    # lambdas, vectors = linalg.eigsh(M, k=n_components + 1)
    # sorted_indices = np.argsort(-np.real(lambdas))
    # vectors = vectors[:, sorted_indices]
    # lambdas = np.real(lambdas[sorted_indices])

    embedding = TruncatedSVD(n_components=n_components+1, random_state=random_state).fit(M) 
    vectors = embedding.components_.T
    lambdas = embedding.singular_values_
    

    # Compute the diffusion map embedding
    psi = vectors / np.tile(vectors[:, 0], (vectors.shape[1], 1)).T
    lambdas[1:] = np.power(lambdas[1:], diffusion_time)

    
    embedding = psi[:, 1:n_components+1] @ np.diag(lambdas[1:n_components+1], 0)
    lambdas = lambdas[1:]

    return embedding, vectors, lambdas



def embed_matrix(M, n_components=2, method="svd", method_kws=None):
    '''
    Embed a matrix M

    Parameters
    ----------
    M : array-like
        The input matrix
    n_components : int, optional
        The number of components to compute (default=2)
    method : str, optional
        The method to compute the embedding. Options are "svd", "diffusion" (default="svd")
        If "svd", uses sklearn.decomposition.TruncatedSVD.
        If "diffusion", uses mapalign.embed.DiffusionMapEmbedding.
    method_kws : dict, optional
        Additional keyword arguments for the embedding algorithm (see method option).

    Returns
    -------
    components : array-like
        The embedding components
    '''

    N = M.shape[0]
    if method == "svd":

        if not isinstance(method_kws, dict):
            method_kws = {}
        if "random_state" in method_kws.keys():
            print("'random_state' set to 0 for reproducibility")
        method_kws["random_state"] = 0
        method_kws["n_components"] = n_components 

        embedding = TruncatedSVD(**method_kws).fit_transform(M) 
        orthogonality = np.dot(embedding.T, embedding)


    elif method == "diffusion":

        if not isinstance(method_kws, dict):
            method_kws = {}
        if "random_state" in method_kws.keys():
            print("'random_state' set to 0 for reproducibility")
        method_kws["random_state"] = 0
        method_kws["n_components"] = n_components

        embedding, vectors, _ = diffusion_map(M, **method_kws)
        orthogonality = np.dot(vectors.T, vectors)

    # Orthogonality check
    np.fill_diagonal(orthogonality, np.nan)
    print("Orthogonality check:",
          "\n\tMax:", np.nanmax(orthogonality),
          "\n\tMin:", np.nanmin(orthogonality),
          "\n\tMean:", np.nanmean(orthogonality),
          "\n\tSD:", np.nanstd(orthogonality), 
          "Shape covariance matrix:", orthogonality.shape)


    return embedding



def joint_embedding(M, R, C=None, n_components=2, method="svd", kernel=None, similarity=None, scale=50, space=None,
                    laplacian="normalized", overwrite=False, method_kws=None, return_ref=False, alignment=None, normalized=True):
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
    method : str, optional
        The method to compute the embedding. Options are "svd", "diffusion" (default="svd")
        If "svd", uses sklearn.decomposition.TruncatedSVD.
        If "diffusion", uses variograd_utils.embed_utils.diffusion_map.
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
    method_kws : dict, optional
        Additional keyword arguments for the embedding algorithm (see method options).
    return_ref : bool, optional
        Whether to return the reference embedding (default=False)
    rotate : bool, optional
        Whether to rotate the joint embedding (default=True)
    normalize : bool, optional
        Whether to apply L2 normalization to the embedding (default=True)
    
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
    if method=="svd":
        L = joint_laplacian(M, R, C, kernel=kernel, similarity=similarity, 
                            scale=scale, space=space, laplacian=laplacian)
        n_components += 1
    elif method=="diffusion":
        L = np.vstack([np.hstack([R, C]), 
                       np.hstack([C.T, M])])


    # Compute the eigenvectors and eigenvalues
    components = embed_matrix(L, n_components=n_components, method=method, method_kws=method_kws)
    A, B = components[:N, :], components[N:, :]


    # Rotate the joint embedding
    if alignment is not None:
        
        if alignment not in ["rotation", "procrustes"]:
            raise ValueError("Unknown alignment method")

        L = R if method=="diffusion" else reference_laplacian(R, kernel=kernel, similarity=similarity, scale=scale, laplacian=laplacian)
        ref = embed_matrix(L, n_components=n_components, method=method, method_kws=method_kws)
        
        ref -= ref.mean()
        A -= A.mean()
        B -= - B.mean()

        print(A[0, :5], B[0, :5], ref[0, :5])

        if alignment == "rotation":  
            rotation_mat = np.dot(A.T, ref)
            A = np.dot(A, rotation_mat)
            B = np.dot(B, rotation_mat)

        elif alignment == "procrustes":
            raise ValueError("Procrustes alignment not implemented")
            rotation_mat, s = orthogonal_procrustes(A, ref, check_finite=False)
            A = np.dot(A, rotation_mat.T) * s
            B = np.dot(B, rotation_mat.T) * s
        

    offset =  1 if method=="svd" else 0
    A, B = A[:, offset:], B[:, offset:]

    # Normalize the embeddings
    if normalized and alignment is None:
        B = normalize(B[:, offset:], axis=1)
        A = normalize(A[:, offset:], axis=1)

    
    if return_ref:
        return B, A, ref, L
    else:
        return B
    