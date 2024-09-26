import numpy as np
from scipy.sparse import diags
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from variograd_utils.core_utils import vector_wise_corr

def kernelize(A, kernel="linear", scale=50):
    """
    Apply kernel to a matrix A
    
    Parameters
    ----------
    A : array-like
        The input matrix
    kernel : str
        The kernel to apply. Options are "cauchy", "gauss", "linear"
    scale : float
        The scaling parameter for the kernel
        
    Returns
    -------
    A : array-like
        The kernelized matrix
    """

    if kernel == "cauchy":
        A = 1.0 / (1.0 + (A ** 2) / (scale ** 2))
    elif kernel == "gauss":
        A = np.exp(-0.5 * (A ** 2) / (scale ** 2))
    elif kernel == "linear":
        A = 1 / (1 + A / scale)
    else:
        raise ValueError("Unknown kernel type")

    return A


def _affinity_matrix(A, B=None, method="cosine", scale=None):
    """
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
    """

    if scale is None and method in ["linear", "cauchy", "gauss"]:
        raise ValueError("Scaling parameter must be provided for kernel methods")


    if method == "cosine":
        B = A if B is None else B
        M = cosine_similarity(A, B)
    elif method == "correlation":
        B = A if B is None else B
        M = np.corrcoef(A, B)[:A.shape[0], A.shape[1]:]
    elif method in ["linear", "cauchy", "gauss"] and B is None:
        M = kernelize(A, kernel=method)
    elif method in ["linear", "cauchy", "gauss"] and B is not None:
        raise ValueError("Kernel methods are not supported when two matrices are provided")
    else:
        raise ValueError("Unknown affinity method")

    return M


def diffusion_map_embedding(A, n_components=2, alpha=0.5, diffusion_time=1, random_state=None):
    """
    Computes the joint diffusion map embedding of an adjaciency matrix.

    Parameters:
    ----------
    M : np.ndarray
        Target matrix to embed.
    R : np.ndarray
        Reference matrix.
    alpha: float, optional
        Modulates contribution of nodes based on their degeree (default=0.5).
    diffusion_time: float, optional 
        Diffusion time of signal on the graph (default=1).
    
    Returns:
    -------
    self : JointEmbedding
        Fitted instance of JointEmbedding.
    """

    L = _random_walk_laplacian(A, alpha=alpha)

    embedding, vectors, lambdas = _compute_diffusion_map(L, n_components=n_components,
                                                           diffusion_time=diffusion_time,
                                                           random_state=random_state)

    return embedding, vectors, lambdas


def laplacian_eigenmap(A, n_components=2, normalized=True, random_state=None):
    """
    Computes the spectral embedding of an adjaciency matrix.

    Parameters:
    ----------
    M : np.ndarray
        Target matrix to embed.
    normalized: bool, optional
        Whether to normalize the Laplacian matrix (default=True).
    
    Returns:
    -------
    self : JointEmbedding
        Fitted instance of JointEmbedding.
    """

    L = _laplacian(A, normalized=normalized)

    n_components = n_components + 1
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embedding = svd.fit_transform(L)[:, 1:]
    vectors = svd.components_.T[:, 1:]
    lambdas = svd.singular_values_[1:]

    return embedding, vectors, lambdas


def _laplacian(M, normalized=True):
    """
    Computes the Laplacian matrix.

    Parameters:
    ----------
    R : np.ndarray
        Affinity matrix.

    Returns:
    -------
    L : np.ndarray
        The reference Laplacian matrix.
    """

    # Calculate degree
    D = np.sum(M, axis=1).reshape(-1, 1)

    # Compute the Laplacian matrix
    if normalized:
        L = -M / np.sqrt(D @ D.T)
    else:
        L = np.diag(D.squeeze()) - M         

    return L


def _random_walk_laplacian(M, alpha=0.5):
    """
    Computes the random walk Laplacian matrix.

    Parameters:
    ----------
    M: np.ndarray
        Affinity matrix.
    alpha: float, optional
        Modulates contribution of nodes based on their degeree (default=0.5).
    
    Returns:
    -------
    L : np.ndarray
        The reference Laplacian matrix.
    """

    # Compute the normalized Laplacian
    degree = np.sum(M, axis=1)
    d_alpha = diags(np.power(degree, -alpha))
    L_alpha = d_alpha @ M @ d_alpha

    # Compute the random walk Laplacian
    degree = np.sum(L_alpha, axis=1)
    d_alpha = diags(np.power(degree, -1))
    L = d_alpha @ L_alpha

    return L

def _compute_diffusion_map(L, n_components=2, diffusion_time=1, random_state=None):
    """
    Computes the diffusion maps from the random walk Laplacian
    
    Parameters:
    ----------
    L: np.ndarray
        Random walk Laplacian.
    diffusion_time: float, optional 
        Diffusion time of signal on the graph (default=1).
    
    Returns:
    -------
    L : np.ndarray
        The reference Laplacian matrix.
    """

    n_components = n_components + 1
    embedding = TruncatedSVD(n_components=n_components, 
                                random_state=random_state
                                ).fit(L) 
    vectors = embedding.components_.T
    lambdas = embedding.singular_values_

    psi = vectors / np.tile(vectors[:, 0], (vectors.shape[1], 1)).T
    lambdas[1:] = np.power(lambdas[1:], diffusion_time)
    embedding = psi[:, 1:n_components+1] @ np.diag(lambdas[1:n_components+1], 0)
    lambdas = lambdas[1:]
    vectors = vectors[:, 1:]

    return embedding, vectors, lambdas


class JointEmbedding:
    def __init__(self, method="dme", n_components=2, alignment=None,
                 random_state=None, copy=True):
        """
        Initializes the JointEmbedding class.

        Parameters:
        ----------
        method : str, optional
            The method to use for the joint embedding:
            - dme: diffusion map embedding (default).
            - lem: lablacian eigenmaps
        n_components : int, optional
            Number of embedding dimensions (default=2).
        alignment : str, optional
            Method to use (if any) to align the joint embedding of the
            target matrix with an independent embedding of the reference.
            - None: no alignment is performed (default).
            - rotation: the rotation matrix obtained from the dot
                        product of the joint reference embedding and the
                        independent reference embedding.
            - procrustes: procrustes analysis.
            - sign_flip: flips the sign of the singular vectors with negative
                         correlation with the reference.
            - dot_product: dot product of the joint reference embedding and the
                           independent reference embedding.
        random_state : int, optional
            Random seed (default=None).
        copy : bool, optional
            Indicates whether original arrays should be copyed or overwritten (default=False).
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.copy = copy
        self.alignment = alignment
        self.vectors = self.lambdas = None
        self.independent_ref = None

    def fit_transform(self, M, R, C=None, affinity="cosine",
                      method_kwargs=None):
        """
        Fit the specified embedding method and return fitted data.

        Parameters:
        ----------
        M : np.ndarray
            Target matrix to embed.
        R : np.ndarray
            Reference matrix.
        C : np.ndarray, optional
            Correspondence matrix.
        affinity : method used to generate the joint affinity matrix.
        
        Returns:
        -------
        self : JointEmbedding
            Fitted instance of JointEmbedding.
        """
        n = M.shape[0]
        R = np.array(R, copy=self.copy)
        M = np.array(M, copy=self.copy)
        if C is not None:
            C = np.array(C, copy=not self.copy)

        A = self._joint_adjacency_matrix(M, R, C=C, affinity=affinity, method=self.method)

        method_kwargs = {} if method_kwargs is None else method_kwargs

        embedding_function = diffusion_map_embedding if self.method == "dme" else laplacian_eigenmap
        embedding, vectors, lambdas = embedding_function(A, n_components=self.n_components,
                                                            random_state=self.random_state,
                                                            **method_kwargs)

        self.vectors = vectors[n:]
        self.lambdas = lambdas
        embedding_R = embedding[n:]
        embedding_M = embedding[:n]

        if self.alignment is not None:
            A = _affinity_matrix(R, method=affinity)
            embedding_R_ind, _, _ = embedding_function(A, n_components=self.n_components,
                                                  random_state=self.random_state,
                                                  **method_kwargs)
            embedding_M, embedding_R = self._align_embeddings(embedding_M, embedding_R,
                                                              embedding_R_ind, method=self.alignment)
            self.independent_ref = embedding_R_ind

        return embedding_M, embedding_R

    def _joint_adjacency_matrix(self, M, R, C=None,
                                method="dme", affinity="cosine", scale=50):
        """
        Computes the joint adjacency matrix.

        Parameters:
        ----------
        M : np.ndarray
            Target matrix to embed.
        R : np.ndarray
            Reference matrix.
        C : np.ndarray, optional
            Correspondence matrix.
        method : str, optional
            The method to use for the joint embedding:
            - dme: diffusion map embedding (default).
            - lem: laplacian eigenmaps
        affinity : str, optional
            The method to compute the affinity matrix.
        scale : float, optional
            The scaling parameter for the kernel methods.
        
        Returns:
        -------
        A : np.ndarray
            The joint adjacency matrix.
        """

        if (method == "dme") & (affinity != "cosine"):
            raise NotImplementedError("Only cosine affinity is implemented for"
                                       + "diffusion map embedding")
        elif method in ["linear", "cauchy", "gauss"] and C is None:
            raise ValueError("Kernel methods require a correspondence matrix")

        if C is None:
            C = _affinity_matrix(R, M, method=affinity, scale=scale)
        # else:
        #     C = _affinity_matrix(C, method=affinity, scale=scale)
        R = _affinity_matrix(R, method=affinity, scale=scale)
        M = _affinity_matrix(M, method=affinity, scale=scale)
        A = np.block([[R, C],
                      [C.T, M]])

        return A

    def _align_embeddings(self, embedding, joint_reference, independent_reference,
                          method="rotation"):
        """
        Aligns the joint embedding with the reference embedding.

        Parameters:
        ----------
        embedding : np.ndarray
            The joint embedding.
        reference : np.ndarray
            The reference embedding.
        method : str, optional
            The alignment method to use:
            - rotation: dot product of the singular vectors of the covariance matrix (default).
            - procrustes: dot product of the singular vectors of the covariance matrix scaled 
                          by the variance.
            - sign flip: flips the sign of the singular vectors with negative correlation with
                         the reference.
        """

        independent_reference -= independent_reference.mean(axis=0)
        joint_reference -= joint_reference.mean(axis=0)
        embedding -= - embedding.mean(axis=0)

        s = 1
        if method == "sign_flip":
            to_flip = vector_wise_corr(embedding.copy(), independent_reference.copy()) < 0
            R = np.diag([-1 if i else 1 for i in to_flip])
        elif method == "rotation":
            independent_reference /= np.linalg.norm(independent_reference)
            joint_reference /= np.linalg.norm(joint_reference)
            embedding /= np.linalg.norm(embedding)
            R, _ = orthogonal_procrustes(joint_reference, independent_reference)
        elif method == "dot_product":
            R = np.dot(joint_reference.T, independent_reference)
        elif method == "procrustes":
            independent_reference /= np.linalg.norm(independent_reference)
            joint_reference /= np.linalg.norm(joint_reference)
            embedding /= np.linalg.norm(embedding)
            R, s = orthogonal_procrustes(joint_reference, independent_reference)
        else:
            raise ValueError(f"Unknown alignment method: {self.alignment}")

        print(method.upper(), "R.T @ R = I -->", np.allclose(np.dot(R.T, R), np.eye(R.shape[1]), atol=1e-5))
        joint_reference = np.dot(joint_reference, R) * s
        embedding = np.dot(embedding, R) * s

        return embedding, joint_reference
    