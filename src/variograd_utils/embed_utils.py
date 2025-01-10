import warnings
import numpy as np
from scipy.sparse import diags
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from variograd_utils.core_utils import vector_wise_corr

class JointEmbedding:
    """
    JointEmbedding class for computing the joint embedding of two matrices.

    Parameters:
    ----------
    method : str, optional
        The embedding method to use. Options are:
        - "dme": diffusion map embedding (default)
        - "le": Laplacian eigenmap
    n_components : int, optional
        Number of components to compute (default=2).
    alignment : str, optional
        The alignment method to use:
        - "procrustes": orthogonal Procrustes rotation with scaling.
        - "rotation": orthogonal Procrustes rotation (default)
        - "sign_flip": multiply the embedding dimension by -1 if the correlation with
            the corresponding dimension in the reference is negative.
        - "dot_product": align the embedding using the dot product of the joint reference
            and independent reference embeddings.
    random_state : int, optional
        Random seed of the SVDs.
    copy : bool, optional
        Whether to copy the input matrices (default=True).

    Attributes:
    ----------
    method : str
        The embedding method to use.
    n_components : int
        Number of components to compute.
    alignment : str
        The alignment method to use.
    random_state : int
        Random seed of the SVDs.
    copy : bool
        Whether to copy the input matrices.
    vectors : np.ndarray
        The eigenvectors of the embedding.
    lambdas : np.ndarray
        The eigenvalues of the embedding.
    independent_ref : np.ndarray
        The independent reference embedding used for alignment.
    
    Methods:
    --------
    fit_transform(M, R, C=None, affinity="cosine", scale=None, method_kwargs=None)
        Compute the joint embedding of M and R using the specified method.
    _joint_affinity_matrix(M, R, C=None, affinity="cosine", scale=None)
        Computes the joint affinity matrix.
    _align_embeddings(embedding, joint_reference, independent_reference, method="rotation")
        Align the joint embedding with the independently computed reference embedding.
    _affinity_matrix(M, method="cosine", scale=None)
        Compute the joint affinity matrix of the input data.
    """

    def __init__(self, method="dme", n_components=2, alignment=None,
                 random_state=None, copy=True):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.copy = copy
        self.alignment = alignment
        self.vectors = self.lambdas = None
        self.independent_ref = None


    def fit_transform(self, M, R, C=None, affinity="cosine", scale=None, method_kwargs=None):
        """
        Compute the joint embedding of M and R using the specified method.

        Parameters:
        ----------
        M : np.ndarray
            Target matrix to embed.
        R : np.ndarray
            Reference matrix.
        C : np.ndarray, optional
            Correspondence matrix.
            If affinity is not "precomputed", C is used as is and the
            specified affinity method is applied only to while M and R.
        affinity : str, optional
            The method to compute the affinity matrices. Options are:
            - "cosine": cosine similarity (default)
            - "correlation": Pearson correlation coefficient
            - "linear": linear kernel
            - "cauchy": Cauchy kernel
            - "gauss": Gaussian kernel
            - "precomputed": precomputed affinity matrix.
                            In this case, M and R are assumed to be affinity matrices
                            and a correspondence matrix C must be specified.
        scale : float, optional
            The scaling parameter for the kernel methods.
        method_kwargs : dict, optional
            Additional keyword arguments for the embedding method.

        Returns:
        -------
        self : JointEmbedding
            Fitted instance of JointEmbedding.
        embedding_M : np.ndarray
            The joint embedding of M.
        embedding_R : np.ndarray
            The joint embedding of R.
        
        Raises:
        -------
        ValueError
            If the affinity is "precomputed" and C is not specified.
        """

        if (affinity == "precomputed") & (C is None):
            raise ValueError("Precomputed affinity assumes M and R are already affinity matrices,"
                             + "so a correspondance affinity matrx C must be specified too.")
        n = M.shape[0]

        R = np.array(R, copy=self.copy)
        M = np.array(M, copy=self.copy)
        if C is not None:
            C = np.array(C, copy=self.copy)

        A = self._joint_affinity_matrix(M, R, C=C, affinity=affinity, scale=scale)

        method_kwargs = {} if method_kwargs is None else method_kwargs
        embedding_function = diffusion_map_embedding if self.method == "dme" else laplacian_eigenmap
        embedding, vectors, lambdas = embedding_function(A, n_components=self.n_components,
                                                            random_state=self.random_state,
                                                            **method_kwargs)

        self.vectors = vectors
        self.lambdas = lambdas
        embedding_R = embedding[:n]
        embedding_M = embedding[n:]

        if self.alignment is not None:
            A = _affinity_matrix(R,method=affinity, scale=scale) if affinity != "precomputed" else R
            embedding_R_ind, _, _ = embedding_function(A, n_components=self.n_components,
                                                       random_state=self.random_state,
                                                       **method_kwargs)

            embedding_M, embedding_R = self._align_embeddings(embedding_M, embedding_R,
                                                              embedding_R_ind,method=self.alignment)
            self.independent_ref = embedding_R_ind


        return embedding_M, embedding_R


    def _joint_affinity_matrix(self, M, R, C=None, affinity="cosine", scale=None):
        """
        Computes the joint affinity matrix.

        Parameters:
        ----------
        M : np.ndarray
            Target matrix to embed.
        R : np.ndarray
            Reference matrix.
        C : np.ndarray, optional
            Correspondence matrix of shape (n_samples_M, n_samples_R).
            If affinity is not "precomputed", C is used as is and the
            specified affinity method is applied only to while M and R.
        affinity : str, optional
            The method to compute the affinity matrices. Options are:
            - "cosine": cosine similarity (default)
            - "correlation": Pearson correlation coefficient
            - "linear": linear kernel
            - "cauchy": Cauchy kernel
            - "gauss": Gaussian kernel
            - "precomputed": precomputed affinity matrix.
        scale : float, optional
            The scaling parameter for the kernel methods.
        
        Returns:
        -------
        A : np.ndarray
            The joint affinity matrix.
        """

        if C is None:
            if affinity in ["cosine", "correlation"]:
                A = np.vstack([R, M])
                A = _affinity_matrix(A, method=affinity, scale=scale)

            elif affinity in ["linear", "cauchy", "gauss"]:
                C = euclidean_distances(M, R)
                scale_C = np.percentile(C, 100 * np.sum(t > R) / R.size)
                C = kernel_affinity(C, kernel=affinity, scale=scale_C)
                A = np.block([[_affinity_matrix(R, method=affinity, scale=scale), C.T],
                              [C, _affinity_matrix(M, method=affinity, scale=scale)]])

        else:
            if affinity == "precomputed":
                A = np.block([[R, C.T],
                              [C, M]])
            else:
                A = np.block([[_affinity_matrix(R, method=affinity, scale=scale), C.T],
                              [C, _affinity_matrix(M, method=affinity, scale=scale)]])

        return A


    def _align_embeddings(self, embedding, joint_reference, independent_reference,
                          method="rotation"):
        """
        Align the joint embedding with the independently computed reference embedding.

        Parameters:
        ----------
        embedding : np.ndarray
            The joint embedding.
        reference : np.ndarray
            The reference embedding.
        method : str, optional
            The alignment method to use:
            - "procrustes": orthogonal Procrustes rotation with scaling.
            - "rotation": orthogonal Procrustes rotation (default)
            - "sign_flip": multiply the embedding dimension by -1 if the correlation with
                the corresponding dimension in the reference is negative.
            - "dot_product": align the embedding using the dot product of the joint reference
                and independent reference embeddings.
            
        Returns:
        -------
        embedding : np.ndarray
            The aligned joint embedding.
        reference : np.ndarray
            The aligned reference embedding.

        """

        independent_reference -= independent_reference.mean(axis=0)
        joint_reference -= joint_reference.mean(axis=0)
        embedding -= embedding.mean(axis=0)

        independent_reference /= np.linalg.norm(independent_reference)
        joint_reference /= np.linalg.norm(joint_reference)
        embedding /= np.linalg.norm(embedding)

        s = 1
        if method == "sign_flip":
            to_flip = vector_wise_corr(embedding.copy(), independent_reference.copy()) < 0
            R = np.diag([-1 if i else 1 for i in to_flip])

        elif method == "rotation":
            R, _ = orthogonal_procrustes(joint_reference, independent_reference)

        elif method == "dot_product":
            # raise NotImplementedError("Dot product rotation not implemented")
            R = np.dot(joint_reference.T, independent_reference)

        elif method == "procrustes":
            R, s = orthogonal_procrustes(joint_reference, independent_reference)

        else:
            raise ValueError(f"Unknown alignment method: {self.alignment}")

        joint_reference = np.dot(joint_reference, R) * s
        embedding = np.dot(embedding, R) * s

        # independent_reference /= np.linalg.norm(independent_reference)
        # joint_reference /= np.linalg.norm(joint_reference)
        # embedding /= np.linalg.norm(embedding)

        return embedding, joint_reference


def _affinity_matrix(M, method="cosine", scale=None):
    """
    Compute the affinity matrix between two matrices A and B
    
    Parameters
    ----------
    M : array-like
        The input matrix of shape smaples x features
    method : str
        The method to compute the affinity matrix. Options are:
        - "cosine": cosine similarity
        - "correlation": Pearson correlation coefficient
        - "linear": linear kernel
        - "cauchy": Cauchy kernel
        - "gauss": Gaussian kernel
          
    scale : float
        The scaling parameter for the kernel
    
    Returns
    -------
    A : array-like
        The affinity matrix
    """


    if method == "cosine":
        A = cosine_similarity(M)

    elif method == "correlation":
        A = np.corrcoef(M)

    elif method in {"linear", "cauchy", "gauss"}:
        # A = M if _is_square(M) else euclidean_distances(M)
        A = kernel_affinity(A, kernel=method, scale=scale)

    else:
        raise ValueError("Unknown affinity method")

    return A


def kernel_affinity(A, kernel="linear", scale=None):
    """
    Apply kernel to a matrix A

    Parameters
    ----------
    A : array-like
        The input matrix of shape smaples x features
    kernel : str
        The kernel to apply. Options are "cauchy", "gauss", "linear"
    scale : float
        The scaling parameter for the kernel

    Returns
    -------
    A : array-like
        The kernelized matrix
    """

    if scale is None:
        scale = 1 / A.shape[1]

    if kernel == "cauchy":
        A = 1.0 / (1.0 + (A ** 2) / (scale ** 2))

    elif kernel == "gauss":
        A = np.exp(-0.5 * (A ** 2) / (scale ** 2))

    elif kernel == "linear":
        A = 1 / (1 + A / scale)

    else:
        raise ValueError("Unknown kernel type")

    return A


def spectral_affinity(M, R, n_components=2, random_state=None):
    """
    Compute the spectral similarity between two matrices using Laplacian Eigenmaps and Procrustes alignment.

    This function calculates the cosine similarity between Laplacian Eigenmaps of two matrices, 
    `M` and `R` of shape observations x features. The embeddings are aligned using 
    the Orthogonal Procrustes transformation before computing similarity.

    Parameters
    ----------
    M : numpy.ndarray
        The first input matrix.
    R : numpy.ndarray
        The second input matrix (also target of the procrustes alignment).
    n_components : int, optional
        Number of dimensions for the Laplacian Eigenmap embeddings. Default is 2.
    random_state : int, optional
        Determines random number generation for eigenmap embedding. Default is None.

    Returns
    -------
    numpy.ndarray
        A cosine similarity matrix representing the similarity between aligned 
        embeddings of `M` and `R`.
    """
    
    embedding_M, _, _ = laplacian_eigenmap(M, n_components=n_components, normalized=True, random_state=random_state)
    embedding_R, _, _ = laplacian_eigenmap(R, n_components=n_components, normalized=True, random_state=random_state)

    embedding_M -= embedding_M.mean(axis=0)
    embedding_M /= np.linalg.norm(embedding_M)
    embedding_R -= embedding_R.mean(axis=0)
    embedding_R /= np.linalg.norm(embedding_R)

    rotation, scaling = orthogonal_procrustes(embedding_M, embedding_R)
    embedding_M = np.dot(embedding_M, rotation) * scaling

    A = cosine_similarity(embedding_M, embedding_R)

    return A


def diffusion_map_embedding(A, n_components=2, alpha=0.5, diffusion_time=1, random_state=None):
    """
    Computes the joint diffusion map embedding of an affinity matrix.

    Parameters:
    ----------
    A : np.ndarray
        Target matrix to embed.
    n_components: int, optional
        Number of components to keep (default=2).
    alpha: float, optional
        Controls Laplacian normalization, balancing local and global structures in embeddings. 
        Alpha <= 0.5 emphasize local structures, alpha >= 1 focus on global organization.
    diffusion_time: float, optional
        Determines the scale of the random walk process (default=1).
    random_state: int, optional
        Random seed of the SVD.
    
    Returns:
    -------
    self : JointEmbedding
        Fitted instance of JointEmbedding.
    """

    if np.any(A < 0):
        warnings.warn("Negative values in the affinity matrix set to 0", RuntimeWarning)
        A[A<0] = 0

    L = _random_walk_laplacian(A, alpha=alpha)

    embedding, vectors, lambdas = _diffusion_map(L, n_components=n_components,
                                                           diffusion_time=diffusion_time,
                                                           random_state=random_state)

    return embedding, vectors, lambdas


def _diffusion_map(L, n_components=2, diffusion_time=1, random_state=None):
    """
    Computes the diffusion map of the random walk Laplacian L.
    
    Parameters:
    ----------
    L: np.ndarray
        Random walk Laplacian.
    n_components: int, optional
        Number of components to compute (default=2).
    diffusion_time: float, optional
        Determines the scale of the random walk process (default=1).
    random_state: int, optional
        Random seed of the SVD.    
    
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
    if np.any(vectors[:, 0] == 0):
        warnings.warn("0 values found in the first eigenvector; 1e-15 was added to all vectors.",
                      RuntimeWarning)
        vectors += 1e-16

    psi = vectors / np.tile(vectors[:, 0], (vectors.shape[1], 1)).T
    lambdas[1:] = np.power(lambdas[1:], diffusion_time)
    embedding = psi[:, 1:n_components] @ np.diag(lambdas[1:n_components], 0)
    lambdas = lambdas[1:]
    vectors = vectors[:, 1:]

    return embedding, vectors, lambdas


def _random_walk_laplacian(A, alpha=0.5):
    """
    Computes the random walk Laplacian of an affinity matrix A.

    Parameters:
    ----------
    A: np.ndarray
        Affinity matrix.
    alpha: float, optional
        Controls Laplacian normalization, balancing local and global structures in embeddings. 
        Alpha <= 0.5 emphasize local structures, alpha >= 1 focus on global organization.
    
    Returns:
    -------
    L : np.ndarray
        The random walk Laplacian of A.
    """

    if (A.shape[0] != A.shape[1]) or not np.allclose(A, A.T):
        raise ValueError("A must be a squared, symmetrical affinity matrix.")

    # Compute the normalized Laplacian
    degree = np.sum(A, axis=1)
    d_alpha = diags(np.power(degree, -alpha))
    L_alpha = d_alpha @ A @ d_alpha

    # Compute the random walk Laplacian
    degree = np.sum(L_alpha, axis=1)
    d_alpha = diags(np.power(degree, -1))
    L = d_alpha @ L_alpha

    return L


def laplacian_eigenmap(A, n_components=2, normalized=True, random_state=None):
    """
    Computes the spectral embedding of an affinity matrix.

    Parameters:
    ----------
    A : np.ndarray
        Target matrix to embed.
    n_components: int, optional
        Number of components to compute (default=2).
    normalized: bool, optional
        Whether to normalize the Laplacian matrix (default=True).
    random_state: int, optional
        Random seed of the SVD.
    
    Returns:
    -------
    self : JointEmbedding
        Fitted instance of JointEmbedding.
    """

    if np.any(A < 0):
        warnings.warn("Negative values in the affinity matrix set to 0", RuntimeWarning)
        A[A<0] = 0

    L = _laplacian(A, normalized=normalized)

    n_components = n_components + 1
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embedding = svd.fit_transform(L)[:, 1:]
    vectors = svd.components_.T[:, 1:]
    lambdas = svd.singular_values_[1:]

    return embedding, vectors, lambdas


def _laplacian(A, normalized=True):
    """
    Computes the Laplacian of an affinity matrix A.

    Parameters:
    ----------
    A : np.ndarray
        Affinity matrix.
    normalized : bool, optional
        Compute the normalized Laplacian (default=True).

    Returns:
    -------
    L : np.ndarray
        Laplacian matrix of M.
    """

    if (A.shape[0] != A.shape[1]) or not np.allclose(A, A.T):
        raise ValueError("A must be a squared, symmetrical affinity matrix.")

    # Calculate degree
    D = np.sum(A, axis=1).reshape(-1, 1)

    # Compute the Laplacian matrix
    if normalized:
        L = -A / np.sqrt(D @ D.T)

    else:
        L = np.diag(D.squeeze()) - A         

    return L
