# core_utils.py

import os.path as os
from itertools import combinations
import numpy as np
from sklearn.utils import Bunch
import nibabel as nib
import hcp_utils as hcp
import variograd_utils
from variograd_utils.brain_utils import save_gifti, vertex_info_10k

def create_bunch_from(A):
    """
    Create a `Bunch` object from various input types.

    Parameters
    ----------
    A : str, dict, or numpy.lib.npyio.NpzFile
        The input to convert into a `Bunch`. Can be:
        - A string representing the path to a `.npz` file.
        - A dictionary containing key-value pairs.
        - A `numpy.lib.npyio.NpzFile` object.

    Returns
    -------
    sklearn.utils.Bunch
        A `Bunch` object containing the key-value pairs from the input.
    """

    if isinstance(A, str):
        return Bunch(**dict(np.load(A).items()))
    elif isinstance(A, dict):
        return Bunch(**A)
    elif isinstance(A, np.lib.npyio.NpzFile):
        return Bunch(**dict(A.items()))
    else:
        raise TypeError("'A' should be a dictionary, a numpy NpzFile, or the path to a NpzFile")


class dataset:
    """
    Manage group-level data and common operations.

    Attributes
    ----------
    group_dir : str
        Directory containing group-level data.
    subj_dir : str
        Directory containing subject-level data.
    output_dir : str
        Directory for saving outputs.
    utils_dir : str
        Directory where the module is installed.
    mesh10k_dir : str
        Directory containing 10k-resolution meshes.
    subj_list : numpy.ndarray
        List of subject IDs.
    N : int
        Number of subjects in the dataset.
    id : str
        Identifier for the dataset.
    pairs : list
        List of all subject pairs for pairwise computations.

    Methods
    -------
    outpath(filename, replace=True)
        Generates an output file path, optionally checking for existing files.
    load_surf(h, k=32, type="midthickness", assign=True)
        Loads the group average cortical surface.
    load_embeddings(h, alg=None, return_bunch=True)
        Loads geometric embeddings.
    allign_embeddings(h=None, alg=None, overwrite=True)
        Aligns embeddings by flipping signs based on reference embeddings.
    generate_avg_surf(h, k=32, assign=False, save=True, filename=None)
        Computes the average cortical surface for a hemisphere.
    load_gdist_triu(hemi=None)
        Loads geodesic distances as a vector.
    load_gdist_matrix(hemi=None, D=0)
        Reconstructs a geodesic distance matrix.
    """

    def __init__(self):
        """
        Initialize the `dataset` object.
        """

        pkg_path = os.dirname(variograd_utils.__file__)
        file = open(f'{pkg_path}/directories.txt','r')
        directories = {line.split("=")[0]: line.split("=")[1].replace("\n", "") for line in file}
        self.group_dir = directories["group_dir"]
        self.subj_dir = directories["subj_dir"]
        self.output_dir = directories["output_dir"]
        self.utils_dir = f"{pkg_path}"
        self.mesh10k_dir = directories["mesh10k_dir"]

        self.subj_list = np.loadtxt(directories["subj_list"]).astype("int32")
        self.N = len(self.subj_list)
        self.id = f"{self.N}avg"
        self.pairs = list(combinations(self.subj_list, 2))

        surf_args = np.array(np.meshgrid(["L", "R"], [10, 32],
                                         ["midthickness", "cortex_midthickness"]),
                             dtype="object").T.reshape(-1, 3)
        for h, k, name in surf_args:
            if k==32:
                path = f"{self.group_dir}/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii"
            elif k==10:
                path = f"{self.mesh10k_dir}/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii"
            else:
                raise ValueError("Only 10k and 32k resolutions are supported")

            setattr(self, f"{h}_{name}_{k}k",  path)


    def outpath(self, filename, replace=True):
        """
        Generate a path for an output file directing it to the output directory (`out_dir`).

        Parameters
        ----------
        filename : str
            The desired filename for the output file.
        replace : bool, optional
            If False, raises an error if the file already exists. Default is True.

        Returns
        -------
        str
            The full path to the output file.
        """

        filename = f"{self.output_dir}/{filename}"
        if os.exists(filename) & (not replace):
            raise ValueError("This file already exists. Change 'filename' or set 'replace' to True")
        return filename
    

    def load_surf(self, h, k=32, type="midthickness", assign=True):
        """
        Load the group average cortical surface mesh.

        Parameters
        ----------
        h : str
            Hemisphere to load ('L' or 'R').
        k : int, optional
            Resolution of the cortical surface (default is 32).
        type : str, optional
            Type of surface to load ('midthickness' or 'cortex_midthickness').
        assign : bool, optional
            If False, adds the surface to the object attributes. Default is True.

        Returns
        -------
        nibabel.gifti.GiftiImage
            The loaded surface file.
        """

        if k==32:
            surf = nib.load(f"{self.group_dir}/{self.id}.{h}.{type}_MSMAll.{k}k_fs_LR.surf.gii")
        elif k==10:
            surf = nib.load(f"{self.mesh10k_dir}/{self.id}.{h}.{type}_MSMAll.{k}k_fs_LR.surf.gii")
        else:
            raise ValueError("Only 10k and 32k resolutions are supported")

        if assign:
            return surf
        setattr(self, f"surf{k}k_{h}", surf)


    def load_embeddings(self, h, alg=None, return_bunch=True):
        """
        Load geometric embeddings for a specific hemisphere and algorithm.

        Parameters
        ----------
        h : str
            Hemisphere to load ('L' or 'R').
        alg : str or list of str, optional
            Algorithm(s) for which to load embeddings. Will load all embeddings whose keys start
            with the specified string(s). If None, all embeddings are loaded.
        return_bunch : bool, optional
            If True, returns a `Bunch` object, else the embedding(s) is(are) added to the dataset
            attributes. Default is True.

        Returns
        -------
        sklearn.utils.Bunch
            A `Bunch` object containing the embeddings for the specified hemisphere.
        """

        embeddings = create_bunch_from(self.outpath(f"All.{h}.embeddings.npz"))

        if alg:
            if isinstance(alg, str):
                alg = [alg]

            attr_to_keep = [attr for attr in embeddings.keys() for a in alg if attr.startswith(a)]
            embeddings = create_bunch_from({k: embeddings[k] for k in attr_to_keep})

        if return_bunch:
            return embeddings

        setattr(self, f"embeds_{h}", embeddings)


    def generate_avg_surf(self, h, k=32, assign=False, save=True, filename=None):
        """
        Generate the average cortical surface mesh for a given hemisphere.

        Parameters
        ----------
        h : str
            Hemisphere for which to generate the average surface ('L' or 'R').
        k : int, optional
            Resolution of the cortical surface (default is 32).
        assign : bool, optional
            If False, adds the average surface to the object's attributes. Default is False.
        save : bool, optional
            If True, saves the average surface to a file. Default is True.
        filename : str, optional
            The filename for saving the average surface. If None, a default filename is used.

        Returns
        -------
        list
            A list containing the averaged pointsets and triangles, if `assign` is True.
        """

        pointsets, triangles = subject(self.subj_list[0]).load_surf(h, k=k).darrays
        pointsets = pointsets.data
        triangles = triangles.data

        for id in self.subj_list[1:]:
            pointsets += subject(id).load_surf(h, k=k).darrays[0].data
        pointsets /= self.N

        avg_surf = [pointsets, triangles]
        if save:
            structure = ["CORTEX_LEFT" if h=="L" else "CORTEX_RIGHT"][0]
            if filename is None:
                filename = f"{self.mesh10k_dir}/{self.id}.{h}.midthickness_MSMAll.{k}k_fs_LR.surf.gii"

            save_gifti(darrays=avg_surf, intents=[1008, 1009],
                       dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"],
                       filename=filename, structure=structure)

        if assign:
            return avg_surf


    def load_gdist_triu(self, hemi=None):
        """
        Load the vectorized upper triangle of the (symmetric) geodesic dstance matrix.

        Parameters
        ----------
        hemi : str
            Hemisphere to load ('L' or 'R').

        Returns
        -------
        numpy.ndarray
            A 1D vector containing the geodesic distances.
        """

        if hemi is None:
            raise TypeError("Please specify one hemisphere: 'L' or 'R'")

        filename = f"{self.output_dir}/{self.id}.{hemi}.gdist_triu.10k_fs_LR.npy"
        if os.exists(filename):
            return np.load(filename)
        else:
            raise ValueError(f"{filename} does not exist, generate it and try again.")


    def load_gdist_matrix(self, hemi=None, D=0):
        """
        Load the geodesic distance matrix of one hemisphere.

        Parameters
        ----------
        hemi : str
            Hemisphere to load ('L' or 'R').
        D : int, optional
            Value to fill the diagonal of the matrix. Default is 0.

        Returns
        -------
        numpy.ndarray
            The reconstructed geodesic distance matrix.
        """

        if hemi is None:
            raise TypeError("Please specify one hemisphere: 'L' or 'R'")

        v = self.load_gdist_triu(hemi)
        return reconstruct_matrix(v, diag_fill=D, k=1)




class subject:
    """
    Manage individual subject-level data.

    Attributes
    ----------
    id : int
        Subject identifier.
    idx : int
        Index of the subject in the dataset subject list.
    dir : str
        Directory containing the subject's data.

    Methods
    -------
    outpath(filename, replace=True)
        Generates an output file path for the subject.
    load_surf(h, k=32, MNINonLinear=False, type="midthickness", assign=True)
        Loads a subject's cortical surface.
    load_grads(h=None, k=32, assign=True)
        Loads functional gradients.
    load_gdist_triu(hemi=None)
        Loads geodesic distances as a vector.
    load_gdist_matrix(hemi=None)
        Reconstructs a geodesic distance matrix.
    load_gdist_vertex(hemi=None, target=None, source=None)
        Loads geodesic distances from a specific vertex.
    load_embeddings(h, alg=None, return_bunch=True)
        Loads embeddings for a subject.
    """

    def __init__(self, ID=None):
        """
        Initialize the `subject` object.
        """

        data = dataset()
        self.id = ID
        self.idx = np.argwhere(data.subj_list==ID).squeeze()
        self.dir = f"{data.subj_dir}/{ID}"

        surf_args = np.array(np.meshgrid(["L", "R"], [10, 32],
                                         ["MNINonLinear","T1w"],
                                         ["midthickness", "cortex_midthickness"]),
                                         dtype="object").T.reshape(-1, 4)
        for h, k, w, name in surf_args:
            path = f"{self.dir}/{w}/fsaverage_LR{k}k/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii"
            if os.exists(path):
                setattr(self, f"{h}_{name}_{k}k_{w}",  path)

            path = self.outpath(f"{w}/fsaverage_LR{k}k/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii")
            if os.exists(path):
                setattr(self, f"{h}_{name}_{k}k_{w}",  path)


    def outpath(self, filename, replace=True):
        """
        Generate an output file path for the subject in the format `output_dir/subject_id/filename`.

        Parameters
        ----------
        filename : str
            The desired filename for the output file.
        replace : bool, optional
            If False, raises an error if the file already exists. Default is True.

        Returns
        -------
        str
            The full path to the output file.
        """

        filename = f"{dataset().output_dir}/{self.id}/{filename}"
        if os.exists(filename) & (not replace):
            raise ValueError("This file already exists. Change 'filename' or set 'replace' to True")
        return filename


    def load_surf(self, h, k=32, MNINonLinear=False, type="midthickness", assign=True):
        """
        Load a subject's cortical surface. This method assumes the standard HCP directory structure.

        Parameters
        ----------
        h : str
            Hemisphere to load ('L' or 'R').
        k : int, optional
            Resolution of the cortical surface (default is 32).
        MNINonLinear : bool, optional
            If True, load the surface in the MNINonLinear HCP directory.
            Default is False (load from the T1w directory).
        type : str, optional
            Type of surface to load. Should be one of:
            - 'midthickness': the midthickness surface.
            - 'cortex_midthickness': the midthickness surface without the medial wall.
        assign : bool, optional
            If True, assigns the surface to the object's attributes. Default is True.

        Returns
        -------
        nibabel.gifti.GiftiImage
            The loaded surface file.
        """
        wrap = ["MNINonLinear" if MNINonLinear else "T1w"][0]
        surf = nib.load(getattr(self, f"{h}_{type}_{k}k_{wrap}"))

        if assign:
            return surf

        setattr(self, f"surf{k}k_{h}", surf)


    def load_grads(self, h=None, k=32, assign=True):
        """
        Load resting-state functional gradients of a subject.

        Parameters
        ----------
        h : str, optional
            Hemisphere to load ('L' or 'R'). If None, loads gradients for both hemispheres.
        k : int, optional
            Resolution of the gradients (default is 32).
        assign : bool, optional
            If True, assigns the gradients to the object's attributes. Default is True.

        Returns
        -------
        numpy.ndarray
            The loaded gradients.
        """

        if k==32:
            raise NotImplementedError("32k gradients are not implemented yet")
        if h:
            vinfo = hcp.vertex_info if k==32 else vertex_info_10k
            start = 0 if h=="L" else vinfo[f"gray{h.lower()}"].size
            stop = vinfo[f"gray{h.lower()}"].size if h=="L" else None
            gradients = np.load(
                self.outpath(f"{self.id}.REST_FC_embedding.npy")
                )[slice(start, stop), :]
        else:
            gradients = np.load(self.outpath(f"{self.id}.REST_FC_embedding.npy"))
            h = "LR"
        if assign:
            return gradients

        setattr(self, f"grad{k}k_{h}", gradients)


    def load_gdist_triu(self, hemi=None):
        """
        Load the vectorized upper triangle of the (symmetric) geodesic dstance matrix.

        Parameters
        ----------
        hemi : str
            Hemisphere to load ('L' or 'R').

        Returns
        -------
        numpy.ndarray
            A 1D vector containing the geodesic distances.
        """

        if hemi is None:
            raise TypeError("Please specify one hemisphere: 'L' or 'R'")
        
        filename = f"{dataset().output_dir}/{self.id}.{hemi}.gdist_triu.10k_fs_LR.npy"
        if os.exists(filename):
            return np.load(filename)
        else:
            raise ValueError(f"{filename} does not exist, generate it and try again.")


    def load_gdist_matrix(self, hemi=None, D=0):
        """
        Load the geodesic distance matrix of one hemisphere.

        Parameters
        ----------
        hemi : str
            Hemisphere to load ('L' or 'R').
        D : int, optional
            Value to fill the diagonal of the matrix. Default is 0.

        Returns
        -------
        numpy.ndarray
            The reconstructed geodesic distance matrix.
        """

        if hemi is None:
            raise TypeError("Please specify one hemisphere: 'L' or 'R'")
            
        v = self.load_gdist_triu(hemi)
        return reconstruct_matrix(v, diag_fill=D, k=1)
    
    def load_gdist_vertex(self, hemi=None, target=None, source=None):

        if hemi is None or target is None:
            raise TypeError("Please specify one hemisphere ('L' or 'R') and a vertex index")

        gdist_triu = subject(self.id).load_gdist_triu(hemi)
        gdist_v = row_from_triu(target, k=1, triu=gdist_triu)

        return gdist_v

    def load_embeddings(self, h, alg=None, return_bunch=True):
        """
        Load geometric embeddings for a specific hemisphere and algorithm.

        Parameters
        ----------
        h : str
            Hemisphere to load ('L' or 'R').
        alg : str or list of str, optional
            Algorithm(s) for which to load embeddings. Will load all embeddings whose keys start
            with the specified string(s). If None, all embeddings are loaded.
        return_bunch : bool, optional
            If True, returns a `Bunch` object, else the embedding(s) is(are) added to the dataset
            attributes. Default is True.

        Returns
        -------
        sklearn.utils.Bunch
            A `Bunch` object containing the embeddings for the specified hemisphere.
        """

        embeddings = create_bunch_from(dataset().outpath(f"All.{h}.embeddings.npz"))

        if alg is None:
            alg = embeddings.keys()
            attr_to_keep = alg
        elif isinstance(alg, str):
            alg = [alg]
            attr_to_keep = alg
        elif isinstance(alg, list):
            attr_to_keep = [attr for attr in embeddings.keys() for a in alg if attr.startswith(a)]

        embeddings = create_bunch_from({k: embeddings[k][self.idx] for k in attr_to_keep})

        if return_bunch:
            return embeddings

        setattr(self, f"embeds_{h}", embeddings)




# MISC
# ----
# Functions for various tasks

def shape_from_triu(n, k=0):
    """
    Compute the matrix dimension from the size of its upper triangular vector.

    Parameters
    ----------
    n : int
        Size of the upper triangular vector.
    k : int, optional
        Diagonal offset. Default is 0.

    Returns
    -------
    int
        Dimension of the square matrix.
    """

    return int(k + (np.sqrt(1 + 8 * n) -1) / 2)


def reconstruct_matrix(vector, diag_fill=0, k=0):
    """
    Reconstruct a symmetric matrix from its upper triangular vector.

    Parameters
    ----------
    vector : numpy.ndarray
        The upper triangular vector of the matrix.
    diag_fill : int or float, optional
        Value to fill the diagonal. Default is 0.
    k : int, optional
        Diagonal offset of the upper triangular vector. Default is 0.

    Returns
    -------
    numpy.ndarray
        The reconstructed symmetric matrix.
    """

    m = shape_from_triu(vector.size, k=k)
    matrix = np.zeros([m, m])
    matrix[np.triu_indices_from(matrix, k)] = vector
    matrix[np.tril_indices_from(matrix, -k)] = matrix.T[np.tril_indices_from(matrix,-k)]
    if diag_fill != 0:
        np.fill_diagonal(matrix, diag_fill)

    return matrix


def euclidean_triu(X, k=1):
    """
    Compute the pairwise Euclidean distances for the upper triangular matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Input matrix.
    k : int, optional
        Diagonal offset. Default is 1.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the pairwise distances.
    """

    r, c = np.triu_indices(X.shape[0], k)
    return abs(X[c] - X[r]).squeeze()


def diagmat_triu_idx(M, n, k=0):
    """
    Extract the diagonal blocks from an upper triangular matrix.

    Parameters
    ----------
    M : numpy.ndarray or int
        Input matrix or the size of the matrix.
    n : int
        Block size.
    k : int, optional
        Diagonal offset. Default is 0.

    Returns
    -------
    numpy.ndarray or tuple
        Diagonal blocks or their indices.
    """

    if isinstance(M, np.ndarray):
        m = M.shape[0]

    elif isinstance(M, (int, np.int32)):
        m = M

    else:
        raise TypeError("M should be a numpy.ndarray or an integer")

    r = m//n
    indices = np.tile(np.triu_indices(n, k), [1,r])
    indices = indices + np.repeat(range(r), len(indices.T)/r) * n

    if isinstance(M, np.ndarray):
        diag_triu = np.zeros(M.shape)
        diag_triu[indices[0], indices[1]] = M[indices[0], indices[1]]
        return diag_triu

    return indices


def row_from_triu(row, k=0, n=None, triu=None, include_diag=True, diag_fill=0):
    """
    Reconstruct a row of a symmetric matrix from its upper triangular vector.
    Returns either the row values or the row indices in the vector.

    Parameters
    ----------
    row : int
        Index of the row to extract.
    k : int, optional
        Diagonal offset used when generating the vector. Default is 0.
    n : int, optional
        Matrix size. If not provided, derived from `triu`.
    triu : numpy.ndarray, optional
        Upper triangular vector of the matrix. If None, returns indices for the vector
    include_diag : bool, optional
        Whether to include the diagonal value. Default is True.
    diag_fill : int or float, optional
        Value to fill the diagonal if `include_diag` is TRUE. Default is 0.

    Returns
    -------
    numpy.ndarray
        Extracted row values or their indices in the upper triangular vector.
    """

    # If given a triu vector get original matrix shape and index
    if (n is None) & isinstance(triu, np.ndarray):
        n = shape_from_triu(triu.size, k=k)

    col = np.arange(n)
    col = col[~np.isin(col, range(row - (k-1), row + k))]

    a = 2*n - 3
    b = col + (-col**2 + a*col) / 2 + col
    c = row + (-row**2 + a*row) / 2 + row

    if k==0:
        row_idx = (b + row - col) * (col <= row) + (c + col-row) * (col > row)
    elif k == 1:
        row_idx = ((b + row - col - (col+k)) * (col <= row)) + ((c + col-row - row-k) * (col > row))
    else:
        raise ValueError("Diagonal offsets k > 1 are not implemented")

    row_idx = np.int32(row_idx)

    if isinstance(triu, np.ndarray):
        row_vals = triu[row_idx]
        if include_diag & k==1:
            row_vals = np.insert(row_vals, row, diag_fill)
        return  row_vals


    return np.int32(row_idx)


def vector_wise_corr(A, B):
    """
    Compute column-wise correlation coefficients between two matrices.

    Parameters
    ----------
    A : numpy.ndarray
        First input matrix.
    B : numpy.ndarray
        Second input matrix.

    Returns
    -------
    numpy.ndarray
        Correlation coefficients for each column.
    """

    # center the matrices
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)

    # Calculate correlation coefficients
    return np.sum(A * B, axis=0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0))


def npz_update(filename, items={}):
    """
    Update an existing `.npz` file with new data or create a new file.

    Parameters
    ----------
    filename : str
        Path to the `.npz` file.
    items : dict, optional
        Key-value pairs to update or add to the `.npz` file. Default is an empty dictionary.
    """

    if os.exists(filename):
        npz = dict(np.load(filename))
        npz.update(items)
        np.savez(filename, **npz)

    else:
        np.savez(filename, **items)



def bins_ol(xmin, xmax, nbins=10, overlap=0.25, inclusive=True):
    """
    Define equally spaced, overlapping bins.
    
    Parameters
    ----------
    xmin, xmax: float or scalar
        the extremes of the span of values to bin
    nbins : scalar
        number of bins to divide the values in
    overlap : float
        the fraction of overlap between bins. Must be between -0.5 and 0.5 (Default=0.25)
        Negative values will result in disjoint bins.
    inclusive : bool
        if True, the bounds of the bins will be the centers of the outer bins.
        If False, the bounds will be the edges of the outer bins.
    
    Returns:
    --------
    lower:
        the lower bound of every bin
    upper:
        the upper bound ov every bin

    """
    if overlap < -0.5 or overlap > 0.5:
        raise ValueError("'overlap' should be between -0.5 and 0.5")

    span = xmax - xmin

    if inclusive:
        step = span / nbins
        center = np.arange(xmin, xmax + step, step)
        half_window = step * 0.5  +  step * overlap

        lower = center - half_window
        upper = center + half_window

    else:
        ratio = nbins * (1 - 2 * overlap) + (nbins + 1) * overlap

        window = span / ratio
        step = window * (1 - overlap)

        lower = np.arange(xmin, xmax, step)[:nbins]
        upper = lower + window
            
    return lower, upper
        