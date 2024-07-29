# core_utils.py

import nibabel as nib
import numpy as np
import os.path as os
from sklearn.utils import Bunch
from variograd_utils import *
from itertools import combinations

def create_bunch_from(A):
    if isinstance(A, str):
        return Bunch(**{key: value for key, value in np.load(A).items()})
    elif isinstance(A, dict):
        return Bunch(**A)
    elif isinstance(A, np.lib.npyio.NpzFile):
        return Bunch(**{key: value for key, value in A.items()})
    else:
        raise TypeError("'A' should be a dictionary, a numpy NpzFile, or string indicating the path to a NpzFile")






class dataset:

    def __init__(self):
        file = open('./variograd_utils/directories.txt','r')
        directories = {line.split("=")[0]: line.split("=")[1].replace("\n", "") for line in file}
        self.group_dir = directories["group_dir"]
        self.subj_dir = directories["subj_dir"]
        self.output_dir = directories["output_dir"]
        self.utils_dir = directories["work_dir"] + "/variograd_utils"
        self.mesh10k_dir = directories["mesh10k_dir"]
        
        self.subj_list = np.loadtxt(directories["subj_list"]).astype("int32")
        self.N = len(self.subj_list)
        self.id = f"{self.N}avg"
        self.pairs = list(combinations(self.subj_list, 2))

        surf_args = np.array(np.meshgrid(["L", "R"], [10, 32], ["midthickness", "cortex_midthickness"]), 
                             dtype="object").T.reshape(-1, 3)
        for h, k, name in surf_args:
            if k==32:
                path = f"{self.group_dir}/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii"
            elif k==10:
                path = f"{self.mesh10k_dir}/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii"

            setattr(self, f"{h}_{name}_{k}k",  path)


    def outpath(self, filename, replace=True):
        filename = f"{self.output_dir}/{filename}"
        if os.exists(filename) & ~replace:
            raise ValueError("This file already exists. Change 'filename' or set 'replace' to True")
        return filename
    

    def load_surf(self, h, k=32, type="midthickness", assign=True):

        if k==32:
            surf = nib.load(f"{self.group_dir}/{self.id}.{h}.{type}_MSMAll.{k}k_fs_LR.surf.gii")
        elif k==10:
            surf = nib.load(f"{self.mesh10k_dir}/{self.id}.{h}.{type}_MSMAll.{k}k_fs_LR.surf.gii")
        
        if assign:
            return surf
        
        setattr(self, f"surf{k}k_{h}", surf)
    


    def load_embeddings(self, h, alg=None, return_bunch=True):

        embeddings = create_bunch_from(self.outpath(f"All.{h}.embeddings.npz"))

        if alg:
            if isinstance(alg, str):
                alg = [alg]

            attr_to_keep = [attr for attr in embeddings.keys() for a in alg if attr.startswith(a)]
            embeddings = create_bunch_from({k: embeddings[k] for k in attr_to_keep})

        if return_bunch:
            return embeddings
            
        setattr(self, f"embeds_{h}", embeddings)

    

    def allign_embeddings(self, h=None, alg=None, overwrite=True):
        h = [["L", "R"] if h is None else [h]][0]

        for hemi in h:
            embeddings = self.load_embeddings(hemi, alg)

            for key, val in embeddings.items():
                ref = val[0]

                for v in val:
                    inverted = vector_wise_corr(ref, v) < 0
                    v[:,inverted] *= -1

            npz_update(self.outpath(f"All.{hemi}.embeddings.npz"), embeddings)


    

    def generate_avg_surf(self, h, k=32, assign=False, save=True, filename=None):
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

            save_gifti(darrays=avg_surf, intents=[1008, 1009], dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"], 
                    filename=filename, structure=structure)

        if assign:
            return avg_surf
        
    
    def load_gdist_triu(self, hemi=None):
        if hemi is None:
                raise TypeError("Please specify one hemisphere: 'L' or 'R'")
        
        filename = f"{self.output_dir}/{self.id}.{hemi}.gdist_triu.10k_fs_LR.npy"
        if os.exists(filename):
            return np.load(filename)
        else:
            raise ValueError(f"{filename} does not exist, generate it and try again.")


    def load_gdist_matrix(self, hemi=None, D=0):

        if hemi is None:
            raise TypeError("Please specify one hemisphere: 'L' or 'R'")
            
        v = self.load_gdist_triu(hemi)
        return reconstruct_matrix(v, diag_fill=0, k=1)






        
        
class subject:

    def __init__(self, ID=None):
        data = dataset()
        self.id = ID
        self.idx = np.argwhere(data.subj_list==ID).squeeze()
        self.dir = f"{data.subj_dir}/{ID}"

        surf_args = np.array(np.meshgrid(["L", "R"], [10, 32], ["MNINonLinear","T1w"], ["midthickness", "cortex_midthickness"]), 
                             dtype="object").T.reshape(-1, 4)
        for h, k, w, name in surf_args:
            path = f"{self.dir}/{w}/fsaverage_LR{k}k/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii"
            if os.exists(path):
                setattr(self, f"{h}_{name}_{k}k_{w}",  path)
                
            path = self.outpath(f"{w}/fsaverage_LR{k}k/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii")
            if os.exists(path):
                setattr(self, f"{h}_{name}_{k}k_{w}",  path)
            

    def outpath(self, filename, replace=True):
        filename = f"{dataset().output_dir}/{self.id}/{filename}"
        if os.exists(filename) & ~replace:
            raise ValueError("This file already exists. Change 'filename' or set 'replace' to True")
        return filename


    def load_surf(self, h, k=32, MNINonLinear=False, type="midthickness", assign=True):
        wrap = ["MNINonLinear" if MNINonLinear else "T1w"][0]
        surf = nib.load(getattr(self, f"{h}_{type}_{k}k_{wrap}"))
        
        if assign:
            return surf
        
        setattr(self, f"surf{k}k_{h}", surf)


    def load_grads(self, h=None, k=32, assign=True):
        if h:
            gradients = nib.load(self.outpath(f"/Analysis/{self.id}.{h}.DME_1.{k}k_fs_LR.shape.gii")).darrays[0].data
        else:
            Gl = nib.load(self.outpath(f"/Analysis/{self.id}.L.DME_1.{k}k_fs_LR.shape.gii")).darrays[0].data
            Gr = nib.load(self.outpath(f"/Analysis/{self.id}.R.DME_1.{k}k_fs_LR.shape.gii")).darrays[0].data
            gradients = np.hstack([Gl, Gr])
            h = "LR"

        if assign:
            return gradients
        
        setattr(self, f"grad{k}k_{h}", gradients)


    def load_gdist_triu(self, hemi=None):
        if hemi is None:
                raise TypeError("Please specify one hemisphere: 'L' or 'R'")
        
        filename = f"{dataset().output_dir}/{self.id}.{hemi}.gdist_triu.10k_fs_LR.npy"
        if os.exists(filename):
            return np.load(filename)
        else:
            raise ValueError(f"{filename} does not exist, generate it and try again.")


    def load_gdist_matrix(self, hemi=None):

        if hemi is None:
            raise TypeError("Please specify one hemisphere: 'L' or 'R'")
            
        v = self.load_gdist_triu(hemi)
        return reconstruct_matrix(v, diag_fill=0, k=1)
    
    def load_gdist_vertex(self, hemi=None, target=None, source=None):

        if hemi is None or target is None:
            raise TypeError("Please specify one hemisphere ('L' or 'R') and a vertex index")

        gdist_triu = subject(self.id).load_gdist_triu(hemi)
        gdist_v = row_from_triu(target, k=1, triu=gdist_triu)

        return gdist_v
    
    def load_embeddings(self, h, alg=None, return_bunch=True):

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

def shape_from_triu(n, k=0):
    return int(k + (np.sqrt(1 + 8 * n) -1) / 2)


def reconstruct_matrix(vector, diag_fill=0, k=0):
    m = shape_from_triu(vector.size, k=k)
    matrix = np.zeros([m, m])
    matrix[np.triu_indices_from(matrix, k)] = vector
    matrix[np.tril_indices_from(matrix, -k)] = matrix.T[np.tril_indices_from(matrix,-k)]
    if diag_fill != 0:
        np.fill_diagonal(matrix, diag_fill)

    return matrix


def euclidean_triu(X, k=1, n_dim=False):
    r, c = np.triu_indices(X.shape[0], k)
    return abs(X[c] - X[r]).squeeze()


def diagmat_triu_idx(M, n, k=0):
    if isinstance(M, np.ndarray):
        m = M.shape[0]

    elif isinstance(M, (int, np.int32)):
        m = M

    r=m//n
    indices = np.tile(np.triu_indices(n, k), [1,r])
    indices = indices + np.repeat(range(r), len(indices.T)/r) * n

    if isinstance(M, np.ndarray):
        diag_triu = np.zeros(M.shape)
        diag_triu[indices[0], indices[1]] = M[indices[0], indices[1]]
        return diag_triu
    
    return indices


def row_from_triu(row, k=0, n=None, triu=None, include_diag=True, diag_fill=0):

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

    # if A.shape != B.shape:
    #     raise ValueError("x and y must have the same shape")

    # mean-center from matrices
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)

    # Calculate correlation coefficients
    return np.sum(A * B, axis=0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0))


def npz_update(filename, items={}):

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
        