# core_utils.py

import os
import json
import h5py
from itertools import combinations
import numpy as np
from sklearn.utils import Bunch
import nibabel as nib
import hcp_utils as hcp
import variograd_utils
from variograd_utils.brain_utils import save_gifti, vertex_info_10k


class dataset:
    """
    Manage group-level data, subject-level data, and common operations for a specific dataset.

    This class initializes a dataset object based on a JSON configuration file that 
    contains paths for different datasets. It supports various operations for managing
    and analyzing subject and group-level data.

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
        List of subject IDs loaded from the specified subject list file.
    N : int
        Number of subjects in the dataset.
    id : str
        Identifier for the dataset, typically based on the number of subjects (e.g., "{N}avg").
    pairs : list
        List of all subject pairs for pairwise computations.
    L_midthickness_32k : str
        Path to the left hemisphere 32k-resolution midthickness surface file.
    R_midthickness_32k : str
        Path to the right hemisphere 32k-resolution midthickness surface file.
    L_cortex_midthickness_32k : str
        Path to the left hemisphere 32k-resolution cortex midthickness surface file.
    R_cortex_midthickness_32k : str
        Path to the right hemisphere 32k-resolution cortex midthickness surface file.
    L_midthickness_10k : str
        Path to the left hemisphere 10k-resolution midthickness surface file.
    R_midthickness_10k : str
        Path to the right hemisphere 10k-resolution midthickness surface file.
    L_cortex_midthickness_10k : str
        Path to the left hemisphere 10k-resolution cortex midthickness surface file.
    R_cortex_midthickness_10k : str
        Path to the right hemisphere 10k-resolution cortex midthickness surface file.

    Parameters
    ----------
    dataset_id : str
        Identifier for the dataset. This must correspond to an entry in the JSON file
        containing dataset-specific paths.

    Methods
    -------
    outpath(filename, replace=True)
        Generates an output file path, optionally checking for existing files.
    load_surf(h, k=32, type="midthickness", assign=True)
        Loads the group average cortical surface for the specified hemisphere and resolution.
    load_embeddings(h, alg=None, return_bunch=True)
        Loads geometric embeddings for the specified hemisphere and algorithm.
    allign_embeddings(h=None, alg=None, overwrite=True)
        Aligns embeddings by flipping signs based on reference embeddings.
    generate_avg_surf(h, k=32, assign=False, save=True, filename=None)
        Computes the average cortical surface for a hemisphere and optionally saves it.
    load_gdist_triu(hemi=None)
        Loads geodesic distances as a vector for the specified hemisphere.
    load_gdist_matrix(hemi=None, D=0)
        Reconstructs a geodesic distance matrix for the specified hemisphere.

    Raises
    ------
    ValueError
        If an unsupported resolution is specified for surface paths (only 10k and 32k are allowed).
        If the dataset_id does not exist in the JSON file.

    Notes
    -----
    - The JSON configuration file must exist in the module's directory and include an entry
      for each dataset, containing the required paths: "group_dir", "subj_dir", "output_dir",
      "mesh10k_dir", and "subj_list".
    - Subject IDs are loaded from the "subj_list" path specified in the JSON file.
    - Surface paths for midthickness and cortex midthickness meshes are dynamically generated
      based on the dataset configuration.
    """

    def __init__(self, dataset_id, **kwargs):
        """
        Initialize the `dataset` object with paths and attributes for a specific dataset.

        This constructor reads dataset-specific paths a JSON file and dynamically generates
        paths for surface files and computes basic subject-related attributes.
    
        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset in the JSON configuration file.
        **kwargs : variogram_utils.core_utils.init_dataset() arguments.
            **kwargs are used to locally override paths from the JSON file
    
        Attributes Initialized
        -----------------------
        group_dir : str
            Directory containing group-level data, read from the JSON file.
        subj_dir : str
            Directory containing subject-level data, read from the JSON file.
        output_dir : str
            Directory for saving outputs, read from the JSON file.
        utils_dir : str
            Directory where the module is installed.
        mesh10k_dir : str
            Directory containing 10k-resolution meshes, read from the JSON file.
        subj_list : numpy.ndarray
            Array of subject IDs loaded from the path specified in the JSON file.
        N : int
            Number of subjects in the dataset, derived from the length of `subj_list`.
        id : str
            Dataset identifier, generated as "{N}avg", where `N` is the number of subjects.
        pairs : list
            List of all subject pairs for pairwise computations.
        <surface_path_attributes> : str
            Dynamically created attributes for surface file paths based on the dataset.
    
        Raises
        ------
        ValueError
            If the provided `dataset_id` does not exist in the JSON file.
    
        Notes
        -----
        - The JSON file (`directories.json`) can be initialized and edited using init_dataset().
        - Surface paths are dynamically generated and added as attributes.
        """

        pkg_path = os.path.dirname(variograd_utils.__file__)
        json_path = f'{pkg_path}/directories.json'
        with open(json_path, "r") as json_file:
            directories = json.load(json_file)

        if dataset_id not in directories.keys():
            raise ValueError("There is no datased with this ID. Use init_dataset() to add it first.")

        directories = {**directories[dataset_id], **kwargs}
        for attribute, value in directories.items():
            setattr(self, attribute, value)

        self.subj_list = np.loadtxt(self.subj_list, dtype="int32")
        self.N = len(self.subj_list)
        self.id = dataset_id #f"{self.N}avg"
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
        if os.path.exists(filename) & (not replace):
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

        pointsets, triangles = subject(self.subj_list[0], self.id).load_surf(h, k=k).darrays
        pointsets = pointsets.data
        triangles = triangles.data

        for ID in self.subj_list[1:]:
            pointsets += subject(ID, self.id).load_surf(h, k=k).darrays[0].data
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
        if os.path.exists(filename):
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

    def __init__(self, ID=None, dataset_id=None):
        """
        Initialize the `subject` object.
        """

        data = dataset(dataset_id)
        self.dataset_id = dataset_id
        self.id = ID
        self.idx = np.argwhere(data.subj_list==ID).squeeze()
        self.dir = f"{data.subj_dir}/{ID}"

        surf_args = np.array(np.meshgrid(["L", "R"], [10, 32],
                                         ["MNINonLinear","T1w"],
                                         ["midthickness", "cortex_midthickness"]),
                                         dtype="object").T.reshape(-1, 4)
        for h, k, w, name in surf_args:
            path = f"{self.dir}/{w}/fsaverage_LR{k}k/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii"
            if os.path.exists(path):
                setattr(self, f"{h}_{name}_{k}k_{w}",  path)

            path = self.outpath(f"{w}/fsaverage_LR{k}k/{self.id}.{h}.{name}_MSMAll.{k}k_fs_LR.surf.gii")
            if os.path.exists(path):
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

        filename = f"{dataset(self.dataset_id).output_dir}/{self.id}/{filename}"
        if os.path.exists(filename) & (not replace):
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
        
        filename = self.outpath(f"{self.id}.{hemi}.gdist_triu.10k_fs_LR.npy")
        if os.path.exists(filename):
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

        gdist_triu = subject(self.id, self.dataset_id).load_gdist_triu(hemi)
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

        embeddings = create_bunch_from(dataset(self.dataset_id).outpath(f"All.{h}.embeddings.npz"))

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


def init_dataset(dataset_id=None, group_dir=None, subj_dir=None,
                 output_dir=None, mesh10k_dir=None, subj_list=None
):
    """
    Create or update a dataset definition in a JSON file within the package directory.

    Parameters
    ----------
    dataset_id : str, optional
        Unique identifier for the dataset. If None or empty, an error is raised.
    group_dir : str, optional
        Directory containing group-level data.
    subj_dir : str, optional
        Directory containing subject-level data.
    output_dir : str, optional
        Directory for saving outputs.
    mesh10k_dir : str, optional
        Directory containing 10k-resolution meshes.
    subj_list : str, optional
        Path to the subject list file.

    Raises
    ------
    ValueError
        If dataset_id is not provided (None or empty).
        If dataset_id is new and any of the paths are missing.
    """

    arguments = locals()

    # 1. Determine the package directory and JSON file path
    pkg_path = os.path.dirname(variograd_utils.__file__)
    json_path = os.path.join(pkg_path, 'directories.json')

    # 2. Check if dataset_id is provided
    if not dataset_id:
        raise ValueError("You must provide a dataset_id.")

    # 3. Load existing data (or start fresh if file doesn't exist)
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # 4. Check if this is a new dataset ID
    required = ["group_dir", "subj_dir", "output_dir", "mesh10k_dir", "subj_list"]
    is_new = dataset_id not in data
    
    if is_new:
        # If it's new, ensure none of the important paths are missing
        missing = [field for field in required if arguments[field] is None]
        if missing:
            raise ValueError(f"Missing  arguments required for creating a new dataset: "
                             + ", ".join(missing))
        data[dataset_id] = {field: arguments[field] for field in required}
        
    else:
        data[dataset_id] = {field: (value if arguments[field] is None else arguments[field])
                            for field, value in data[dataset_id].items()}

    # 6. Write back to JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    # 7. Print info
    if is_new:
        print(f"New dataset '{dataset_id}' created in '{json_path}'.")
    else:
        print(f"Dataset '{dataset_id}' has been updated in '{json_path}'.")


# Dict-like data handling utilities
# ---------------------------------
def print_structure(d, indent=0):
    """
    Recursively prints the structure of a nested dictionary.

    This function traverses through all levels of a dictionary and prints each key
    along with the type of its associated value. For values that are dictionaries,
    the function calls itself recursively with an increased indentation level.

    Parameters
    ----------
    d : dict
        The dictionary whose structure is to be printed.
    indent : int, optional
        The current indentation level (default is 0). This is used internally
        for formatting the output.

    Returns
    -------
    None
    """
    for key, value in d.items():
        print(' ' * indent + f"{key}: {type(value).__name__}")
        if isinstance(value, dict):
            print_structure(value, indent + 4)


def save_hdf5(d, filename):
    """
    Recursively saves a dictionary of dictionaries (with numpy arrays at the end)
    to an HDF5 file.

    Parameters
    ----------
    d : dict
        The dictionary to save. Nested dictionaries will be stored as groups.
        Numpy arrays (or array-like data) will be stored as datasets.
    filename : str
        The name of the HDF5 file to create.

    Returns
    -------
    None
    """
    
    with h5py.File(filename, 'w') as h5file:
        _rec_save(h5file, d)


def _rec_save(h5group, dict_obj):
    for key, item in dict_obj.items():
        if isinstance(item, dict):
            # Create a new group for nested dictionaries
            if not isinstance(key, str): key = str(key)
            subgroup = h5group.create_group(key)
            _rec_save(subgroup, item)
        else:
            # Save arrays or other data (must be convertible to numpy array)
            h5group.create_dataset(key, data=np.array(item))


def load_hdf5(filename):
    """
    Recursively loads an HDF5 file into a dictionary of dictionaries.
    
    The function reads groups as dictionaries and datasets as numpy arrays.
    It expects that the HDF5 file was written with a similar structure, where
    nested dictionaries are stored as groups and arrays or array-like data are
    stored as datasets.
    
    Parameters
    ----------
    filename : str
        The name of the HDF5 file to load.
    
    Returns
    -------
    dict
        A dictionary containing the data from the HDF5 file, preserving the 
        nested structure.
    """

    with h5py.File(filename, 'r') as h5file:
        return _rec_load(h5file)


def _rec_load(h5obj):
    ans = {}
    # Iterate over all items in the current group or file
    for key, item in h5obj.items():
        if isinstance(item, h5py.Group):
            # If the item is a group, recursively load it into a dict
            ans[key] = _rec_load(item)
        elif isinstance(item, h5py.Dataset):
            # If the item is a dataset, read its value (as a numpy array)
            ans[key] = item[()]
    return ans


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

    if os.path.exists(filename):
        npz = dict(np.load(filename))
        npz.update(items)
        np.savez(filename, **npz)

    else:
        np.savez(filename, **items)


# Matrix operations
# -----------------
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
        Input matrix of shape observations x features.
    k : int, optional
        Diagonal offset. Default is 1.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the pairwise distances.
    """

    r, c = np.triu_indices(X.shape[0], k)
    D = abs(X[c] - X[r]).squeeze()
    D = np.sqrt(np.sum(D**2, axis=1))
    return D


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
        