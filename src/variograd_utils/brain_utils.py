# brain_utils

import os.path as os
from subprocess import run
import numpy as np
from scipy.stats import rankdata
from sklearn.utils import Bunch
import nibabel as nib
from nibabel import save, gifti, nifti1, cifti2


def save_dscalar(scalars, template_cifti, output_cifti, names=None):
    """
    Save 2D np.array to a dscalar file.

    Saves an M x N numpy array to a .dscalar.nii cifti file.
    

    Parameters
    ----------
    scalars : numpy.array
        M x N array of ints or floats.
    template_cifti : nibabel.cifti2.cifti2.Cifti2Image
        Cifti2Image with N greyordinates
    output_cifti : str
        Path of the output dscalar.nii file
    Names : list of str or None
        List of M names to be assigned to the ScalarAxis.
        If None is np.arange(M).
    """

    if scalars.ndim == 1:
        scalars = np.atleast_2d(scalars)

    data = np.zeros([scalars.shape[0],template_cifti.shape[1]])
    data[0:,0:scalars.shape[1]] = scalars

    if names is None:
        map_labels = np.arange(scalars.shape[0])+1
    else:
        map_labels = names
    ScalarAxis = cifti2.cifti2_axes.ScalarAxis(map_labels)
    BrainModelAxis = cifti2.cifti2_axes.from_index_mapping(template_cifti.header.get_index_map(1))
    nifti_hdr = template_cifti.nifti_header
    del template_cifti

    new_img = nib.Cifti2Image(data, header=[ScalarAxis, BrainModelAxis],nifti_header=nifti_hdr)
    new_img.update_headers()

    save(new_img, output_cifti)


def save_dlabel(labels, label_dicts, template_cifti, output_cifti):
    """
    Save 1D np.array to .dlabel.nii file.

    Saves an M x N numpy array to a .dlabel.nii cifti file.
    

    Parameters
    ----------
    labels : numpy.array
        (N,) array of ints.
    label_dicts : dict of dicts
        dictionary mapping M parcellation names to dictionaries
        the parcel indices of that parcellation to the corresponding
        (parcel_name, (R, G, B, A)) tuple.
        R, G, B, and A are floats between 0 and 1.
    template_cifti : nibabel.cifti2.cifti2.Cifti2Image
        Cifti2Image with N greyordinates
    output_cifti : str
        Path of the output dlabel.nii file
    """

    if labels.ndim==1:
        labels = np.atleast_2d(labels)

    label_names = [key for key in label_dicts.keys()]
    label_dicts = [val for key, val in label_dicts.items()]
    LabelAxis = cifti2.cifti2_axes.LabelAxis(label_names, label_dicts)
    BrainModelAxis = cifti2.cifti2_axes.from_index_mapping(template_cifti.header.get_index_map(1))
    nifti_hdr = template_cifti.nifti_header
    del template_cifti

    new_img = nib.Cifti2Image(labels, header=[LabelAxis, BrainModelAxis],nifti_header=nifti_hdr)
    new_img.update_headers()

    save(new_img, output_cifti)


def _prep_GiftiDataArray_args(darrays=None, intents=None, dtypes=None, metas=None):
    """
    Prepare arguments for creating `GiftiDataArray` objects.

    Helper function to prepare the `darrays`, `intents`, `dtypes`, and `metas` arguments
    to be used to initialize the nibabel.gifti.GiftiDataArray object.

    Parameters
    ----------
    darrays : numpy.ndarray or list of numpy.ndarray
        The data arrays to include in the `GiftiDataArray`. Must be specified.

    intents : int, float, str, or list, optional
        NIFTI intent code(s) for the `GiftiDataArray`. Can be a single value (applied to all
        `darrays`) or a list matching the number of `darrays`. See `nibabel.nifti1.intent_codes`.

    dtypes : numpy.dtype or list, optional
        NIFTI data type code(s) for the `GiftiDataArray`. Can be a single value (applied to all
        `darrays`) or a list matching the number of `darrays`. See nifti1.data_type_codes

    metas : dict or list of dicts, optional
        Metadata for the `GiftiDataArray`. Can be a single dictionary (applied to all `darrays`) 
        or a list matching the number of `darrays`.

    Returns
    -------
    tuple
        A tuple containing:
        - list: The input `darrays` wrapped in a list.
        - list: Prepared `intents`, where each value corresponds to a `darray`.
        - list: Prepared `dtypes`, where each value corresponds to a `darray`.
        - list: Prepared `metas`, where each value corresponds to a `darray`.

    """


    if darrays is None:
        raise TypeError("at least one darray must be specified")
    elif isinstance(darrays, np.ndarray):
        darrays = [darrays]

    if dtypes is None:
        dtypes = [darray.dtype for darray in darrays]

    prepped_args = (darrays,)
    N = len(darrays)

    arg_names = ["intents", "dtypes", "metas"]

    for i, arg in enumerate([intents, dtypes, metas]):
        accepted_types = (int, float, str, np.int32, np.int64, np.float64, np.float32, dict)
        if isinstance(arg, list):
            prepped_args += (arg,)
            continue
        elif (isinstance(arg, accepted_types)) | (arg is None):
            prepped_args += ([arg for _ in range(N)],)
        else:
            raise TypeError(f"'{arg_names[i]}' is not the correct data type, "
                            + "consult documentation to find the compatible ones.")

    return prepped_args


def save_gifti(darrays=None, intents=0, dtypes=None, structure=None, filename=None,
               metas=None, coordsys=None, encoding='GIFTI_ENCODING_B64GZ', endian='little' ):
    """
    Save NumPy arrays as a GIFTI file.

    This function generates a new GIFTI file from NumPy arrays, specifying their intents, 
    data types, metadata, and other attributes. Optionally, it can assign a WorkBench
    brain structure to the file using `wb_command`.

    Parameters
    ----------
    darrays : numpy.ndarray or list of numpy.ndarray
        The data arrays to include in the `GiftiDataArray`. Must be specified.

    intents : int, float, str, or list, optional
        NIFTI intent code(s) for the `GiftiDataArray`. Can be a single value (applied to all
        `darrays`) or a list matching the number of `darrays`. See `nibabel.nifti1.intent_codes`.

    dtypes : numpy.dtype or list, optional
        NIFTI data type code(s) for the `GiftiDataArray`. Can be a single value (applied to all
        `darrays`) or a list matching the number of `darrays`. See nifti1.data_type_codes

    structure : str, optional
        The brain structure to assign to the GIFTI file (e.g., `"CORTEX_LEFT"`). 
        See ` wb_command -set-structure`.

    filename : str
        The file path to which the GIFTI file will be saved. Must be specified.

    metas : dict or list of dicts, optional
        Metadata for the `GiftiDataArray`. Can be a single dictionary (applied to all `darrays`) 
        or a list matching the number of `darrays`.

    coordsys : gifti.GiftiCoordSystem, optional
        The coordinate system to assign to the GIFTI arrays.See nibabel.gifti.GiftiCoordSystem.

    encoding : str, optional
        Encoding of the data, see `nibabel.util.gifti_encoding_codes`.
        Default is `GIFTI_ENCODING_B64GZ`.

    endian : str, optional
        The endianness of the GIFTI file. Default is `'little'`.

    Raises
    ------
    TypeError
        If `darrays`, `intents`, `dtypes`, or `metas` are specified as lists but do not 
        have the same number of items.
        If `filename` is not specified.

    """

    darrays, intents, dtypes, metas = _prep_GiftiDataArray_args(darrays, intents, dtypes, metas)

    len_specified = [len(arg) for arg in [darrays, intents, dtypes, metas] if arg is not None]
    if ~np.all(len_specified):
        raise TypeError("If darrays, intents, dtypes, and metas are specified as lists, "
                        + "they should all have  the same number of items.")

    image = gifti.GiftiImage()
    for darray, intent, dtype, meta in zip(darrays, intents, dtypes, metas):
        darray = gifti.GiftiDataArray(data=darray,
                                      intent=intent,
                                      datatype=nifti1.data_type_codes[dtype],
                                      coordsys=coordsys,
                                      encoding=encoding,
                                      endian=endian,
                                      meta=meta)
        image.add_gifti_data_array(darray)

    save(image, filename)

    if structure:
        command = f"wb_command -set-structure {filename} '{structure}'"
        run(command, shell=True)


# indices, number , and offset of cortex vertices in LR10k
vertex_info_filename = f"{os.dirname(os.realpath(__file__))}/fMRI_vertex_info_10k.npz"
args = dict(np.load(vertex_info_filename).items())
vertex_info_10k = Bunch(**args)
del vertex_info_filename


def cortex_data_10k(arr, fill=0, vertex_info=vertex_info_10k):
    """
    Add medial wall to array of cortex-only vertex values.

    This function takes a 1D array of fMRI grayordinates and returns the values
    on the vertices of the full 10k cortex mesh, including both left and right hemispheres. 
    The vertices of the medial wall are filled with a constant.

    Parameters
    ----------
    arr : numpy.ndarray
        The input cortical data, expected to correspond to gray matter vertices. 

    fill : int or float, optional
        The value to use for filling vertices not included in the cortical data. Default is 0.

    vertex_info : object, optional
        Vertex information object containing indices of gray matter vertices 
        and the total number of vertices for the left and right hemispheres. 
        Default is `vertex_info_10k`.

    Returns
    -------
    numpy.ndarray
        The full 10k cortical surface data, including both left and right hemispheres. 
        Filled values are assigned to vertices outside the cortical data.

    Notes
    -----
    This implementation adapts similar functionality provided by the `hcp_utils` to a 10k
    surface. For the original implementation and additional context, refer to:
    https://github.com/rmldj/hcp-utils.git

    """

    if arr.ndim==1:
        arr = arr.reshape([1, -1])

    out = np.zeros([arr.shape[0], vertex_info.num_meshl + vertex_info.num_meshr])
    out[:, :] = fill
    graylr = np.hstack([vertex_info.grayl, vertex_info.grayr + vertex_info.num_meshl])
    out[:, graylr] = arr

    return out.squeeze()


def left_cortex_data_10k(arr, fill=0, vertex_info=vertex_info_10k):
    """
    Add medial wall to array of cortex-only vertex values.

    This function takes a 1D array of fMRI grayordinates and returns the values
    on the vertices of the left 10k cortex mesh, including both left and right hemispheres. 
    The vertices of the medial wall are filled with a constant.

    Parameters
    ----------
    arr : numpy.ndarray
        The input cortical data, expected to correspond to gray matter vertices.

    fill : int or float, optional
        The value to use for filling vertices not included in the cortical data. Default is 0.

    vertex_info : object, optional
        Vertex information object containing indices of gray matter vertices 
        and the total number of vertices for the left and right hemispheres. 
        Default is `vertex_info_10k`.

    Returns
    -------
    numpy.ndarray
        The full 10k cortical surface data, including both left and right hemispheres. 
        Filled values are assigned to vertices outside the cortical data.

    Notes
    -----
    This implementation adapts similar functionality provided by the `hcp_utils` to a 10k
    surface. For the original implementation and additional context, refer to:
    https://github.com/rmldj/hcp-utils.git

    """

    if arr.ndim==1:
        arr = arr.reshape([1, -1])

    out = np.zeros([arr.shape[0], vertex_info.num_meshl])
    out[:, :] = fill
    out[:, vertex_info.grayl] = arr[:, :len(vertex_info.grayl)]
    return out.squeeze()


def right_cortex_data_10k(arr, fill=0, vertex_info=vertex_info_10k):
    """
    Add medial wall to array of cortex-only vertex values.

    This function takes a 1D array of fMRI grayordinates and returns the values
    on the vertices of the right 10k cortex mesh, including both left and right hemispheres. 
    The vertices of the medial wall are filled with a constant.

    Parameters
    ----------
    arr : numpy.ndarray
        The input cortical data, expected to correspond to gray matter vertices.

    fill : int or float, optional
        The value to use for filling vertices not included in the cortical data. Default is 0.

    vertex_info : object, optional
        Vertex information object containing indices of gray matter vertices 
        and the total number of vertices for the left and right hemispheres. 
        Default is `vertex_info_10k`.

    Returns
    -------
    numpy.ndarray
        The full 10k cortical surface data, including both left and right hemispheres. 
        Filled values are assigned to vertices outside the cortical data.

    Notes
    -----
    This implementation adapts similar functionality provided by the `hcp_utils` to a 10k
    surface. For the original implementation and additional context, refer to:
    https://github.com/rmldj/hcp-utils.git

    """

    if arr.ndim==1:
        arr = arr.reshape([1, -1])

    out = np.zeros([arr.shape[0], vertex_info.num_meshr])
    out[:, :] = fill
    if arr.shape[1] == len(vertex_info.grayr):
        out[:, vertex_info.grayr] = arr
    else:
        idx = slice(vertex_info.grayl.size, vertex_info.grayl.size + vertex_info.grayr.size)
        out[:, vertex_info.grayr] = arr[:, idx]
    return out.squeeze()


def sl_to_surf_map(metric, sl_indices):

    if isinstance(metric, dict):
        sl_to_surf = np.full_like(sl_indices, np.nan, dtype=float)
        for k, v in metric.items():
            sl_to_surf[sl_indices == k] = v
        
    elif isinstance(metric, np.ndarray):
        if metric.ndim == 1: metric = np.atleast_2d(metric.copy())
        sl_indices = rankdata(sl_indices.copy(), method="dense").astype(int).reshape(sl_indices.shape) - 1
        row_indices = np.atleast_2d(range(metric.shape[0])).T
        sl_to_surf = np.squeeze(metric[row_indices, sl_indices])

    return sl_to_surf