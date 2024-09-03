# brain_utils

import nibabel as nib
from nibabel import save, gifti, nifti1, cifti2
import numpy as np
from sklearn.utils import Bunch
from subprocess import run
import os.path as os


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

    if scalars.ndim==1:
        scalars = np.atleast_2d(scalars)
        
    data = np.zeros([scalars.shape[0],template_cifti.shape[1]])
    data[0:,0:scalars.shape[1]] = scalars
    
    if names == None:
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
        if isinstance(arg, list):
            prepped_args += (arg,)
            continue
        elif (isinstance(arg, (int, float, str, np.int32, np.int64, np.float64, np.float32, dict))) | (arg is None):
            prepped_args += ([arg for _ in range(N)],)
        else:
            raise TypeError(f"'{arg_names[i]}' is not the correct data type, consult documentation to find the compatible ones.")
        
    return prepped_args


def save_gifti(darrays=None, intents=0, dtypes=None, structure=None, filename=None, metas=None, coordsys=None, encoding='GIFTI_ENCODING_B64GZ', endian='little' ):

    darrays, intents, dtypes, metas = _prep_GiftiDataArray_args(darrays, intents, dtypes, metas)

    len_specified = [len(arg) for arg in [darrays, intents, dtypes, metas] if arg is not None]
    if ~np.all(len_specified):
        raise TypeError("If darrays, intents, dtypes, and metas are specified as lists, they should all have  the same number of items.")
    
    image = gifti.GiftiImage()
    for darray, intent, dtype, meta in zip(darrays, intents, dtypes, metas):
        darray = gifti.GiftiDataArray(data=darray, intent=intent, datatype=nifti1.data_type_codes[dtype], coordsys=coordsys, encoding=encoding, endian=endian, meta=meta)
        image.add_gifti_data_array(darray)
    
    save(image, filename)
    
    if structure:
        command = f"wb_command -set-structure {filename} '{structure}'"
        run(command, shell=True)


# indices, number , and offset of cortex vertices in LR10k
vertex_info_filename = f"{os.dirname(os.realpath(__file__))}/fMRI_vertex_info_10k.npz"
args = {k: v for k, v in np.load(vertex_info_filename).items()}
vertex_info_10k = Bunch(**args)
del vertex_info_filename


def cortex_data_10k(arr, fill=0, vertex_info=vertex_info_10k):

    if arr.ndim==1:
        arr = arr.reshape([1, -1])

    out = np.zeros([arr.shape[0], vertex_info.num_meshl + vertex_info.num_meshr])
    out[:, :] = fill
    graylr = np.hstack([vertex_info.grayl, vertex_info.grayr + vertex_info.num_meshl])
    out[:, graylr] = arr

    return out.squeeze()

def left_cortex_data_10k(arr, fill=0, vertex_info=vertex_info_10k):

    if arr.ndim==1:
        arr = arr.reshape([1, -1])

    out = np.zeros([arr.shape[0], vertex_info.num_meshl])
    out[:, :] = fill
    out[:, vertex_info.grayl] = arr[:, :len(vertex_info.grayl)]
    return out.squeeze()

def right_cortex_data_10k(arr, fill=0, vertex_info=vertex_info_10k):

    if arr.ndim==1:
        arr = arr.reshape([1, -1])

    out = np.zeros([arr.shape[0], vertex_info.num_meshr])
    out[:, :] = fill
    if arr.shape[1] == len(vertex_info.grayr):
        out[:, vertex_info.grayr] = arr
    else:
        out[:, vertex_info.grayr] = arr[:, vertex_info.grayl.size : vertex_info.grayl.size + vertex_info.grayr.size]
    return out.squeeze()

# def fun(x, y=default):
#     """
#     One sentence description of the function.

#     Extended
#     description
#     of the function

#     Parameters
#     ----------
#     x : type
#         Description of x.
#     y : type or <null option>, optional
#         Description of y.
#         Default is default.

#     Returns
#     -------
#     z : type
#         Description of z.
#     """