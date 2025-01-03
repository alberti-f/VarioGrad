"""
This script computes geodesic distances between vertices on the 10k cortical surface 
of individual subjects.

Steps:
    1. Parse the subject index and retrieve the corresponding subject ID.
    2. For each hemisphere (`L` and `R`):
        - Load the cortical midthickness surface without the medial wall.
        - Compute geodesic distances between all vertices using `wb_command`.
        - Extract and save the upper triangular portion of the geodesic distance matrix.
        - Remove temporary `.dconn` files generated during computation.

Parameters:
    <index>: Integer
        The index (1-based) of the subject in the subject list file specified in `directories.txt`. 

Dependencies:
    - `variograd_utils`: For dataset and subject handling.
    - `nibabel`: For handling Workbench `.dconn` files.
    - `wb_command`: Workbench tool for geodesic distance computations.

Outputs:
    - Geodesic distance matrices (upper triangular portion):
        `<output_dir>/<subject>.<H>.gdist_triu.10k_fs_LR.npy`

Notes:
    - Ensure that `variograd_utils.dataset` provides the correct paths for input and output files.
    - The script assumes that the Workbench-compatible meshes (10k resolution) are already prepared.
    - Temporary `.dconn` files are deleted after geodesic distances are computed.

"""


import sys
import os
from os.path import exists
from subprocess import run
import numpy as np
import nibabel as nib
from variograd_utils import dataset, subject


index = int(sys.argv[1]) - 1

data = dataset()
ID = data.subj_list[index]

command = "wb_command -surface-geodesic-distance-all-to-all {0} {1}"

subj = subject(ID)

for h in ["L", "R"]:

    filename = data.outpath(f"{ID}.{h}.gdist_triu.10k_fs_LR.npy")
    if exists(filename):
        continue

    surface = getattr(subj, f"{h}_cortex_midthickness_10k_T1w")
    dconn = data.outpath(f"{ID}.tmp.{h}.geodesic_distance.dconn.nii")
    run(command.format(surface, dconn), shell=True)
    gdist_matrix = nib.load(dconn)

    np.save(filename, gdist_matrix.get_fdata(caching="unchanged"
                                             )[np.triu_indices_from(gdist_matrix, k=1)
                                               ].astype("float32"))

    os.remove(dconn)
    
