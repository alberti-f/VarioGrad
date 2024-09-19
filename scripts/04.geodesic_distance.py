# 04 Compute geodesic distance mats

"""
Create meshes of the cortical surface only (without medial wall)

The script uses the workbench command line tool to compue the geodesic distance
between vertices on the 10k surface.
The script is designed to be run as array job receiving the task index as argument.
The <index> argument is used to select a subject from the subject list file.

Parameters:
<index>: Integer
    Index of the subject in the subject list file
    specified in the directories.txt file
"""

import sys
import os
from subprocess import run
import numpy as np
import nibabel as nib
from variograd_utils.core_utils import dataset, subject



index = int(sys.argv[1])-1

data = dataset()
ID = data.subj_list[index]

command = "wb_command -surface-geodesic-distance-all-to-all {0} {1}"

subj = subject(ID)

for h in ["L", "R"]:

    filename = data.outpath(f"{ID}.{h}.gdist_triu.10k_fs_LR.npy")
    if os.path.exists(filename):
        continue
    surface = getattr(subj, f"{h}_cortex_midthickness_10k_T1w")
    dconn = data.outpath(f"{ID}.tmp.{h}.geodesic_distance.dconn.nii")
    run(command.format(surface, dconn), shell=True, check=True)
    gdist_matrix = nib.load(dconn)

    triu_idx = np.triu_indices_from(gdist_matrix, k=1)
    np.save(filename, gdist_matrix.get_fdata(caching="unchanged")[triu_idx].astype("float32"))
    os.remove(dconn)
