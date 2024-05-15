# 04 Compute geodesic dist mats

import nibabel as nib
from subprocess import run
from variograd_utils import *
from os.path import exists
import sys


index = int(sys.argv[1])-1

data = dataset()
id = data.subj_list[index]

command = "wb_command -surface-geodesic-distance-all-to-all {0} {1}"


# Compute geodesic distance on group surface
for h in ["L", "R"]:

    filename = data.outpath(f"{data.id}.{h}.gdist_triu.10k_fs_LR.npy")
    if exists(filename):
        continue

    surface = getattr(data, f"{h}_cortex_midthickness_10k")
    dconn = data.outpath(f"{data.id}.{h}.geodesic_distance.dconn.nii")
    run(command.format(surface, dconn), shell=True)
    gdist_matrix = nib.load(dconn)

    np.save(filename, gdist_matrix.get_fdata(caching="unchanged")[np.triu_indices_from(gdist_matrix, k=1)].astype("float32"))


subj = subject(id)

for h in ["L", "R"]:

    filename = data.outpath(f"{id}.{h}.gdist_triu.10k_fs_LR.npy")
    if exists(filename):
        continue
    
    surface = getattr(subj, f"{h}_cortex_midthickness_10k_T1w")
    dconn = data.outpath(f"tmp.{h}.geodesic_distance.dconn.nii")
    run(command.format(surface, dconn), shell=True)
    gdist_matrix = nib.load(dconn)

    np.save(filename, gdist_matrix.get_fdata(caching="unchanged")[np.triu_indices_from(gdist_matrix, k=1)].astype("float32"))
    
