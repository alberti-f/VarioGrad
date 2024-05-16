from variograd_utils import *
from os.path import exists
from subprocess import run
import nibabel as nib
import sys


# Generate average surface
print("\n\nGenerating average group surface.")

data = dataset()
data.generate_avg_surf("L", 10)
data.generate_avg_surf("R", 10)


# Compute geodesic distance
print("\n\nComputing geodesic distances on the group surface.")

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