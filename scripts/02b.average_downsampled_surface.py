"""
Generate Group-Average Surface and Compute Geodesic Distances.

The script relies on pre-defined paths and data structures from the `variograd_utils` package.

Steps:
    1. **Generate Average Surfaces**:
        - Creates average midthickness surfaces for left (`L`) and right (`R`) hemispheres
          at 10k resolution.
    2. **Remove Medial Wall**:
        - Extracts cortical vertices from group-average surfaces using `vertex_info_10k`.
        - Saves the processed surfaces in GIFTI.
    3. **Compute Geodesic Distances**:
        - Calculates geodesic distances between all vertices on the cortical surfaces.
        - Outputs a numpy array of the upper triangular geodesic distance matrix.

Dependencies:
    - `variograd_utils`: Provides dataset handling and utility functions.
    - `surfdist`: For cortical surface processing.
    - `nibabel`: For loading and saving GIFTI files.
    - `wb_command`: Workbench tool for geodesic distance computations.

Outputs:
    - GIFTI surfaces: `<mesh10k_dir>/<id>.<H>.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii`
    - Geodesic distance matrix: `<output_dir>/<id>.<H>.gdist_triu.10k_fs_LR.npy`

"""

import sys
from os.path import exists
from subprocess import run
import numpy as np
import nibabel as nib
import surfdist as sd
from variograd_utils import dataset
from variograd_utils.brain_utils import vertex_info_10k, save_gifti

dataset_id = str(sys.argv[1])

# Generate average surface
print("\n\nGenerating average group surface.")

data = dataset(dataset_id)
data.generate_avg_surf("L", 10)
data.generate_avg_surf("R", 10)



# Remove medial wall from group surface
for H in ["L", "R"]:
    surf_path = getattr(data, f"{H}_midthickness_10k")
    full_surf = [darray.data for darray in nib.load(surf_path).darrays]
    cortex_idx = vertex_info_10k[f"gray{H.lower()}"]
    cortex_surf = sd.utils.surf_keep_cortex(full_surf, cortex=cortex_idx)
    structure = ["CORTEX_LEFT" if H=="L" else "CORTEX_RIGHT"][0]
    filename = f"{data.mesh10k_dir}/{data.id}.{H}.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii"
    save_gifti(darrays=cortex_surf, intents=[1008, 1009],
               dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"], 
            filename=filename, structure=structure)



# Compute geodesic distance on group surface
print("\n\nComputing geodesic distances on the group surface.")

command = "wb_command -surface-geodesic-distance-all-to-all {0} {1}"

for h in ["L", "R"]:

    filename = data.outpath(f"{data.id}.{h}.gdist_triu.10k_fs_LR.npy")
    if exists(filename):
        continue

    surface = getattr(data, f"{h}_cortex_midthickness_10k")
    dconn = data.outpath(f"{data.id}.{h}.geodesic_distance.dconn.nii")
    run(command.format(surface, dconn), shell=True)
    gdist_matrix = nib.load(dconn)

    np.save(filename,
            gdist_matrix.get_fdata(caching="unchanged"
                                   )[np.triu_indices_from(gdist_matrix, k=1)].astype("float32"))