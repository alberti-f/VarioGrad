from subprocess import run
from os.path import exists
import numpy as np
import nibabel as nib
import surfdist as sd
from variograd_utils.core_utils import dataset
from variograd_utils.brain_utils import vertex_info_10k, save_gifti


# Generate average surface
print("\n\nGenerating average group surface.")

data = dataset()
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
    save_gifti(darrays=cortex_surf,
               intents=[1008, 1009],
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
    run(command.format(surface, dconn), shell=True, check=True)
    gdist_matrix = nib.load(dconn)
    triu_idx = np.triu_indices_from(gdist_matrix, k=1)
    np.save(filename, gdist_matrix.get_fdata(caching="unchanged")[triu_idx].astype("float32"))
