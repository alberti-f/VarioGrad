"""
This script generates cortical surface meshes without the medial wall for individual 
subjects by downsampling their midthickness surfaces at 10k resolution.

Steps:
    1. Parse the subject index and retrieve the corresponding subject ID.
    2. For each hemisphere (`L` and `R`):
        - Load the full midthickness surface for the subject.
        - Retain only cortical vertices using indices from `vertex_info_10k`.
        - Save the processed surface in GIFTI format with metadata (e.g., structure and intents).
    3. Print success messages if the output files are successfully created.

Parameters:
    <index>: Integer
        The index (1-based) of the subject in the subject list file specified in `directories.txt`. 

Dependencies:
    - `variograd_utils`: For dataset and subject handling.
    - `surfdist`: For surface processing utilities.

Outputs:
    - Processed cortical surfaces without the medial wall:
        `<output_dir>/<subject>/T1w/fsaverage_LR10k/<subject>.<H>.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii`

Notes:
    - The medial wall removal relies on pre-computed indices in `vertex_info_10k`.
    - Ensure the `variograd_utils.dataset` class provides the correct paths for input and output.
    
"""


import sys
from os.path import exists
import surfdist as sd
from variograd_utils import dataset, subject
from variograd_utils.brain_utils import save_gifti, vertex_info_10k

index = int(sys.argv[1])-1

data = dataset()
ID = data.subj_list[index]

# removing medial wall from subj surface
print("\n\nRemoving medial wall from subject surface.")
for H in ["L", "R"]:
    subj = subject(ID)
    full_surf = [darray.data for darray in subj.load_surf(H, 10).darrays]
    cortex_idx = vertex_info_10k[f"gray{H.lower()}"]
    cortex_surf = sd.utils.surf_keep_cortex(full_surf, cortex=cortex_idx)
    structure = ["CORTEX_LEFT" if H=="L" else "CORTEX_RIGHT"][0]
    filename = subj.outpath(f"/T1w/fsaverage_LR10k/{ID}.{H}.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii")
    save_gifti(darrays=cortex_surf, intents=[1008, 1009], dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"],
                filename=filename, structure=structure)
    
if exists(subj.outpath(f"/T1w/fsaverage_LR10k/{ID}.L.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii")):
    print("\tLeft successful: ", subj.outpath(f"/T1w/fsaverage_LR10k/{ID}.L.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii"))
if exists(subj.outpath(f"/T1w/fsaverage_LR10k/{ID}.R.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii")):
    print("\tRight successful: ", subj.outpath(f"/T1w/fsaverage_LR10k/{ID}.R.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii"))