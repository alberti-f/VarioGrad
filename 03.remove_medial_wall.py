"""
Create meshes of the cortical surface only (without medial wall)

The script uses the workbench command line tool to downsample the midthickness
surface of each hemisphere from 32k to 10k vertices. The output is saved in the
T1w/fsaverage_LR10k directory of each subject. The script is designed to be run
on a cluster as array job. The script requires the set_directories.sh script to 
be run first to set the necessary directories. The <index> argument is used to 
select a subject from the subject list file.

Parameters:
<index>: Integer
    Index of the subject in the subject list file
    specified in the directories.txt file

Note:
    The downsampling uses sphere meshes from the fsaverage_LR32k and fsaverage_LR10k
    provided by Xu et al.(2020) at:
    https://github.com/TingsterX/alignment_macaque-human/trunk/surfaces/Human/10k_fs_LR
    The mesh10k_dir variable in the directories.txt file should point to this directory.
"""

from variograd_utils import *
import nibabel as nib
import surfdist as sd 
import sys
import os

index = int(sys.argv[1])-1

data = dataset()
ID = data.subj_list[index]

 

for H in ["L", "R"]:
    surf_path = getattr(data, f"{H}_midthickness_10k")
    full_surf = [darray.data for darray in nib.load(surf_path).darrays]
    cortex_idx = vertex_info_10k[f"gray{H.lower()}"]
    cortex_surf = sd.utils.surf_keep_cortex(full_surf, cortex=cortex_idx)
    structure = ["CORTEX_LEFT" if H=="L" else "CORTEX_RIGHT"][0]
    filename = f"{data.mesh10k_dir}/{data.id}.{H}.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii"
    save_gifti(darrays=cortex_surf, intents=[1008, 1009], dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"], 
            filename=filename, structure=structure)

for H in ["L", "R"]:
    subj = subject(ID)
    full_surf = [darray.data for darray in subj.load_surf(H, 10).darrays]
    cortex_idx = vertex_info_10k[f"gray{H.lower()}"]
    cortex_surf = sd.utils.surf_keep_cortex(full_surf, cortex=cortex_idx)
    structure = ["CORTEX_LEFT" if H=="L" else "CORTEX_RIGHT"][0]
    filename = subj.outpath(f"/T1w/fsaverage_LR10k/{ID}.{H}.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii")
    save_gifti(darrays=cortex_surf, intents=[1008, 1009], dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"], 
                filename=filename, structure=structure)