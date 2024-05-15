# generate cortex-only surfaces

from variograd_utils import *
import nibabel as nib
import surfdist as sd 

data = dataset()
data.generate_avg_surf("L", 10)
data.generate_avg_surf("R", 10)
data.__init__()

for H in ["L", "R"]:
    surf_path = getattr(data, f"{H}_midthickness_10k")
    full_surf = [darray.data for darray in nib.load(surf_path).darrays]
    cortex_idx = vertex_info_10k[f"gray{H.lower()}"]
    cortex_surf = sd.utils.surf_keep_cortex(full_surf, cortex=cortex_idx)
    structure = ["CORTEX_LEFT" if H=="L" else "CORTEX_RIGHT"][0]
    filename = data.group_dir+f"/10k_fs_LR/{data.id}.{H}.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii"
    save_gifti(darrays=cortex_surf, intents=[1008, 1009], dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"], 
            filename=filename, structure=structure)

for ID in data.subj_list:
    for H in ["L", "R"]:
        subj = subject(ID)
        full_surf = [darray.data for darray in subj.load_surf(H, 10).darrays]
        cortex_idx = vertex_info_10k[f"gray{H.lower()}"]
        cortex_surf = sd.utils.surf_keep_cortex(full_surf, cortex=cortex_idx)
        structure = ["CORTEX_LEFT" if H=="L" else "CORTEX_RIGHT"][0]
        filename = subj.outpath(f"/T1w/fsaverage_LR10k/{ID}.{H}.cortex_midthickness_MSMAll.10k_fs_LR.surf.gii")
        save_gifti(darrays=cortex_surf, intents=[1008, 1009], dtypes=["NIFTI_TYPE_FLOAT32","NIFTI_TYPE_INT32"], 
                   filename=filename, structure=structure)
