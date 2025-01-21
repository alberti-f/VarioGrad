"""
This script creates masks for the cortical mantle, excluding the medial wall, for 
group-average surfaces. The generated masks are used in downstream analyses, such as 
removing the medial wall when computing geodesic distances between vertices in WorkBench.

Directories and paths are derived from the `dataset` class in `variograd_utils`.

Steps:
    - Generates 32k cortical masks for the left and right hemispheres based on 
      vertex information (`hcp.vertex_info`).
    - Saves 32k masks as GIFTI files.
    - Downsamples the masks to 10k resolution using `wb_command` with the BARYCENTRIC method.
    - Refines the downsampled masks with mathematical operations.
    - Saves vertex information, including cortical regions and offsets, as a `.npz` file 
      for use in the `brain_utils` module.

Dependencies:
    - Requires the `hcp_utils` package for cortical vertex information.
    - Uses the `workbench` command-line tool (`wb_command`) for resampling operations.
    - Requires directories for 32k and 10k spheres, set in the `variograd_utils` dataset class.

Outputs:
    - 32k cortical masks: `<output_dir>/S1200.<H>.CortexMask.32k_fs_LR.shape.gii`
    - 10k cortical masks: `<output_dir>/S1200.<H>.CortexMask.10k_fs_LR.shape.gii`
    - Vertex information: `<utils_dir>/fMRI_vertex_info_10k.npz`

Notes:
    - Ensure that `group_dir`, `mesh10k_dir`, and `output_dir` are correctly set in the 
      `variograd_utils.dataset` class.
    - Sphere files for 32k and 10k surfaces must exist in the appropriate directories:
        - 32k spheres: `<group_dir>/S1200.<H>.sphere.32k_fs_LR.surf.gii`
        - 10k spheres: `<mesh10k_dir>/S1200.<H>.sphere.10k_fs_LR.surf.gii`

"""

import sys
import os
from subprocess import run
import numpy as np
from nibabel import gifti, save, load
import hcp_utils as hcp
import variograd_utils as vu

dataset_id = sys.argv[1]
data = vu.dataset(dataset_id)
group_dir = data.group_dir
mesh10k_dir = data.mesh10k_dir
output_dir = data.output_dir


# set formattable paths
mask32k_path = output_dir + "/S1200.{0}.CortexMask.32k_fs_LR.shape.gii"
mask10k_path = output_dir + "/S1200.{0}.CortexMask.10k_fs_LR.shape.gii"
sphere32k_path = group_dir + "/S1200.{0}.sphere.32k_fs_LR.surf.gii"
sphere10k_path = mesh10k_dir + "/S1200.{0}.sphere.10k_fs_LR.surf.gii"

# get cortex vertices
vinfo = hcp.vertex_info
left_cortex = np.zeros(vinfo.num_meshl)
left_cortex[vinfo.grayl] = 1
right_cortex = np.zeros(vinfo.num_meshr)
right_cortex[vinfo.grayr] = 1

left_cortex=gifti.GiftiDataArray(data=left_cortex, intent=0, datatype="NIFTI_TYPE_INT32")
right_cortex=gifti.GiftiDataArray(data=right_cortex, intent=0, datatype="NIFTI_TYPE_INT32")

# generate gifti masks
left_gifti = gifti.GiftiImage()
left_gifti.add_gifti_data_array(left_cortex)
save(left_gifti, mask32k_path.format("L"))

right_gifti = gifti.GiftiImage()
right_gifti.add_gifti_data_array(right_cortex)
save(right_gifti, mask32k_path.format("R"))

# add structure and downsample mask to 10k
command = f"\
    wb_command -set-structure {mask32k_path.format('L')} 'CORTEX_LEFT' ;\
    wb_command -metric-resample {mask32k_path.format('L')} {sphere32k_path.format('L')} {sphere10k_path.format('L')} \
                                BARYCENTRIC {mask10k_path.format('L')} ;\
    wb_command -metric-math '(x>0)' {mask10k_path.format('L')} -var 'x' {mask10k_path.format('L')} ;\
    \
    wb_command -set-structure {mask32k_path.format('R')} 'CORTEX_RIGHT' ;\
    wb_command -metric-resample {mask32k_path.format('R')} {sphere32k_path.format('R')} {sphere10k_path.format('R')} \
                                BARYCENTRIC {mask10k_path.format('R')} ;\
    wb_command -metric-math '(x>0)' {mask10k_path.format('R')} -var 'x' {mask10k_path.format('R')} \
    "

#create .npz with vertex info for use by brain_utils module
vinfo = {}
run(command, shell=True)
for h in ["L", "R"]:
    mask = load(f"{data.output_dir}/S1200.{h}.CortexMask.10k_fs_LR.shape.gii"
                ).darrays[0].data.astype("bool")
    vinfo[f"num_mesh{h.lower()}"] = len(mask)
    vinfo[f"gray{h.lower()}"] = np.arange(len(mask))[mask]
    vinfo[f"offset{h.lower()}"] = len(mask)*(h=="R")

utils_dir = os.path.dirname(vu.__file__)
np.savez(f"{utils_dir}/fMRI_vertex_info_10k", 
         grayl=vinfo["grayl"], grayr=vinfo["grayr"],
         num_meshl=vinfo["num_meshl"], num_meshr=vinfo["num_meshr"],
         offsetl=vinfo["offsetl"], offsetr=vinfo["offsetr"])
    