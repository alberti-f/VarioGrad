# Generate group-level CortexMask for the 10k_fs_LR surface.
#           This script was used to generate masks separating the cortical mantle 
#           from the medial wall. These are necessary to later on remove the
#           medial wall from individual surfaces when computing the
#           geodesic distance between vertices in WorkBench
#
#           Note: The script assumes that the 32k and 10k spheres are found in
#           <group-dir> and <group-dir>/10k_fs_LR/ respectively
#           
#           usage: python create_cortex_dlabel <group-dir>
#                   <group-dir>:    directory containing the goup average
#                                   surfaces and the 10k_fs_LR subdirectory
# ------------------------------------------------------------------

import numpy as np
from nibabel import gifti, save, load
import hcp_utils as hcp
from subprocess import run
from variograd_utils import *

data = dataset()
group_dir = data.group_dir
mesh10k_dir = data.mesh10k_dir
output_dir = data.output_dir


# set formattable paths
mask32k_path = output_dir + "/S1200.{0}.CortexMask.32k_fs_LR.shape.gii"
mask10k_path = mesh10k_dir + "/S1200.{0}.CortexMask.10k_fs_LR.shape.gii"
sphere32k_path = group_dir + "/S1200.{0}.sphere.32k_fs_LR.surf.gii"
sphere10k_path = mesh10k_dir + "/S1200.{0}.sphere.10k_fs_LR.surf.gii"

# get cortex vertices
vinfo = hcp.vertex_info
left_cortex = np.zeros(vinfo.num_meshl); left_cortex[vinfo.grayl] = 1
right_cortex = np.zeros(vinfo.num_meshr); right_cortex[vinfo.grayr] = 1

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
    mask = load(f"{mesh_10k}/S1200.{h}.CortexMask.10k_fs_LR.shape.gii").darrays[0].data.astype("bool")
    vinfo[f"num_mesh{h.lower()}"] = len(mask)
    vinfo[f"gray{h.lower()}"] = np.arange(len(mask))[mask]
    vinfo[f"offset{h.lower()}"] = len(mask)*(h=="R")

utils_dir = dataset().utils_dir
np.savez(f"{utils_dir}/fMRI_vertex_info_10k", 
         grayl=vinfo["grayl"], grayr=vinfo["grayr"],
         num_meshl=vinfo["num_meshl"], num_meshr=vinfo["num_meshr"],
         offsetl=vinfo["offsetl"], offsetr=vinfo["offsetr"])
    