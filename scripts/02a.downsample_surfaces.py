import sys
import os
from subprocess import run
import numpy as np
from variograd_utils.core_utils import dataset, subject

dataset_id = str(sys.argv[1])
index = np.int32(sys.argv[2]) - 1

data = dataset(dataset_id)
ID = data.subj_list[index]
subj = subject(ID, dataset_id)


output_path = f"{data.output_dir}/{ID}/T1w/fsaverage_LR10k"
os.makedirs(output_path, exist_ok=True)

for H in ["L", "R"]:
    surface_out = f"{output_path}/{subj.id}.{H}.midthickness_MSMAll.10k_fs_LR.surf.gii"
    
    # Skip if output file already exists
    if os.path.isfile(surface_out):
        continue

    surface_in = f"{subj.dir}/T1w/fsaverage_LR32k/{subj.id}.{H}.midthickness_MSMAll.32k_fs_LR.surf.gii"
    current_sphere = f"{subj.dir}/MNINonLinear/fsaverage_LR32k/{subj.id}.{H}.sphere.32k_fs_LR.surf.gii"
    new_sphere = f"{data.mesh10k_dir}/S1200.{H}.sphere.10k_fs_LR.surf.gii"
    method = "BARYCENTRIC"

    # Run wb_command
    run(f"wb_command -surface-resample {surface_in} {current_sphere} {new_sphere} {method} {surface_out}", 
        shell=True)
