#!/bin/bash

"""
Downsample individual midthickness surfaces from 32k to 10k vertices

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



source ./variograd_utils/directories.txt

subject=$(sed -n "${1}p" $subj_list)

if [ ! -d "${output_dir}/${subject}/T1w/fsaverage_LR10k" ]; then mkdir "${output_dir}/${subject}/T1w/fsaverage_LR10k"; fi

for H in "L" "R";do

    surface_out="${output_dir}/${subject}/T1w/fsaverage_LR10k/${subject}.${H}.midthickness_MSMAll.10k_fs_LR.surf.gii"
    if [ -f ${surface_out} ]; then continue; fi

    surface_in="${subj_dir}/${subject}/T1w/fsaverage_LR32k/${subject}.${H}.midthickness_MSMAll.32k_fs_LR.surf.gii"
    current_sphere="${subj_dir}/${subject}/MNINonLinear/fsaverage_LR32k/${subject}.${H}.sphere.32k_fs_LR.surf.gii"
    new_sphere="${mesh10k_dir}/S1200.${H}.sphere.10k_fs_LR.surf.gii"
    method="BARYCENTRIC"

    wb_command -surface-resample ${surface_in} ${current_sphere} ${new_sphere} ${method} ${surface_out}

done