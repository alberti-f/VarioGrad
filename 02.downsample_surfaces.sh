#!/bin/bash

# Downsample individual midthickness surfaces
#           This short script iterates through the subject IDs found in subj_list 
#           and downsamples the individual fsaverage 32k surfaces found in the T1w HCP
#           folder to 10k vertices using the sphere provided by Xu et al.(2020) at
#           https://github.com/TingsterX/alignment_macaque-human/trunk/surfaces/Human/10k_fs_LR

# Dependency:
#     https://www.humanconnectome.org/software/workbench-command
# ------------------------------------------------------------------

# VERSION=0.1.0
# NAME="./downsample_surfaces"
# function USAGE {
#     echo "DOWNSAMPLE INDIVIDUAL 32K SURFACES TO 10K VERTICES"
#     echo -e "\nusage: $NAME <group_dir> <subj_dir> <subj_list>"
#     echo -e "    <group_dir>      Directory where group average data is stored."
#     echo -e "    <subj_dir>      Directory where individual data folders are stored"
#     echo -e "    <subj_list>     path to .txt containing subject IDs"
#     echo -e "\n  Notes:"
#     echo -e "  The script assumes the Human Connectome Project's directory structure and."
#     echo -e "  searches for the 32k surfaces in: <subj_dir>/<subjectID>/T1w/fsaverage_LR32k."
#     echo -e "  The group average 10k spere is searched for in: <group_dir>/fsaverage_LR10k"
#     echo -e "  If this directory is missing it will be downloaded from:"
#     echo -e "  https://github.com/TingsterX/alignment_macaque-human/trunk/surfaces/Human/10k_fs_LR"
#     echo -e "\n  References: "
#     echo -e "  Xu, Nenning et al., (2020). Cross-species functional alignment reveals "
#     echo -e "    evolutionary hierarchy within the connectome. NeuroImage, 223,117346"
#     exit 1
# }

# # --- Option processing --------------------------------------------
# if [ $# == 0 ] ; then
#     USAGE
#     exit 1;
# fi

# group_dir=$1
# subj_dir=$2
# subj_list=$3

source ./variograd_utils/directories.txt

dir10k="${group_dir}/10k_fs_LR"
wd=$(pwd)

if [ ! -d ${dir10k} ]; then
    cd ${group_dir}
    svn checkout https://github.com/TingsterX/alignment_macaque-human/trunk/surfaces/Human/10k_fs_LR
    cd ${wd}
fi

while read subject; do
    if [ ! -d "${subj_dir}/${subject}/T1w/fsaverage_LR10k" ]; then mkdir "${subj_dir}/${subject}/T1w/fsaverage_LR10k"; fi
    for H in "L" "R";do

        surface_out="${subj_dir}/${subject}/T1w/fsaverage_LR10k/${subject}.${H}.midthickness_MSMAll.10k_fs_LR.surf.gii"
        if [ -f ${surface_out} ]; then continue; fi

        surface_in="${subj_dir}/${subject}/T1w/fsaverage_LR32k/${subject}.${H}.midthickness_MSMAll.32k_fs_LR.surf.gii"
        current_sphere="${subj_dir}/${subject}/MNINonLinear/fsaverage_LR32k/${subject}.${H}.sphere.32k_fs_LR.surf.gii"
        new_sphere="${dir10k}/S1200.${H}.sphere.10k_fs_LR.surf.gii"
        
        method="BARYCENTRIC"

        wb_command -surface-resample ${surface_in} ${current_sphere} ${new_sphere} ${method} ${surface_out}

    done
done < ${subj_list}


