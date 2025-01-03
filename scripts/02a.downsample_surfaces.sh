#!/bin/bash
#
# Downsample Midthickness Surfaces from 32k to 10k Vertices
#
# This script downsamples individual midthickness surfaces from 32k to 10k vertices
# using the `wb_command` tool.
#
# USAGE:
#   ./downsample_surfaces.sh <index>
#
# PARAMETERS:
#   <index>: Integer
#       The index of the subject in the subject list file in `directories.txt`.
#
# FUNCTIONALITY:
#   - Reads the subject list and selects the subject corresponding to the provided index.
#   - Creates the `T1w/fsaverage_LR10k` directory for the subject if it does not exist.
#   - For each hemisphere (`L` and `R`):
#       - Downsamples the midthickness surface using the BARYCENTRIC method.
#       - Saves the output in the `T1w/fsaverage_LR10k` directory.
#
# DEPENDENCIES:
#   - Requires the `set_directories.sh` script to set paths in `directories.txt`.
#   - Requires the `wb_command` tool for surface resampling.
#   - Requires the sphere meshes provided by Xu et al. (2020) available at:
#       https://github.com/TingsterX/alignment_macaque-human/trunk/surfaces/Human/10k_fs_LR
#     The `mesh10k_dir` in `directories.txt` should point to this directory.
#
# OUTPUT:
#   - Downsampled midthickness surfaces are saved as:
#       `<output_dir>/<subject>/T1w/fsaverage_LR10k/<subject>.<H>.midthickness_MSMAll.10k_fs_LR.surf.gii`
#
# NOTE:
#   - Ensure the `wb_command` tool is installed and available in your environment.
#
# EXAMPLES:
#   ./downsample_surfaces.sh 1
#       Downsamples the midthickness surfaces for the first subject in the list.
#

function USAGE {
    echo "Usage: ./downsample_surfaces.sh <index>"
    echo ""
    echo "Downsample midthickness surfaces from 32k to 10k vertices using the wb_command tool."
    echo ""
    echo "Arguments:"
    echo "  <index>  Integer index of the subject in the subject list file (specified in directories.txt)."
    echo ""
    echo "Description:"
    echo "  - Reads the subject list to select the subject corresponding to <index>."
    echo "  - Creates the 'T1w/fsaverage_LR10k' directory for the subject if it does not exist."
    echo "  - For each hemisphere ('L' and 'R'):"
    echo "      - Downsamples the midthickness surface using the BARYCENTRIC method."
    echo "      - Saves the output in the 'T1w/fsaverage_LR10k' directory."
    echo ""
    echo "Dependencies:"
    echo "  - set_directories.sh: Sets paths in directories.txt."
    echo "  - wb_command: Required for surface resampling."
    echo "  - Sphere meshes: Provided by Xu et al. (2020):"
    echo "      https://github.com/TingsterX/alignment_macaque-human/trunk/surfaces/Human/10k_fs_LR"
    echo "    The mesh10k_dir in directories.txt should point to this directory."
    echo ""
    echo "Output:"
    echo "  - Downsampled surfaces saved as:"
    echo "      <output_dir>/<subject>/T1w/fsaverage_LR10k/<subject>.<H>.midthickness_MSMAll.10k_fs_LR.surf.gii"
    echo ""
    echo "Example:"
    echo "  ./downsample_surfaces.sh 1"
    echo "      Downsamples the midthickness surfaces for the first subject in the list."
    exit 1
}

if [ $# -eq 0 ]; then
    USAGE
    exit 1;
fi


source ./variograd_utils/directories.txt

subject=$(sed -n "${1}p" $subj_list)

if [ ! -d "${output_dir}/${subject}/T1w/fsaverage_LR10k" ]; then mkdir -p "${output_dir}/${subject}/T1w/fsaverage_LR10k"; fi

for H in "L" "R";do

    surface_out="${output_dir}/${subject}/T1w/fsaverage_LR10k/${subject}.${H}.midthickness_MSMAll.10k_fs_LR.surf.gii"
    if [ -f ${surface_out} ]; then continue; fi

    surface_in="${subj_dir}/${subject}/T1w/fsaverage_LR32k/${subject}.${H}.midthickness_MSMAll.32k_fs_LR.surf.gii"
    current_sphere="${subj_dir}/${subject}/MNINonLinear/fsaverage_LR32k/${subject}.${H}.sphere.32k_fs_LR.surf.gii"
    new_sphere="${mesh10k_dir}/S1200.${H}.sphere.10k_fs_LR.surf.gii"
    method="BARYCENTRIC"

    wb_command -surface-resample ${surface_in} ${current_sphere} ${new_sphere} ${method} ${surface_out}

done