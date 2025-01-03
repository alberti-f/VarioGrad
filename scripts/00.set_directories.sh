#!/bin/bash
#
# This script saves key directories required for analysis in a file named
# `directories.txt` located in the `variograd_utils` directory.
#
# USAGE:
#   ./set_directories.sh -g <group_dir> -s <subj_dir> -o <output_dir> -l <subj_list> -m <10k_meshes>
#
# OPTIONS:
#   -g <group_dir>   Directory containing group average data (e.g., `HCP_S1200_GroupAvg_v1`).
#   -s <subj_dir>    Directory containing individual subject subdirectories.
#   -o <output_dir>  Directory where results will be saved.
#   -l <subj_list>   Path to a text file listing subject IDs (one ID per line).
#   -m <mesh10k_dir> Path to the directory containing 10k surface meshes.
#
# FUNCTIONALITY:
#   - Writes the current working directory as `work_dir` to the `directories.txt` file.
#   - Appends the specified paths for `group_dir`, `subj_dir`, `output_dir`, `subj_list`, and `mesh10k_dir`.
#   - Truncates the existing `directories.txt` file if it already exists.
#
# NOTES:
#   - Ensure the `variograd_utils` directory exists in the current working directory.
#   - The script relies on Python to locate the `variograd_utils` module.
#   - All paths are saved in the format: `key=value` for use in downstream analyses.
#
# EXAMPLES:
#   ./set_directories.sh -g /path/to/group_data -s /path/to/subject_data -o /path/to/output \
#                        -l /path/to/subject_list.txt -m /path/to/10k_meshes
#


# ------------------------------------------------------------------

function USAGE {
    echo "Usage: ./set_directories.sh -g <group_dir> -s <subj_dir> -o <output_dir> -l <subj_list> -m <10k_meshes>"
    echo ""
    echo "This script saves key directories required for analysis in a file named"
    echo "'directories.txt' located in the 'variograd_utils' directory."
    echo ""
    echo "Options:"
    echo "  -g <group_dir>   Directory containing group average data (e.g., 'HCP_S1200_GroupAvg_v1')."
    echo "  -s <subj_dir>    Directory containing individual subject subdirectories."
    echo "  -o <output_dir>  Directory where results will be saved."
    echo "  -l <subj_list>   Path to a text file listing subject IDs (one ID per line)."
    echo "  -m <mesh10k_dir> Path to the directory containing 10k surface meshes."
    echo ""
    echo "Functionality:"
    echo "  - Writes the current working directory as 'work_dir' to the 'directories.txt' file."
    echo "  - Appends the specified paths for 'group_dir', 'subj_dir', 'output_dir', 'subj_list', and 'mesh10k_dir'."
    echo "  - Truncates the existing 'directories.txt' file if it already exists."
    echo ""
    echo "Notes:"
    echo "  - Ensure the 'variograd_utils' directory exists in the current working directory."
    echo "  - This script uses Python to locate the 'variograd_utils' module."
    echo "  - All paths are saved in the format 'key=value' for use in downstream analyses."
    echo ""
    echo "Examples:"
    echo "  ./set_directories.sh -g /path/to/group_data -s /path/to/subject_data -o /path/to/output \\"
    echo "                       -l /path/to/subject_list.txt -m /path/to/10k_meshes"
    exit 1
}


if [ $# == 0 ] ; then
    USAGE
    exit 1;
fi

dir_pkg=$(python -c "import os; import variograd_utils; print(os.path.dirname(variograd_utils.__file__))")
dir_file=$dir_pkg/directories.txt

# truncate file if existing 
if [ -f ./variograd_utils/directories.txt ]; then : > $dir_file; fi

# set current working directory as base
echo "work_dir=$(pwd)" >> $dir_file

# assign flagged paths
while getopts g:s:o:l:m: flag
do
    case "${flag}" in
        g) echo "group_dir=$OPTARG" >> $dir_file;;
        s) echo "subj_dir=$OPTARG" >> $dir_file;;
        o) echo "output_dir=$OPTARG" >> $dir_file;;
        l) echo "subj_list=$OPTARG" >> $dir_file;;
        m) echo "mesh10k_dir=$OPTARG" >> $dir_file;;
    esac
done
