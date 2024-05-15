#!/bin/bash

# Set the directories for the whole study.
#           This script creates a file in the variograd_utils directory 
#           stating the key directories used throughout the analyses.
#           For this to be possible the variograd_utils directory should 
#           be in the same working directory from which this script is 
#           executed from.
#           

# ------------------------------------------------------------------

VERSION=0.1.0
NAME="./set_directories.sh"
function USAGE {
    echo "SET WORK AND DATASET DIRECTORIES FOR THE ANALYSES"
    echo -e "\nusage: $NAME <group_dir> <subj_dir> <subj_list>"
    echo -e "    -g <group_dir>      Directory where group average data is stored."
    echo -e "                         This should be the HCP_S1200_GroupAvg_v1"
    echo -e "                         folder of the HCP dataset"
    echo -e "    -s <subj_dir>       Directory where individual data folders are stored"
    echo -e "                         This should be the directory containing the"
    echo -e "                         subjects' subdirectories"
    echo -e "    -o <output_dir>     Directory where the results are saved"
    echo -e "    -s <subj_list>      Path to a txt file containing HCP subject IDs"
    exit 1
}

if [ $# == 0 ] ; then
    USAGE
    exit 1;
fi

dir_file=./variograd_utils/directories.txt

# truncate file if existing
if [ -f ./variograd_utils/directories.txt ]; then : > $dir_file; fi

# set currend working directory as base
echo "work_dir=$(pwd)" >> $dir_file

# assign flagged paths
while getopts g:s:o:l: flag
do
    case "${flag}" in
        g) echo "group_dir=$OPTARG" >> $dir_file;;
        s) echo "subj_dir=$OPTARG" >> $dir_file;;
        o) echo "output_dir=$OPTARG" >> $dir_file;;
        l) echo "subj_list=$OPTARG" >> $dir_file;;
    esac
done

