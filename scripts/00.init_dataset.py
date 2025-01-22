import sys
import os
import argparse
from variograd_utils import init_dataset

# Create the parser
parser = argparse.ArgumentParser(
    description="This script is a wrapper for variograd_utils.core_utils.init_dataset()."
                + "It adds/updates an entry to the directories.json file which contains"
                + "the key directories paths used to initialize the"
                + "variograd_utils.core_utils.dataset object"
)

# Add arguments
parser.add_argument(
    "-i", "--dataset_id",
    type=str,
    required=True,
    help="ID used to refer to the dataset in the variograd_utils.core_utils.dataset object"
)
parser.add_argument(
    "-g", "--group_dir",
    type=str,
    required=True,
    help="Directory containing group average data (e.g., 'HCP_S1200_GroupAvg_v1')."
)
parser.add_argument(
    "-s", "--subj_dir",
    type=str,
    required=True,
    help="Directory containing individual subject subdirectories."
)
parser.add_argument(
    "-o", "--output_dir",
    type=str,
    required=True,
    help="Directory where results will be saved."
)
parser.add_argument(
    "-l", "--subj_list",
    type=str,
    required=True,
    help="Path to a text file listing subject IDs (one ID per line).."
)
parser.add_argument(
    "-m", "--mesh10k_dir",
    type=str,
    required=True,
    help="Path to the directory containing 10k surface meshes."
)

# Parse arguments
args = parser.parse_args()

# Edit dataset JSON configuration file
init_dataset(**args.__dict__)

# Create output directory
os.makedirs(args.__dict__["output_dir"], exist_ok=True)
