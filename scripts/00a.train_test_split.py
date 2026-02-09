import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from variograd_utils import init_dataset

# Create the parser
parser = argparse.ArgumentParser(
    description="This executable script splits the participants from the HCP dataset into "
                + "a train and test set excluding participants with QC code A (anatomical "
                + "anomalies). Subjects are split maintaining members of each family in "
                + "one group and stratifying by the number of members in each family to "
                + "avoid uneve representations of genetic features between subsamples."
                + "Saves two .TXT files with the subject IDs for each subsample."
)

# Add arguments
parser.add_argument(
    "-u", "--unrestricted",
    type=str,
    required=True,
    help="Path to the unrestricted HCP data (/path/to/unrestricted_user_0_0_0000_0_00_00.csv)"
)
parser.add_argument(
    "-r", "--restricted",
    type=str,
    required=True,
    help="Path to the RESTRICTED HCP data (/path/to/RESTRICTED_user_0_0_0000_0_00_00.csv)"
)
parser.add_argument(
    "-s", "--suffix",
    type=str,
    required=False,
    help="Suffix to add at the beginning of the filenames to differentiate iterations."
)
parser.add_argument(
    "-x", "--random_state",
    type=int,
    required=False,
    help="Random state for sklearn.model_selection.train_test_split()."
)
parser.add_argument(
    "-o", "--output_dir",
    type=str,
    required=False,
    help="Output directoy where to save the subject ID lists."
)

args = parser.parse_args().__dict__


restricted = pd.read_csv(args["restricted"], usecols=["Subject", "Family_ID"])
unrestricted = pd.read_csv(args["unrestricted"], usecols=["Subject", "QC_Issue", "3T_RS-fMRI_PctCompl"])

data = restricted.merge(unrestricted, on="Subject")

QC_A = data["QC_Issue"].str.contains("A")
QC_A[QC_A.isna()] = False

rfMRI100 = data["3T_RS-fMRI_PctCompl"] == 100

clean_data = data[~QC_A & rfMRI100]

families = clean_data.Family_ID.unique()

n_relatives = np.array([np.sum(clean_data.Family_ID == family) for family in families])
n_relatives[n_relatives > 3] = 4
train_families, test_families = train_test_split(families, stratify=n_relatives, random_state=args["random_state"])

train_data = clean_data[clean_data.Family_ID.isin(train_families)]
test_data = clean_data[clean_data.Family_ID.isin(test_families)]

train_IDs = train_data.Subject.values
test_IDs = test_data.Subject.values


cond1 = train_data.Family_ID.isin(test_families).any()
cond2 = test_data.Family_ID.isin(train_families).any()
cond3 = train_data.Subject.isin(test_data.Subject).any()
cond4 = test_data.Subject.isin(train_data.Subject).any()
if cond1 | cond2 | cond3 | cond4:
    print("Error in splitting")

suffix = "" if args["suffix"] is None else f"{args["suffix"]}_"
# # print(args["output_dir"] + "/" args["suffix"] + "train_IDs.txt")
np.savetxt(f"{args["output_dir"]}/{suffix}train_IDs.txt", train_IDs, fmt="%d")
np.savetxt(f"{args["output_dir"]}/{suffix}test_IDs.txt", test_IDs, fmt="%d")
