"""
This script creates covariate arrays for left and right hemispheres.
These include demographic information (age, gender) and morphometric metrics
(cortical thickness, curvature, myelin map, sulcal depth).

Parameters:
    <dataset_id>: String
        Identifier for the dataset being processed.
    <unrestricted>: String
        Path to the CSV file containing HCP unrestricted demographic data.
    <restricted>: String
        Path to the CSV file containing HCP restricted demographic data.

Outputs:
    - Left hemisphere covariate array:
        `<output_dir>/<dataset_id>.L.covariates.npy`
    - Right hemisphere covariate array:
        `<output_dir>/<dataset_id>.R.covariates.npy`
    The covariate arrays are 3D numpy arrays with dimensions:
        [number of subjects, number of vertices, number of covariates]
"""

import os, sys
import pandas as pd
import variograd_utils as vu
import numpy as np

cov_names = ["Age_in_Yrs", "Gender"]
metrics = ["corrThickness_MSMAll",
            "curvature_MSMAll",
            "MyelinMap_BC_MSMAll",
            "sulc_MSMAll"]

dataset_id = sys.argv[1]
unrestricted = sys.argv[2]
restricted = sys.argv[3]

dataset = vu.dataset(dataset_id)
nv_l = vu.vertex_info_10k.grayl.size
nv_r = vu.vertex_info_10k.grayr.size

demographics = pd.read_csv(restricted, index_col=0)
demographics = demographics.merge(pd.read_csv(unrestricted, index_col=0), left_index=True, right_index=True)
demographics = demographics.loc[dataset.subj_list, cov_names]
demographics["Gender"] = demographics["Gender"].map({"M": 0, "F": 1})
demographics_l = np.tile(demographics.values, [nv_l, 1, 1]).transpose(1, 0, 2)
demographics_r = np.tile(demographics.values, [nv_r, 1, 1]).transpose(1, 0, 2)

path = dataset.output_dir + "/{0}/{0}.{1}.10k_fs_LR.npy"
morphometry = np.zeros([dataset.N, nv_l + nv_r, len(metrics)])
for i, m in enumerate(metrics):
    for j, s in enumerate(dataset.subj_list):
        if not os.path.exists(path.format(s, m)):
            print(f"Missing file for subject {s}, metric {m}")
            continue
        morphometry[j, :, i] = np.load(path.format(s, m))

morphometry_l = morphometry[:, :nv_l, :]
morphometry_r = morphometry[:, nv_l:, :]

covariates_l = np.concatenate([demographics_l, morphometry_l], axis=2)
covariates_r = np.concatenate([demographics_r, morphometry_r], axis=2)


np.save(dataset.outpath(f"{dataset_id}.L.covariates.npy"), covariates_l)
np.save(dataset.outpath(f"{dataset_id}.R.covariates.npy"), covariates_r)
