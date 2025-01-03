"""
This script processes resting-state fMRI timeseries data for individual subjects, 
resamples the timeseries from 32k to 10k vertices, normalizes the data, and computes 
a group-average functional connectivity (FC) matrix.

Parameters:
    <idx>: Integer
        The index of the subject in the subject list file.
    <fwhm>: Integer
        Full-width half-maximum (FWHM) for smoothing the timeseries.

Steps:
1. Preprocessing:
    - Split the 32k-resolution CIFTI timeseries into left and right hemisphere GIFTI files.
    - Smooth the hemispheric timeseries using the specified FWHM.
    - Resample the smoothed timeseries from 32k to 10k vertices.

2. Normalization:
    - Z-score normalize the resampled timeseries for each hemisphere.
    - Save the normalized timeseries in GIFTI format.

3. Concatenation:
    - Concatenate the normalized runs into a single dense timeseries file in CIFTI format.
    - Clean up intermediate files to save disk space.

4. Group-Level FC Matrix:
    - Wait until all subjects are processed.
    - Compute group-average functional connectivity matrix.

Dependencies:
    - `variograd_utils`: For dataset and subject handling, and file path utilities.
    - `nibabel`: For loading and saving neuroimaging data.
    - `wb_command`: For surface-based operations and timeseries processing.

Outputs:
    - 10k-resolution concatenated dense timeseries:
        `<output_dir>/<subject>.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii`
    - Group-average functional connectivity matrix:
        `<output_dir>/All.REST_FC.10k_fs_LR.npy`

Notes:
    - Intermediate files are removed to optimize storage usage.

"""

import sys
import os
import time
from subprocess import run
import numpy as np
from scipy.stats import zscore
import nibabel as nib
from variograd_utils import dataset, subject
from variograd_utils.brain_utils import save_gifti, vertex_info_10k


idx = int(sys.argv[1])-1
fwhm = int(sys.argv[2])

data = dataset()
ID = data.subj_list[idx]
subj = subject(ID)


# set formattable paths
runs = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]
tseries10k = subj.outpath(f"{ID}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
tseries10k_gii = subj.outpath(f"{ID}." + "{0}.rfMRI_{1}_Atlas_MSMAll.10k_fs_LR.func.gii")
tseries32k = subj.dir + "/MNINonLinear/Results/rfMRI_{0}/rfMRI_{0}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
tseries32k_gii = subj.outpath(f"{ID}." + "{0}.rfMRI_{1}_Atlas_MSMAll.32k_fs_LR.func.gii")
sphere32k = data.group_dir + "/S1200.{0}.sphere.32k_fs_LR.surf.gii"
sphere10k = data.mesh10k_dir + "/S1200.{0}.sphere.10k_fs_LR.surf.gii"
subj_surf32k = "{0}_midthickness_32k_T1w"
fc_matrix = data.outpath(f"{data.id}.REST_FC.10k_fs_LR.npy")


print("\n\n\nSubject:", ID)
print("\n\nPredefined Paths:")
print("tseries10k:", tseries10k)
print("tseries10k_gii:", tseries10k_gii)
print("tseries32k:", tseries32k)
print("tseries32k_gii:", tseries32k_gii)
print("sphere32k:", sphere32k)
print("sphere10k:", sphere10k)


concat_L = f"wb_command -metric-merge {tseries10k_gii.format('L', 'REST')}"
concat_R = f"wb_command -metric-merge {tseries10k_gii.format('R', 'REST')}"


print("\n\nSplitting 32k timeseries into hemispheres:")
for r in runs:
    print(f"\n\nProcessing {r}")

    separate_hemispheres = f"wb_command -cifti-separate {tseries32k.format(r)} COLUMN \
        -metric CORTEX_LEFT {tseries32k_gii.format('L', r)} \
            -metric CORTEX_RIGHT {tseries32k_gii.format('R', r)}"
    run(separate_hemispheres, shell=True)


    for h in ["L", "R"]:
        smooth = f"wb_command -metric-smoothing {getattr(subj, subj_surf32k.format(h))} \
            {tseries32k_gii.format(h, r)} \
                {fwhm} \
                    {tseries32k_gii.format(h, r)} \
                        -fwhm"
        run(smooth, shell=True)

        resample = f"wb_command -metric-resample {tseries32k_gii.format(h, r)} \
            {sphere32k.format(h)} \
                {sphere10k.format(h)} \
                    ADAP_BARY_AREA \
                        {tseries10k_gii.format(h, r)} \
                            -area-surfs {subj.L_midthickness_32k_T1w} {subj.R_midthickness_10k_T1w}"
        run(resample, shell=True)

    concat_L += f" -metric {tseries10k_gii.format('L', r)}"
    concat_R += f" -metric {tseries10k_gii.format('R', r)}"

    os.remove(tseries32k_gii.format('L', r))
    os.remove(tseries32k_gii.format('R', r))


    # Normalize timeseries
    print("Normalizing timeseries:")
    for h in ["L", "R"]:

        tseries = [t.data for t in nib.load(tseries10k_gii.format(h, r)).darrays]
        tseries_z = [z for z in zscore(np.array(tseries), axis=0, nan_policy='omit')]
        structure = "CORTEX_LEFT" if h == "L" else "CORTEX_RIGHT"

        print("Hemisphere:", h, "\tRun:", r)

        save_gifti(darrays=tseries_z, intents=11, dtypes=16, structure=structure, 
                   filename=tseries10k_gii.format(h, r), encoding=3, endian=2)


    # Concatenate runs
    if r == runs[-1]:
        run(concat_L, shell=True)
        run(concat_R, shell=True)

        print("\n\nConcatenated runs: ")
        print(f"\t{tseries10k_gii.format('L', 'REST')}")
        print(f"\t{tseries10k_gii.format('R', 'REST')}")

        # Clean up
        for ri in runs:
            os.remove(tseries10k_gii.format('L', ri))
            os.remove(tseries10k_gii.format('R', ri))



create_cifti = f"wb_command -cifti-create-dense-timeseries {tseries10k} \
    -left-metric {tseries10k_gii.format('L', 'REST')} \
        -right-metric {tseries10k_gii.format('R', 'REST')}"
run(create_cifti, shell=True)
print("\n\nOutput CIFTI:\n\t", tseries10k)


os.remove(tseries10k_gii.format('L', 'REST'))
os.remove(tseries10k_gii.format('R', 'REST'))


if idx == len(data.subj_list)-1:

    print("\nWaiting for all subjects to be processed before computing FC matrix")
    compute_FC = False
    while not compute_FC:
        print("\t...")
        time.sleep(30)
        compute_FC = all(
            os.path.exists(subj.outpath(f"{ID}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii"))
            for i in data.subj_list[:-1])
    time.sleep(30)

    print("Computing group-average FC matrix")
    n_vertices = vertex_info_10k.num_meshl + vertex_info_10k.num_meshr
    M = np.zeros((n_vertices, n_vertices))

    for ID in data.subj_list:
        subj = subject(ID)
        tseries = nib.load(subj.outpath(f"{ID}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
                           ).get_fdata()
        M += np.corrcoef(tseries.T)

    M /= len(data.subj_list)

    np.save(fc_matrix, M)
