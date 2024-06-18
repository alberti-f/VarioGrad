from subprocess import run
import nibabel as nib
from scipy.stats import zscore
from variograd_utils import *
import os, sys


idx = int(sys.argv[1])-1

data = dataset()
id = data.subj_list[idx]
subj = subject(id)


# set formattable paths
runs = ["REST1_LR", "REST2_RL"]
tseries10k = subj.outpath(f"{id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
tseries10k_gii = subj.outpath(f"{id}." + "{0}.rfMRI_{1}_Atlas_MSMAll.10k_fs_LR.func.gii")
tseries32k = subj.dir + "/MNINonLinear/Results/rfMRI_{0}/rfMRI_{0}_Atlas_MSMAll_hp2000_clean.dtseries.nii" #rfMRI_{0}_Atlas_MSMAll
tseries32k_gii = subj.outpath(f"{id}." + "{0}.rfMRI_{1}_Atlas_MSMAll.32k_fs_LR.func.gii")
sphere32k = data.group_dir + "/S1200.{0}.sphere.32k_fs_LR.surf.gii"
sphere10k = data.mesh10k_dir + "/S1200.{0}.sphere.10k_fs_LR.surf.gii"

print("\n\n\nSubject:", id)
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
    print(f"\tProcessing {r}")

    separate_hemispheres = f"wb_command -cifti-separate {tseries32k.format(r)} COLUMN \
        -metric CORTEX_LEFT {tseries32k_gii.format('L', r)} \
            -metric CORTEX_RIGHT {tseries32k_gii.format('R', r)}"
    run(separate_hemispheres, shell=True)
    

    for h in ["L", "R"]:
        resample = f"wb_command -metric-resample {tseries32k_gii.format(h, r)} \
            {sphere32k.format(h)} \
                {sphere10k.format(h)} \
                    ADAP_BARY_AREA \
                        {tseries10k_gii.format(h, r)} \
                            -area-surfs {subj.L_midthickness_32k_T1w}  {subj.L_midthickness_10k_T1w}"
        run(resample, shell=True)
    
    concat_L += f" -metric {tseries10k_gii.format('L', r)}"
    concat_R += f" -metric {tseries10k_gii.format('R', r)}"
    
    os.remove(tseries32k_gii.format('L', r))
    os.remove(tseries32k_gii.format('R', r))


    # Normalize timeseries
    print("\n\nNormalizing timeseries:")
    for h in ["L", "R"]:

        tseries = [t.data for t in nib.load(tseries10k_gii.format(h, r)).darrays]
        tseries_z = [z for z in zscore(np.array(tseries), axis=0, nan_policy='omit')]
        structure = "CORTEX_LEFT" if h == "L" else "CORTEX_RIGHT"

        print(f"\n\tHemisphere:", h, "\tRun:", r)
        print("\tt points:", len(tseries_z), 
              "\tVertices:", len(tseries_z[0]),
              "\tMean", np.nanmean(np.nanmean(tseries_z, axis=0)), 
              "\tSD", np.nanmean(np.nanstd(tseries_z, axis=0))
   
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




