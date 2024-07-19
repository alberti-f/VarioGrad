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
runs = ["REST1_LR", "REST2_LR"]
tseries10k = subj.outpath(f"{id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")
tseries10k_gii = subj.outpath(f"{id}." + "{0}.rfMRI_{1}_Atlas_MSMAll.10k_fs_LR.func.gii")
tseries32k = subj.dir + "/MNINonLinear/Results/rfMRI_{0}/rfMRI_{0}_Atlas_MSMAll_hp2000_clean.dtseries.nii" # 
tseries32k_gii = subj.outpath(f"{id}." + "{0}.rfMRI_{1}_Atlas_MSMAll.32k_fs_LR.func.gii")
sphere32k = data.group_dir + "/S1200.{0}.sphere.32k_fs_LR.surf.gii"
sphere10k = data.mesh10k_dir + "/S1200.{0}.sphere.10k_fs_LR.surf.gii"
subj_surf32k = "{0}_midthickness_32k_T1w"
fc_matrix = data.outpath(f"{data.id}.REST_FC.10k_fs_LR.npy")
fwhm = 6

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
                            -area-surfs {subj.L_midthickness_32k_T1w}  {subj.L_midthickness_10k_T1w}"
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

        print(f"Hemisphere:", h, "\tRun:", r)
   
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
    
    n_vertices = vertex_info_10k.num_meshl + vertex_info_10k.num_meshr
    M = np.zeros((n_vertices, n_vertices))

    for id in data.subj_list:
        subj = subject(id)
        tseries = nib.load(subj.outpath(f"{id}.rfMRI_REST_Atlas_MSMAll.10k_fs_LR.dtseries.nii")).get_fdata()
        M += np.corrcoef(tseries.T)

    M /= len(data.subj_list)

    np.save(fc_matrix, M)



