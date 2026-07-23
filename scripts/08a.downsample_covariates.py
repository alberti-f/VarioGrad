# Approximate downsampling function

from subprocess import run
import nibabel as nib
from scipy.stats import zscore
from variograd_utils import *
import os, sys, time


def quick_downsample(file32k_nii, file10k_nii, surf32k, surf10k, sphere32k, sphere10k, outdir, verbose=False):
    file10k_gii = "10k_tmp.{0}." + file32k_nii.split("/")[-1]
    file10k_gii = file10k_gii.split(".")[:-2]
    file10k_gii = outdir +  "/" + ".".join(file10k_gii) + ".func.gii"
    file32k_gii = "32k_tmp.{0}." + file32k_nii.split("/")[-1]
    file32k_gii = file32k_gii.split(".")[:-2]
    file32k_gii = outdir +  "/" + ".".join(file32k_gii) + ".func.gii"
    file = file10k_nii.split("/")[-1].split(".")
    npyout = outdir + "/" + ".".join(file[:-2]) + ".npy"

    # Separate in two hemispheres
    if verbose:
        print("\n\nSplitting 32k file into hemispheres:")
    separate_hemispheres = f"wb_command -cifti-separate {file32k_nii} COLUMN \
        -metric CORTEX_LEFT {file32k_gii.format('L')} \
            -metric CORTEX_RIGHT {file32k_gii.format('R')}"
    run(separate_hemispheres, shell=True, check=True)

    if verbose:
        print("\n\nSmoothing 32k files:")
    for h in ["L", "R"]:
            smooth = f"wb_command -metric-smoothing {surf32k.format(h)} \
                {file32k_gii.format(h)} \
                    6 \
                        {file32k_gii.format(h)} \
                            -fwhm"
            run(smooth, shell=True)
    
    # Resample to 10k
    for h in ["L", "R"]:
        resample = f"wb_command -metric-resample {file32k_gii.format(h)} \
            {sphere32k.format(h)} \
                {sphere10k.format(h)} \
                    ADAP_BARY_AREA \
                        {file10k_gii.format(h)} \
                            -area-surfs {surf32k.format(h)} {surf10k.format(h)}"
        run(resample, shell=True, check=True)

    # Merge hemispheres
    create_cifti = f"wb_command -cifti-create-dense-scalar {file10k_nii} \
        -left-metric {file10k_gii.format('L')} \
            -right-metric {file10k_gii.format('R')}"
    run(create_cifti, shell=True, check=True)
    if verbose:
        print("\n\nOutput CIFTI:\n\t", file10k_nii)
    
    
    os.remove(file10k_gii.format('L'))
    os.remove(file10k_gii.format('R'))
    os.remove(file32k_gii.format('L'))
    os.remove(file32k_gii.format('R'))    
    
    # Convert to numpy and save
    if verbose:
        print("\n\nConverting to numpy:")
    cifti = nib.load(file10k_nii).get_fdata()
    cortex = np.hstack([vertex_info_10k.grayl, vertex_info_10k.num_meshl + vertex_info_10k.grayr])
    np.save(npyout, cifti[:, cortex].squeeze())
    if verbose:
        print("\n\nOutput numpy:\n\t", npyout)
    
    os.remove(file10k_nii)


####################################################################################################


dataset_id = str(sys.argv[1])
index = int(sys.argv[2])-1
data = dataset(dataset_id)
ID = data.subj_list[index]
subj = subject(ID, data.id)


# Downsample morpho data
to_downample_quick = ["corrThickness_MSMAll",
                      "curvature_MSMAll",
                      "MyelinMap_BC_MSMAll",
                      "sulc_MSMAll"]

outdir = subj.outpath("")[:-1]
filedir = f"{data.subj_dir}/{ID}/MNINonLinear/fsaverage_LR32k"
surf32k = f"{filedir}/{ID}.{{0}}.midthickness_MSMAll.32k_fs_LR.surf.gii"
surf10k = f"{outdir}/T1w/fsaverage_LR10k/{ID}.{{0}}.midthickness_MSMAll.10k_fs_LR.surf.gii"
for file in to_downample_quick:
    file32k_nii =  f"{filedir}/{ID}.{file}.32k_fs_LR.dscalar.nii"
    file10k_nii =  f"{outdir}/{ID}.{file}.10k_fs_LR.dscalar.nii"
    sphere32k = f"{data.group_dir}/S1200.{{0}}.sphere.32k_fs_LR.surf.gii"
    sphere10k = f"{data.mesh10k_dir}/S1200.{{0}}.sphere.10k_fs_LR.surf.gii"
    quick_downsample(file32k_nii, file10k_nii, surf32k, surf10k, sphere32k, sphere10k, outdir, verbose=True)
    if not os.path.exists(f"{outdir}/{ID}.{file}.10k_fs_LR.npy"):
        raise RuntimeError(ID, file)
