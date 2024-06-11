# compute between-subject similarity

import numpy as np
from variograd_utils import *
from os.path import exists
from time import sleep
import sys, os


pair_idx = int(sys.argv[1])-1

overwrite = True
algorithm = ["GCCA", "JE"]


data =  dataset()
embed_l, embed_r = (data.load_embeddings("L", algorithm), data.load_embeddings("R", algorithm))

n_comps = embed_l[list(embed_l.keys())[0]].shape[2]
n_ks = len(embed_l.keys())
vinfo = vertex_info_10k

skip_algs_l = skip_algs_r =[]
if exists(data.outpath("All.L.between_subj_similarity.npz")) and not overwrite:
    skip_algs_l = np.load(data.outpath(f"All.L.between_subj_similarity.npz")).keys()
if exists(data.outpath("All.R.between_subj_similarity.npz")) and not overwrite:
    skip_algs_r = np.load(data.outpath(f"All.R.between_subj_similarity.npz")).keys()


subj_pair = data.pairs[pair_idx]
subj_i = subject(subj_pair[0])
subj_j = subject(subj_pair[1])


print("\n\nSubject pair:", subj_i.id, subj_j.id)

# Wait until all subject pairs are done
if pair_idx == len(data.pairs)-1:
    print("\nWaiting for all pairs to be done before running the last one")
    run_last = False
    while not run_last:
        print("\t...")
        sleep(30)
        run_last = all([os.path.exists(data.outpath(f"{i}-{j}.R.embed_similarity.npz")) 
                        for i, j in data.pairs])
    
sleep(30)

print("\n\nComputing correlation of embeddings\n")
correlations_l = {}
correlations_r = {}
for k in embed_l.keys():

    if k in skip_algs_l and not overwrite:
        print(f"\t{k} skipped")
        continue

    correlations_l[k] = vector_wise_corr(embed_l[k][subj_i.idx].T, embed_l[k][subj_j.idx].T).astype("float32")
    correlations_r[k] = vector_wise_corr(embed_r[k][subj_i.idx].T, embed_r[k][subj_j.idx].T).astype("float32")

    print(f"\t{k} completed")


npz_update(data.outpath(f"{subj_i.id}-{subj_j.id}.L.embed_similarity.npz"), correlations_l)
npz_update(data.outpath(f"{subj_i.id}-{subj_j.id}.R.embed_similarity.npz"), correlations_r)


###################################################################################


# If all pairs are done, stack them for significance testing
# and remove the individual files

algorithms = list(embed_l.keys())
filenames_l = [data.outpath(f"All.L.bwsim.{k}.npy") for k in algorithms]
filenames_r = [data.outpath(f"All.R.bwsim.{k}.npy") for k in algorithms]

# Preallocating files
print("Preallocating empty files:\n")
for f in filenames_l:
    if exists(f) and not overwrite:
        print("\t", f, "already exists")
        continue

    np.save(f, np.zeros([len(data.pairs), vinfo.grayl.size], dtype="float32"))
    print("\t", f)

for f in filenames_r:
    if exists(f) and not overwrite:
        print("\t", f, "already exists")
        continue

    np.save(f, np.zeros([len(data.pairs), vinfo.grayr.size], dtype="float32"))
    print("\t", f)




print("\n\nUpdating group files:\n")

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f"\n\tStart = {current_time}")


print("\n\tLeft hemisphere")
for file in filenames_l:
    bwsim_l = np.load(file, mmap_mode="r+")

    for idx, (i, j) in enumerate(data.pairs):
        if not (exists(data.outpath(f"{i}-{j}.L.embed_similarity.npz")) and exists(data.outpath(f"{i}-{j}.R.embed_similarity.npz"))):
                print(f"Missing pair {i}-{j}")
                continue        
        
        bwsim_l[idx] = np.load(data.outpath(f"{i}-{j}.L.embed_similarity.npz"))[file.split(".")[-2]]

    bwsim_l.flush()    
    print(f"\t\t{file} done")


print("\n\tRight hemisphere")
for file in filenames_r:
    bwsim_r = np.load(file, mmap_mode="r+")

    for idx, (i, j) in enumerate(data.pairs):
        if not (exists(data.outpath(f"{i}-{j}.L.embed_similarity.npz")) and exists(data.outpath(f"{i}-{j}.R.embed_similarity.npz"))):
                print(f"Missing pair {i}-{j}")
                continue
        
        bwsim_r[idx] = np.load(data.outpath(f"{i}-{j}.R.embed_similarity.npz"))[file.split(".")[-2]]

    bwsim_r.flush()    
    print(f"\t\t{file} done")


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f"\n\tEnd = {current_time}\n")
