from variograd_utils import *
import numpy as np
import os


n_components = 10
data = dataset()


# preassign empty arrays
zeroarrayl = np.zeros([data.N, vertex_info_10k.grayl.size, n_components])
zeroarrayr = np.zeros([data.N, vertex_info_10k.grayr.size, n_components])


# add subject embeddings to the group dictionary
embeddings_l = {}
embeddings_r = {}
for i, id in enumerate(data.subj_list):

    subj = subject(id)

    embeddings_subj = np.load(subj.outpath(f"{id}.L.embeddings.npz"))
    for key, value in embeddings_subj.items():
        if key not in embeddings_l:
            embeddings_l[key] = zeroarrayl.copy()
        embeddings_l[key][i] = value


    embeddings_subj = np.load(subj.outpath(f"{id}.R.embeddings.npz"))
    for key, value in embeddings_subj.items():
        if key not in embeddings_r:
            embeddings_r[key] = zeroarrayr.copy()
        embeddings_r[key][i] = value
    

# save group embeddings
npz_update(data.outpath("All.L.embeddings.npz"), embeddings_l)
npz_update(data.outpath("All.R.embeddings.npz"), embeddings_r)


if os.path.exists(subject(id).outpath(f"{id}.L.embeddings.npz")) and os.path.exists(subject(id).outpath(f"{id}.R.embeddings.npz")):
    for id in data.subj_list:
        os.remove(subject(id).outpath(f"{id}.L.embeddings.npz"))
        os.remove(subject(id).outpath(f"{id}.R.embeddings.npz"))