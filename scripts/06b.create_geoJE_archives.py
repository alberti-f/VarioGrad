import os
import numpy as np
from variograd_utils.core_utils import dataset, subject, npz_update
from variograd_utils.brain_utils import vertex_info_10k

n_components = 20
data = dataset()


# preassign empty arrays
zeroarrayl = np.zeros([data.N, vertex_info_10k.grayl.size, n_components])
zeroarrayr = np.zeros([data.N, vertex_info_10k.grayr.size, n_components])


# add subject embeddings to the group dictionary
print("Building group archive of embeddings:")
embeddings_l = {}
embeddings_r = {}
for i, ID in enumerate(data.subj_list):

    subj = subject(ID)

    embeddings_subj = np.load(subj.outpath(f"{ID}.L.embeddings.npz"))
    for key, value in embeddings_subj.items():
        if key not in embeddings_l:
            embeddings_l[key] = zeroarrayl.copy()
        embeddings_l[key][i] = value


    embeddings_subj = np.load(subj.outpath(f"{ID}.R.embeddings.npz"))
    for key, value in embeddings_subj.items():
        if key not in embeddings_r:
            embeddings_r[key] = zeroarrayr.copy()
        embeddings_r[key][i] = value

    print(f"\tAdded subject {i+1} of {data.N}")
    

# save group embeddings
npz_update(data.outpath("All.L.embeddings.npz"), embeddings_l)
npz_update(data.outpath("All.R.embeddings.npz"), embeddings_r)
print("\nArchives saved")


if os.path.exists(subject(ID).outpath(f"{ID}.L.embeddings.npz")) and os.path.exists(subject(ID).outpath(f"{ID}.R.embeddings.npz")):
    for ID in data.subj_list:
        os.remove(subject(ID).outpath(f"{ID}.L.embeddings.npz"))
        os.remove(subject(ID).outpath(f"{ID}.R.embeddings.npz"))

print("\nSubject embeddings removed")
