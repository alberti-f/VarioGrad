"""
This script collects individual geometric embeddings into group-level archives for 
the left and right hemispheres.

Steps:
    1. Pre-allocate empty arrays for storing group-level embeddings for each hemisphere.
    2. Loop through the subject list:
        - Load individual subject embeddings.
        - Add each subject's embeddings to the group-level arrays.
        - Print progress for each subject.
    3. Save the consolidated group embeddings as `.npz` files for both hemispheres.
    4. If the subject embeddings still exist, delete the individual `.npz` files to save space.

Dependencies:
    - `variograd_utils`: For dataset and subject handling, and file update utilities.
    - `numpy`: For numerical computations and file handling.

Outputs:
    - Group-level archive of embeddings:
        `<output_dir>/All.L.embeddings.npz`
        `<output_dir>/All.R.embeddings.npz`

Notes:
    - The group embeddings are saved as `.npz` archives, with keys corresponding to 
      different parameter combinations used during embedding computation.
    - The script deletes individual embeddings after successful consolidation to save space.

"""


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


# remove individual embeddings
exists_L = os.path.exists(subject(ID).outpath(f"{ID}.L.embeddings.npz"))
exists_R = os.path.exists(subject(ID).outpath(f"{ID}.R.embeddings.npz"))
if  exists_L and exists_R:
    for ID in data.subj_list:
        os.remove(subject(ID).outpath(f"{ID}.L.embeddings.npz"))
        os.remove(subject(ID).outpath(f"{ID}.R.embeddings.npz"))

print("\nSubject embeddings removed")
