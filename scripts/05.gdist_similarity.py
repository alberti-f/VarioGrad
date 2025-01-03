"""
Compute Inter-Subject Similarity of Vertex Geodesic Distance Profiles.

Steps:
    1. Generate all unique subject pairs from the subject list.
    2. For each hemisphere (`L` and `R`):
        - Load geodesic distance matrices for the two subjects in each pair.
        - Compute the diagonal of the pairwise Euclidean distance matrixbetween its rows.
        - Save the resulting matrix as a `.npy` file.
    3. Print the output file paths and the dimensions of the saved matrices.

Dependencies:
    - `variograd_utils`: For dataset and subject handling.
    - `numpy`: For numerical computations and file handling.
    - `scikit-learn`: For computing pairwise Euclidean distances.
    - `joblib`: For parallel processing of subject pairs.

Outputs:
    - Pairwise similarity matrices for left and right hemispheres:
        `<output_dir>/AllToAll.L.gdist_L2.npy`
        `<output_dir>/AllToAll.R.gdist_L2.npy`

Notes:
    - The script uses all available CPU cores for parallel processing (`n_jobs=-1`).
    - Ensure that geodesic distance matrices are precomputed and available for all subjects.
    - The outputs contain the diagonal of the Euclidean distance for each subject pair.

"""


from itertools import combinations
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances
from variograd_utils import dataset, subject


data =  dataset()

subj_pairs = list(combinations(data.subj_list, 2))

print("\n\nComputing the inter-subject similarity of vertex geodesic\n"
      + f"distance profile for {len(subj_pairs)} subject pairs\n")

# Left hemisphere
gdist_L2_l = Parallel(n_jobs=-1)(
    delayed(np.diag)(euclidean_distances(subject(i).load_gdist_matrix("L"),
                                         subject(j).load_gdist_matrix("L")))
                                         for i, j in subj_pairs)
filename = data.outpath("AllToAll.L.gdist_L2.npy")
np.save(filename, np.asarray(gdist_L2_l).T)
print(f"Left hemisphere output saved at {filename} \n", 
      f"Matrix size: {np.asarray(gdist_L2_l).T.shape} \n")

# Right hemisphere
gdist_L2_r = Parallel(n_jobs=-1)(delayed(np.diag)(
    euclidean_distances(subject(i).load_gdist_matrix("R"),
                        subject(j).load_gdist_matrix("R")))
                        for i, j in subj_pairs)
filename = data.outpath("AllToAll.R.gdist_L2.npy")
np.save(filename, np.asarray(gdist_L2_r).T)
print(f"Right hemisphere output saved at {filename} \n",
      f"Matrix size: {np.asarray(gdist_L2_r).T.shape} \n")
