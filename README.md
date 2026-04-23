<div align="center">

# Intrinsic cortical geometry is associated with individual differences in local functional organization
F. Alberti, P.-L. Bazin, R. A. Benn, R. Scholz, W. Wei, A. Holmes,
V. Shevchenko, U. Klatzmann, C. Pallavicini, R. Leech, D. S. Margulies      
[Preprint here](https://doi.org/10.21203/rs.3.rs-9200088/v1)


</div>

## Overview
This project investigates whether individual differences in cortical geometry explain variability in local functional organization. Resting-state fMRI data are summarized using vertex-wise functional connectivity gradients. Individual cortical surfaces are projected into a shared geometric embedding derived from geodesic distance between vertices. This provides a common space where distance reflects how a vertex location on the cortex changes across individuals. Local spatial models are then fitted to assess whether vertex position in this embedding is associated with interindividual differences in vertex function. Analyses are performed across spatial scales and validated using group-average and permutation controls.

## Dependencies
[![GitHub License](https://img.shields.io/github/license/alberti-f/VarioGrad)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; ">
<div>

### Python
- numpy
- scipy
- scikit-learn>=1.1.3
- joblib>=1.4.2
- nibabel>=4.0.2
- nilearn>=0.10.4
- hcp_utils==0.1.0
- surfdist @ git+https://github.com/alberti-f/surfdist.git

</div>
<div>

### R
- LatticeKrig
- caret
- reticulate
- abind
- jsonlite

</div>
<div>

### Other
- [wb_command](https://humanconnectome.org/software/connectome-workbench) (Connectome Workbench command line tools)
</div>
</div>
