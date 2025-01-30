#' Lattice Kriging Spatial Modeling Script
#'
#' This script performs Lattice Kriging analyses of functional gradients
#' in intrinsic cortical geometry.
#' 
#' ## Key Features
#' 1. **Environment Setup**:
#'    - Ensures `renv` environment is initialized and necessary libraries are loaded.
#'    - Sources a `setup.R` script for environment preparation if `renv.lock` is missing.
#' 2. **Parameter Configuration**:
#'    - Reads input arguments and sets model parameters (e.g., number of vertices, scale).
#' 3. **Data Preparation**:
#'    - Loads, partitions, and processes gradient and embedding data.
#'    - Selects data points within a spatial domain centered on a vertex of interest.
#' 4. **Model Fitting**:
#'    - Fits Lattice Kriging models with and without covariates.
#'    - Configures covariance matrices and spatial model parameters.
#' 5. **Prediction and Evaluation**:
#'    - Generates predictions for test data.
#'    - Computes R² and correlation metrics for model evaluation.
#' 6. **Results Saving**:
#'    - Saves parameters, model fits, and evaluation metrics to output directories.
#'
#' ## Inputs
#' - **Command-line arguments**:
#'    - `H`: Hemisphere (`"L"` for left, `"R"` for right).
#'    - `vtx`: Index of the vertex to be analyzed.
#'    - `scale`: Scaling factor for the analysis.
#' - **Data files**: Embeddings and gradients for spatial modeling.
#'
#' ## Outputs
#' - Parameters and results, including R² and correlations, saved in structured directories.
#'
#' ## Dependencies
#' Requires the following packages: `caret`, `reticulate`, `LatticeKrig`, and `renv`.
#'
#' @keywords Lattice Kriging, Spatial Modeling, Gradient Analysis


################################################################################
################################################################################
# 
# if(!file.exists("renv.lock")) {
#   cat("Running script setup.R to create the environment.\n")
#   args = commandArgs(trailingOnly=F)
#   setup_path <- dirname(sub("--file=", "", args[grep("--file=", args)]))
#   source(paste0(setup_path, "/setup.R"))
# }

# Setup environment
renv::activate()
usr.lib <- NULL
library(caret, lib.loc = usr.lib)
library(reticulate, lib.loc = usr.lib)
library(LatticeKrig, lib.loc = usr.lib)

################################################################################

# Set up parameters

# Overwrite results
ow <- TRUE

# Input arguments
args = commandArgs(trailingOnly=T)
dataset_id = as.character(args[[1]])
H = as.character(args[[2]])
vtx = as.integer(args[[3]])
scale = as.integer(args[[4]])


# Embeddings-related parameters
algorithm <- paste0("JE_cauchy", scale)
diffusion <- "a05_t1"
g <- 1
if (H == "L") {
  ivtx <- 1:9394
  nvtx <- length(ivtx)
} else {
  ivtx <- 9395:18792
  nvtx <- length(ivtx)
}

# LK-related parameters
nlevel = 1
NC = 20
NC.buffer = 1
overlap = 1.5
a.wght = 6.019
alpha = c(1)
LKGeometry = "LKBox"
max.points = NULL
mean.neighbor = 25
verbose = TRUE

################################################################################

# Set paths to relevant directories

np <- import("numpy")
vgu <- import("variograd_utils")
ds <- vgu$dataset(dataset_id)


outdir <- ds$output_dir
nsub <- ds$N
subj_list <- as.matrix(ds$subj_list)
subj_list <- subj_list[1:nsub]


################################################################################
if (!ow) {
  filename <- paste0(outdir, "/LKresults.", algorithm, "/", H,
                     "/R2/LK.searchlight.nvtx.v", vtx, ".csv")
  if (file.exists(filename)) {stop("Vertex already processed")}
}
################################################################################

# Load and set up data

# Generate train and test sets
train.idx <- createDataPartition(y = 1:nsub, p = .75, list = FALSE)
test.idx <- (1:nsub)[-train.idx]
n.train = length(train.idx)
n.test = nsub - n.train

# Load gradients
filename <- ds$outpath(paste0(ds$id, ".", H, ".FC_embeddings.a05_t1.G", g, ".csv"))
gradients <- as.matrix(read.csv(filename))

# Create group mean and aplit train/test dataset
gradients.avg <- colMeans(gradients[train.idx,])
grad.subj.avg.train <- rowMeans(gradients[train.idx,])
grad.subj.avg.test <- rowMeans(gradients[-train.idx,])
gradients.test <- gradients[-train.idx,]
gradients.test <- matrix(as.vector(t(gradients.test)), ncol = 1)
gradients.train <- gradients[train.idx,]
gradients.train <- matrix(as.vector(t(gradients.train)), ncol = 1)


# Load embeddings and reate group mean and aplit train/test dataset
embeddings <- np$load(paste0(outdir, "/All.",H,".embeddings.npz"))[[algorithm]]
embeddings <- embeddings[1:nsub, 1:nvtx, 1:3]
locations.avg <- colMeans(embeddings[train.idx,,])
locations.train <- matrix(aperm(embeddings[train.idx,,], c(2, 1, 3)),
                          nrow = n.train*nvtx, ncol = 3)
locations.test <- matrix(aperm(embeddings[-train.idx,,], c(2, 1, 3)),
                         nrow = n.test*nvtx, ncol = 3)

################################################################################

# Select data points within cube centerd on vtx

# Get median distance to closest vertex within the avg surfce
avg.dists = rdist(locations.avg)
avg.dists[avg.dists==0] <- max(avg.dists)
max.dist <- median(apply(avg.dists, 1, min)) * 2
center <- locations.avg[vtx,]

# Create masks of vertices included in this domain
sep.vecs.train <- locations.train - matrix(rep(center, each=nrow(locations.train)),
                                           nrow = nrow(locations.train))
mask.train <- rowSums(abs(sep.vecs.train) > max.dist) == 0
sep.vecs.test <- locations.test - matrix(rep(center, each=nrow(locations.test)),
                                         nrow = nrow(locations.test))
mask.test <- rowSums(abs(sep.vecs.test) > max.dist) == 0

# Apply masks
locations.train <- locations.train[mask.train,]
gradients.train <- gradients.train[mask.train]
locations.test <- locations.test[mask.test, ]
gradients.test <- gradients.test[mask.test]

################################################################################

# Compute subj-wise mean of gradient within the domain

# in the train set
subj.nvtx <- numeric()
grad.subj.locavg.train <- list()
for (i in train.idx) {
  subj.mask <- (embeddings[i,,] - matrix(rep(center, each=nvtx), ncol = 3))
  subj.mask <- rowSums(abs(subj.mask) > max.dist) == 0
  subj.nvtx <- c(subj.nvtx, sum(subj.mask))
  grad.subj.locavg.train <- append(grad.subj.locavg.train,
                                   rep(mean(gradients[i,subj.mask]),
                                       sum(subj.mask)))
}
grad.subj.locavg.train <- unlist(grad.subj.locavg.train)

# and in the test set
grad.subj.locavg.test <- list()
for (i in test.idx) {
  subj.mask <- (embeddings[i,,] - matrix(rep(center, each=nvtx), ncol = 3))
  subj.mask <- rowSums(abs(subj.mask) > max.dist) == 0
  grad.subj.locavg.test <- append(grad.subj.locavg.test,
                                  rep(mean(gradients[i,subj.mask]),
                                      sum(subj.mask)))
}
grad.subj.locavg.test <- unlist(grad.subj.locavg.test)

# Create locally centered data
gradients.train.ctr <- gradients.train - grad.subj.locavg.train
gradients.test.ctr <- gradients.test - grad.subj.locavg.test

# Compute group averages for vertex index
group.avg.train <- rep(gradients.avg, n.train)[mask.train]
group.avg.test <- rep(gradients.avg, n.test)[mask.test]

# Compute individual average
subj.avg.train <- rep(grad.subj.avg.train, each=nvtx)[mask.train]
subj.avg.test <-rep(grad.subj.avg.test, each=nvtx)[mask.test]

################################################################################
################################################################################

# Create and fit spatial models

# Set covariance matrix as individual local average
Z.train =  matrix(grad.subj.locavg.train, ncol=1)
Z.train <- (Z.train - mean(Z.train)) / sd(Z.train)
Z.test =  matrix(grad.subj.locavg.test, ncol=1)
Z.test <- (Z.test - mean(Z.test)) / sd(Z.test)

# Setup and fit LK model
LKinfo <- LKrigSetup(locations.train,
                     nlevel = nlevel,
                     NC = NC,
                     NC.buffer = NC.buffer,
                     overlap = overlap,
                     a.wght = a.wght,
                     alpha = alpha,
                     LKGeometry = LKGeometry,
                     max.points = max.points,
                     mean.neighbor = mean.neighbor,
                     verbose = verbose)
LKinfo$setupArgs <- list() # Necessary for findAwght for unclear reason

# Model without covariates
LKfit <- LatticeKrig(locations.train,
                     gradients.train,
                     LKinfo = LKinfo,
                     verbose = verbose,
                     normalize = TRUE)

# Model with covariates
LKfit.Z <- LatticeKrig(locations.train,
                       gradients.train,
                       Z = Z.train,
                       LKinfo = LKinfo,
                       verbose = TRUE,
                       normalize = TRUE)

cat("")
cat("Model fitting complete\n")
################################################################################
################################################################################

# Generate predictions


# Predict test locations
predict.test <- predict.LKrig(LKfit, xnew = locations.test)
predict.test.Z.full <- predict.LKrig(LKfit.Z, xnew = locations.test,
                                     Znew =Z.test)
predict.test.Z.fixed <- predict.LKrig(LKfit.Z, xnew = locations.test,
                                      Znew = Z.test,
                                      just.fixed=TRUE)
predict.test.Z.avg <- Z.test * LKfit.Z$d.coef["Z1",] + LKfit.Z$d.coef["Intercept",] * mean(gradients.test)
predict.train.Z.fixed <- predict.LKrig(LKfit.Z, xnew = locations.train,
                                      Znew = Z.train,
                                      just.fixed=TRUE)
predict.train.Z.avg <- Z.train * LKfit.Z$d.coef["Z1",] + LKfit.Z$d.coef["Intercept",] * mean(gradients.train)

################################################################################

# Calculate R2

# Helper for computing R2
rsquare <- function(y, y_hat) {
  ss.total <- sum((y - mean(y))^2)
  ss.resid <- sum((y - y_hat)^2)
  r2 = 1 - (ss.resid/ ss.total)
  return(r2)
}


# On the train set
r2.train <- rsquare(gradients.train, LKfit$fitted.values)
r2.train.gavg <- rsquare(gradients.train, group.avg.train)
r2.train.savg <- rsquare(gradients.train, grad.subj.locavg.train)
r2.train.Z.full <- rsquare(gradients.train, LKfit.Z$fitted.values)
r2.train.Z.fixed <- rsquare(gradients.train, predict.train.Z.fixed)
r2.train.Z.avg <- rsquare(gradients.train, predict.train.Z.avg)

# On the test set
r2.test <- rsquare(gradients.test,  predict.test)
r2.test.gavg <- rsquare(gradients.test, group.avg.test)
r2.test.savg <- rsquare(gradients.test, grad.subj.locavg.test)
r2.test.Z.full <- rsquare(gradients.test, predict.test.Z.full)
r2.test.Z.fixed <- rsquare(gradients.test, predict.test.Z.fixed)
r2.test.Z.avg <- rsquare(gradients.test, predict.test.Z.avg)

cat("")
cat("Computed r square\n")
################################################################################

# Calculate R

# On the train set
cor.train <- cor(gradients.train, LKfit$fitted.values)
cor.train.gavg <- cor(gradients.train, group.avg.train)
cor.train.savg <- cor(gradients.train, grad.subj.locavg.train)
cor.train.Z.full <- cor(gradients.train, LKfit.Z$fitted.values)
cor.train.Z.fixed <- cor(gradients.train, predict.train.Z.fixed)
cor.train.Z.avg <- cor(gradients.train, predict.train.Z.avg)

# On the test set
cor.test <- cor(gradients.test,  predict.test)
cor.test.gavg <- cor(gradients.test, group.avg.test)
cor.test.savg <- cor(gradients.test, grad.subj.locavg.test)
cor.test.Z.full <- cor(gradients.test, predict.test.Z.full)
cor.test.Z.fixed <- cor(gradients.test, predict.test.Z.fixed)
cor.test.Z.avg <- cor(gradients.test, predict.test.Z.avg)

cat("")
cat("Computed correlation\n")

################################################################################
################################################################################

# Save output

# Strip heavy data

LKfit$x = list()
LKfit$y = list()
LKfit$Z = list()
LKfit$residuals = list()
LKfit$weights = list()

LKfit.Z$x = list()
LKfit.Z$y = list()
LKfit.Z$Z = list()
LKfit.Z$residuals = list()
LKfit.Z$weights = list()

################################################################################

# Save output
outpath <- paste0(outdir, "/LKresults.", algorithm)
if (!dir.exists(outpath)) {dir.create(outpath)}

outpath <- paste0(outpath, "/", H)
if (!dir.exists(outpath)) {
  dir.create(outpath)
  dir.create(paste0(outpath, "/Cors"))
  dir.create(paste0(outpath, "/R2"))
  dir.create(paste0(outpath, "/MLEs"))
  dir.create(paste0(outpath, "/LKmodels"))
  dir.create(paste0(outpath, "/NVertex"))
}


params <- data.frame(
  vtx = vtx,
  scale = scale,
  algorithm = algorithm,
  diffusion = diffusion,
  g = g,
  nvtx = nvtx,
  nsub = nsub,
  nlevel = nlevel,
  NC = NC,
  NC.buffer = NC.buffer,
  overlap = overlap,
  a.wght = a.wght,
  alpha = alpha,
  LKGeometry = LKGeometry,
  mean.neighbor = mean.neighbor
)
nameout <- paste0(outpath, "/parameters.csv")
if (!exists(nameout)) {
  write.csv(params, nameout)
}


MLE.df <- t(data.frame(LKfit = LKfit$MLE$summary,
                       LKfit.Z = LKfit.Z$MLE$summary))
nameout <- paste0(outpath, "/MLEs/LK.searchlight.MLE.v", vtx, ".csv")
write.csv(MLE.df, nameout)


R2.df <- data.frame(r2.train=r2.train,
                    r2.train.gavg=r2.train.gavg,
                    r2.train.savg=r2.train.savg,
                    r2.train.Z.full=r2.train.Z.full,
                    r2.train.Z.fixed=r2.train.Z.fixed,
                    r2.train.Z.avg=r2.train.Z.avg,
                    r2.test=r2.test,
                    r2.test.gavg=r2.test.gavg,
                    r2.test.savg=r2.test.savg,
                    r2.test.Z.full=r2.test.Z.full,
                    r2.test.Z.fixed=r2.test.Z.fixed,
                    r2.test.Z.avg=r2.test.Z.avg)
nameout <- paste0(outpath, "/R2/LK.searchlight.R2.v", vtx, ".csv")
write.csv(R2.df, nameout, row.names=FALSE)


cor.df <- data.frame(cor.train=cor.train,
                    cor.train.gavg=cor.train.gavg,
                    cor.train.savg=cor.train.savg,
                    cor.train.Z.full=cor.train.Z.full,
                    cor.train.Z.fixed=cor.train.Z.fixed,
                    cor.train.Z.avg=cor.train.Z.avg,
                    cor.test=cor.test,
                    cor.test.gavg=cor.test.gavg,
                    cor.test.savg=cor.test.savg,
                    cor.test.Z.full=cor.test.Z.full,
                    cor.test.Z.fixed=cor.test.Z.fixed,
                    cor.test.Z.avg=cor.test.Z.avg)
nameout <- paste0(outpath, "/Cors/LK.searchlight.cor.v", vtx, ".csv")
write.csv(cor.df, nameout, row.names = FALSE)


nameout <- paste0(outpath, "/NVertex/LK.searchlight.nvtx.v", vtx, ".csv")
write.csv(data.frame(subj.nvtx), nameout, row.names=FALSE)


# LKmodels <- list(LKfit=LKfit, LKfit.Z=LKfit.Z)
# nameout <- paste0(outpath, "/LKmodels/LK.searchlight.LKmdl.v", vtx, ".RData")
# save(LKmodels, file=nameout)


cat("")
cat("Output saved\n")
