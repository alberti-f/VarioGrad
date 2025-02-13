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
library(abind)

################################################################################

# Set up parameters

# Overwrite results
ow <- TRUE

# Input arguments
# dataset_id = "train"
# H = "L"
# vtx = 4612
# scale = 2
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
verbose = FALSE

################################################################################

# Set paths to relevant directories

np <- import("numpy")
vgu <- import("variograd_utils")
ds <- vgu$dataset(dataset_id)


outdir <- ds$output_dir
subj_list_all <- as.matrix(ds$subj_list)
subj_list <- as.matrix(np$loadtxt(ds$outpath(paste0(ds$id,".fixed_model_subj_IDs.npy"))))
subj_mask <- subj_list_all %in% subj_list
nsub <- length(subj_list)

################################################################################
if (!ow) {
  filename <- paste0(outdir, "/LKresults.", algorithm, "/", H,
                     "/R2/LK.searchlight.nvtx.v", vtx, ".csv")
  if (file.exists(filename)) {stop("Vertex already processed")}
}
################################################################################

# Load and set up data

# Generate train and test sets
set.seed(123)
train.idx <- createDataPartition(y = 1:nsub, p = .75, list = FALSE)
test.idx <- test.idx <- setdiff(1:nsub, train.idx) 
n.train = length(train.idx)
n.test = nsub - n.train


# Load embeddings and reate group mean and split train/test dataset
# subj.idx <- matrix(rep(1:ds$N, nvtx), nrow = ds$N, ncol = nvtx)
embeddings <- np$load(paste0(outdir, "/All.",H,".embeddings.npz"))[[algorithm]]
# embeddings <- abind(embeddings[,,1:3], subj.idx, along = 3)
embeddings <- embeddings[subj_mask,,1:3]
locations.avg <- colMeans(embeddings[train.idx,,])
locations.train <- matrix(aperm(embeddings[train.idx,,], c(2, 1, 3)),
                          nrow = n.train*nvtx, ncol = 3)
locations.test <- matrix(aperm(embeddings[-train.idx,,], c(2, 1, 3)),
                         nrow = n.test*nvtx, ncol = 3)


# Load gradients and reate group mean and aplit train/test dataset
filename <- ds$outpath(paste0(ds$id, ".", H, ".FC_embeddings.a05_t1.G", g, ".csv"))
gradients <- as.matrix(read.csv(filename))[subj_mask,]
gradients.avg <- apply(gradients[train.idx,], 2, median)
gradients.train <- matrix(t(gradients[train.idx,]), ncol=1)
grad.subj.avg.train <- apply(gradients[train.idx,], 1, median)
grad.grp.avg.train <- rep(gradients.avg, n.train)
gradients.test <- matrix(t(gradients[test.idx,]), ncol=1)
grad.subj.avg.test <- apply(gradients[test.idx,], 1, median)
grad.grp.avg.test <- rep(gradients.avg, n.test)


# Load covariates
covariates <- np$load(ds$outpath(paste0(ds$id, ".fixed_model.",H,".npy")))
covariates.train <- matrix(aperm(covariates[train.idx,,], c(2, 1, 3)),
                          nrow = n.train*nvtx, ncol = dim(covariates)[3])
covariates.test <- matrix(aperm(covariates[test.idx,,], c(2, 1, 3)),
                          nrow = n.test*nvtx, ncol = dim(covariates)[3])
covs.names <- np$loadtxt(ds$outpath(paste0(ds$id, ".fixed_model_features",".npy")), dtype="str")

################################################################################

# Select data points within cube centerd on vtx

# Get median distance to closest vertex within the avg surfce
avg.dists = rdist(locations.avg)
avg.dists[avg.dists==0] <- max(avg.dists)
sl.side <- median(apply(avg.dists, 1, min)) * 3
center <- locations.avg[vtx,]

# Create masks of vertices included in this domain
mask.train <- rowSums(abs(t(t(locations.train) - center)) > (sl.side/2)) == 0
mask.test <- rowSums(abs(t(t(locations.test) - center)) > (sl.side/2)) == 0

# Apply masks
locations.train <- locations.train[mask.train,]
gradients.train <- gradients.train[mask.train]
covariates.train <- covariates.train[mask.train,]
subj.idx.train <- rep(train.idx, each=nvtx)[mask.train]
expand.row.train <- match(subj.idx.train, sort(unique(subj.idx.train)))
locations.test <- locations.test[mask.test,]
gradients.test <- gradients.test[mask.test]
covariates.test <- covariates.test[mask.test,]
subj.idx.test <- rep(test.idx, each=nvtx)[mask.test]
expand.row.test <- match(subj.idx.test, sort(unique(subj.idx.test)))

################################################################################

# Compute subj-wise mean of gradient within the domain

# in the train set
subj.nvtx <- numeric()
grad.subj.locavg.train <- numeric()
covs.subj.locavg.train <- array(0, dim=c(length(unique(subj.idx.train)), dim(covariates.train)[2]))
for (i in 1:length(unique(subj.idx.train))) {
    subj.idx = unique(subj.idx.train)[i]
    subj.nvtx[i] <- sum(subj.idx.train==subj.idx)
    if (subj.nvtx[i]>0) {        
        grad.subj.locavg.train[i] <- median(gradients.train[subj.idx.train==subj.idx])
        covars.tmp <- covariates.train[subj.idx.train==subj.idx,]
        if (is.null(dim(covars.tmp))) {covs.subj.locavg.train[i,] <- covars.tmp; next}
        covs.subj.locavg.train[i,] <- apply(covars.tmp, 2, median)
    }  
}
# grad.subj.locavg.train <- rep(grad.subj.avg.train, each=nvtx)[mask.train]
# covs.subj.locavg.train <- matrix(rep(covs.subj.locavg.train, each=nvtx),
#                                  ncol=dim(covariates.train)[2])[mask.train,]

# and in the test set
grad.subj.locavg.test <- numeric()
covs.subj.locavg.test <- array(0, dim=c(length(unique(subj.idx.test)), dim(covariates.test)[2]))
for (i in 1:length(unique(subj.idx.test))) {
    subj.idx = unique(subj.idx.test)[i]
    if (sum(subj.idx.test==subj.idx)>0) {        
        grad.subj.locavg.test[i] <- median(gradients.test[subj.idx.test==subj.idx])
        covars.tmp <- covariates.test[subj.idx.test==subj.idx,]
        if (is.null(dim(covars.tmp))) {covs.subj.locavg.test[i,] <- covars.tmp; next}
        covs.subj.locavg.test[i,] <- apply(covars.tmp, 2, median)
    }  
}
# grad.subj.locavg.test <- rep(grad.subj.avg.test, each=nvtx)[mask.test]
# covs.subj.locavg.test <- matrix(rep(covs.subj.locavg.test, each=nvtx),
#                                 ncol=dim(covariates.test)[2])[mask.test,]

# in the train set
# subj.nvtx <- numeric()
# grad.subj.locavg.train <- list()
# for (i in train.idx) {
#   subj.mask <- (embeddings[i,,] - matrix(rep(center, each=nvtx), ncol = 3))
#   subj.mask <- rowSums(abs(subj.mask) > sl.side) == 0
#   subj.nvtx <- c(subj.nvtx, sum(subj.mask))
#   grad.subj.locavg.train <- append(grad.subj.locavg.train,
#                                    rep(mean(gradients[i,subj.mask]),
#                                        sum(subj.mask)))
# }
# grad.subj.locavg.train <- unlist(grad.subj.locavg.train)

# and in the test set
# grad.subj.locavg.test <- list()
# for (i in test.idx) {
#   subj.mask <- (embeddings[i,,] - matrix(rep(center, each=nvtx), ncol = 3))
#   subj.mask <- rowSums(abs(subj.mask) > sl.side) == 0
#   grad.subj.locavg.test <- append(grad.subj.locavg.test,
#                                   rep(mean(gradients[i,subj.mask]),
#                                       sum(subj.mask)))
# }
# grad.subj.locavg.test <- unlist(grad.subj.locavg.test)

# Compute group averages for vertex index
grad.grp.avg.train <- grad.grp.avg.train[mask.train]
grad.grp.avg.test <- grad.grp.avg.test[mask.test]

################################################################################
################################################################################

# Covariates

# Normalize covariates based on train and convert to df
colnames(covs.subj.locavg.train) <- covs.names
covs.mean.train <- colMeans(covs.subj.locavg.train)
covs.sd.train <- apply(covs.subj.locavg.train, 2, sd)
covs.subj.locavg.train <- scale(covs.subj.locavg.train, center = covs.mean.train, scale = covs.sd.train)
covs.subj.locavg.train <- as.data.frame(covs.subj.locavg.train)

colnames(covs.subj.locavg.test) <- covs.names
covs.subj.locavg.test <- scale(covs.subj.locavg.test, center = covs.mean.train, scale = covs.sd.train)
covs.subj.locavg.test <- as.data.frame(covs.subj.locavg.test)

# covariates.train <- scale(covariates.train[mask.train,])
# colnames(covariates.train) <- covs.names
# covariates.test <- scale(covariates.test[mask.test,])
# colnames(covariates.test) <- covs.names

################################################################################
################################################################################

# Calculate residuals of covariates

covs.subj.locavg.train$gradients <- grad.subj.locavg.train 

lm.covs.train <- lm(gradients ~ ., data = covs.subj.locavg.train)
grads.train.fit.covs <- lm.covs.train$fitted.values
grads.train.res.covs <- lm.covs.train$residuals
grads.test.fit.covs <- predict(lm.covs.train, newdata = covs.subj.locavg.test)
grads.test.res.covs <- gradients.test - grads.test.fit.covs[expand.row.test]

grads.train.res.loc <- gradients.train - grad.subj.locavg.train[expand.row.train]
grads.test.res.loc <- gradients.test - grad.subj.locavg.test[expand.row.test]

################################################################################
################################################################################

# Create and fit spatial models

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

# Model with local averages
LKfit.locavg <- LatticeKrig(locations.train,
                       grads.train.res.loc[expand.row.train],
                       LKinfo = LKinfo,
                       verbose = verbose,
                       normalize = TRUE)

# Model with covariates
LKfit.covs <- LatticeKrig(locations.train,
                          grads.train.res.covs[expand.row.train],
                          LKinfo = LKinfo,
                          verbose = verbose,
                          normalize = TRUE)

cat("")
cat("Model fitting complete\n")
################################################################################
################################################################################

# Generate predictions

# Predict train locations
predict.train.locavg.full <- grad.subj.locavg.train[expand.row.train] + LKfit.locavg$fitted.values
predict.train.locavg.fixed <- grad.subj.locavg.train[expand.row.train] + predict.LKrig(LKfit.locavg, xnew = locations.train, just.fixed=TRUE)
predict.train.locavg.avg <- grad.subj.locavg.train[expand.row.train]

predict.train.covs.full <- grads.train.fit.covs[expand.row.train] + LKfit.covs$fitted.values
predict.train.covs.fixed <- grads.train.fit.covs[expand.row.train] + predict.LKrig(LKfit.covs, xnew = locations.train, just.fixed=TRUE)
predict.train.covs.avg <- grads.train.fit.covs[expand.row.train]

# Predict test locations
predict.test <- predict.LKrig(LKfit, xnew = locations.test)

predict.test.locavg.full <- grad.subj.locavg.test[expand.row.test] + predict.LKrig(LKfit.locavg, xnew = locations.test)
predict.test.locavg.fixed <- grad.subj.locavg.test[expand.row.test] + predict.LKrig(LKfit.locavg, xnew = locations.test, just.fixed=TRUE)
predict.test.locavg.avg <- grad.subj.locavg.test[expand.row.test]

predict.test.covs.full <- grads.test.fit.covs[expand.row.test] + predict.LKrig(LKfit.covs, xnew = locations.test)
predict.test.covs.fixed <- grads.test.fit.covs[expand.row.test] + predict.LKrig(LKfit.covs, xnew = locations.test, just.fixed=TRUE)
predict.test.covs.avg <- grads.test.fit.covs[expand.row.test]

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
r2.train.gavg <- rsquare(gradients.train, grad.grp.avg.train)
r2.train.savg <- rsquare(gradients.train, grad.subj.locavg.train[expand.row.train])

r2.train.locavg.full <- rsquare(gradients.train, predict.train.locavg.full)
r2.train.locavg.fixed <- rsquare(gradients.train, predict.train.locavg.fixed)
r2.train.locavg.avg <- rsquare(gradients.train, predict.train.locavg.avg)

r2.train.covs.full <- rsquare(gradients.train, predict.train.covs.full)
r2.train.covs.fixed <- rsquare(gradients.train, predict.train.covs.fixed)
r2.train.covs.avg <- rsquare(gradients.train, predict.train.covs.avg)

# On the test set
r2.test <- rsquare(gradients.test,  predict.test)
r2.test.gavg <- rsquare(gradients.test, grad.grp.avg.test)
r2.test.savg <- rsquare(gradients.test, grad.subj.locavg.test[expand.row.test])

r2.test.locavg.full <- rsquare(gradients.test, predict.test.locavg.full)
r2.test.locavg.fixed <- rsquare(gradients.test, predict.test.locavg.fixed)
r2.test.locavg.avg <- rsquare(gradients.test, predict.test.locavg.avg)

r2.test.covs.full <- rsquare(gradients.test, predict.test.covs.full)
r2.test.covs.fixed <- rsquare(gradients.test, predict.test.covs.fixed)
r2.test.covs.avg <- rsquare(gradients.test, predict.test.covs.avg)

cat("")
cat("Computed r square\n")
################################################################################

# Calculate R

# On the train set
cor.train <- cor(gradients.train, LKfit$fitted.values)
cor.train.gavg <- cor(gradients.train, grad.grp.avg.train)
cor.train.savg <- cor(gradients.train, predict.train.locavg.avg)

cor.train.locavg.full <- cor(gradients.train, predict.train.locavg.full)
cor.train.locavg.fixed <- cor(gradients.train, predict.train.locavg.fixed)
cor.train.locavg.avg <- cor(gradients.train, predict.train.locavg.avg)

cor.train.covs.full <- cor(gradients.train, predict.train.covs.full)
cor.train.covs.fixed <- cor(gradients.train, predict.train.covs.fixed)
cor.train.covs.avg <- cor(gradients.train, predict.train.covs.avg)

# On the test set
cor.test <- cor(gradients.test,  predict.test)
cor.test.gavg <- cor(gradients.test, grad.grp.avg.test)
cor.test.savg <- cor(gradients.test, predict.test.locavg.avg)

cor.test.locavg.full <- cor(gradients.test, predict.test.locavg.full)
cor.test.locavg.fixed <- cor(gradients.test, predict.test.locavg.fixed)
cor.test.locavg.avg <- cor(gradients.test, predict.test.locavg.avg)

cor.test.covs.full <- cor(gradients.test, predict.test.covs.full)
cor.test.covs.fixed <- cor(gradients.test, predict.test.covs.fixed)
cor.test.covs.avg <- cor(gradients.test, predict.test.covs.avg)

cat("")
cat("Computed correlation\n")

################################################################################
################################################################################

# Save output

# Strip heavy data

LKfit$x = list()
LKfit$y = list()
LKfit$locavg = list()
LKfit$residuals = list()
LKfit$weights = list()

LKfit.locavg$x = list()
LKfit.locavg$y = list()
LKfit.locavg$Z = list()
LKfit.locavg$residuals = list()
LKfit.locavg$weights = list()

################################################################################

# Save output
outpath <- paste0(outdir, "/LKresults.", algorithm, ".covs")
if (!dir.exists(outpath)) {dir.create(outpath)}

outpath <- paste0(outpath, "/", H)
if (!dir.exists(outpath)) {
    dir.create(outpath)
    dir.create(paste0(outpath, "/Cors"))
    dir.create(paste0(outpath, "/R2"))
    dir.create(paste0(outpath, "/MLEs"))
    dir.create(paste0(outpath, "/LKmodels"))
    dir.create(paste0(outpath, "/NVertex"))
    dir.create(paste0(outpath, "/Coeffs"))
    dir.create(paste0(outpath, "/Predictions"))
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
                       LKfit.locavg = LKfit.locavg$MLE$summary,
                       LKfit.covs = LKfit.covs$MLE$summary))
nameout <- paste0(outpath, "/MLEs/LK.searchlight.MLE.v", vtx, ".csv")
write.csv(MLE.df, nameout)


R2.df <- data.frame(r2.train=r2.train,
                    r2.train.gavg=r2.train.gavg,
                    r2.train.savg=r2.train.savg,
                    r2.train.locavg.full=r2.train.locavg.full,
                    r2.train.locavg.fixed=r2.train.locavg.fixed,
                    r2.train.locavg.avg=r2.train.locavg.avg,
                    r2.train.covs.full=r2.train.covs.full,
                    r2.train.covs.fixed=r2.train.covs.fixed,
                    r2.train.covs.avg=r2.train.covs.avg,
                    r2.test=r2.test,
                    r2.test.gavg=r2.test.gavg,
                    r2.test.savg=r2.test.savg,
                    r2.test.locavg.full=r2.test.locavg.full,
                    r2.test.locavg.fixed=r2.test.locavg.fixed,
                    r2.test.locavg.avg=r2.test.locavg.avg,
                    r2.test.covs.full=r2.test.covs.full,
                    r2.test.covs.fixed=r2.test.covs.fixed,
                    r2.test.covs.avg=r2.test.covs.avg)
nameout <- paste0(outpath, "/R2/LK.searchlight.R2.v", vtx, ".csv")
write.csv(R2.df, nameout, row.names=FALSE)


cor.df <- data.frame(cor.train=cor.train,
                     cor.train.gavg=cor.train.gavg,
                     cor.train.savg=cor.train.savg,
                     cor.train.locavg.full=cor.train.locavg.full,
                     cor.train.locavg.fixed=cor.train.locavg.fixed,
                     cor.train.locavg.avg=cor.train.locavg.avg,
                     cor.train.covs.full=cor.train.covs.full,
                     cor.train.covs.fixed=cor.train.covs.fixed,
                     cor.train.covs.avg=cor.train.covs.avg,
                     cor.test=cor.test,
                     cor.test.gavg=cor.test.gavg,
                     cor.test.savg=cor.test.savg,
                     cor.test.locavg.full=cor.test.locavg.full,
                     cor.test.locavg.fixed=cor.test.locavg.fixed,
                     cor.test.locavg.avg=cor.test.locavg.avg,
                     cor.test.covs.full=cor.test.covs.full,
                     cor.test.covs.fixed=cor.test.covs.fixed,
                     cor.test.covs.avg=cor.test.covs.avg)
nameout <- paste0(outpath, "/Cors/LK.searchlight.cor.v", vtx, ".csv")
write.csv(cor.df, nameout, row.names = FALSE)


predict.train.df <- data.frame(subject=subj.idx.train,
                               vertex=rep(1:nvtx, n.train)[mask.train],
                               true.train=gradients.train,
                               predict.train=LKfit$fitted.values,
                               predict.train.locavg.full=predict.train.locavg.full,
                               predict.train.locavg.fixed=predict.train.locavg.fixed,
                               predict.train.locavg.avg=predict.train.locavg.avg,
                               predict.train.covs.full=predict.train.covs.full,
                               predict.train.covs.fixed=predict.train.covs.fixed,
                               predict.train.covs.avg=predict.train.covs.avg)
nameout <- paste0(outpath, "/Predictions/LK.searchlight.train.v", vtx, ".csv")
write.csv(predict.train.df, nameout, row.names = FALSE)


predict.test.df <- data.frame(subject=subj.idx.test,
                              vertex=rep(1:nvtx, n.test)[mask.test],
                              true.train=gradients.test,
                              predict.test=predict.test,
                              predict.test.locavg.full=predict.test.locavg.full,
                              predict.test.locavg.fixed=predict.test.locavg.fixed,
                              predict.test.locavg.avg=predict.test.locavg.avg,
                              predict.test.covs.full=predict.test.covs.full,
                              predict.test.covs.fixed=predict.test.covs.fixed,
                              predict.test.covs.avg=predict.test.covs.avg)
nameout <- paste0(outpath, "/Predictions/LK.searchlight.test.v", vtx, ".csv")
write.csv(predict.test.df, nameout, row.names = FALSE)


nameout <- paste0(outpath, "/NVertex/LK.searchlight.nvtx.v", vtx, ".csv")
write.csv(data.frame(subj.nvtx), nameout, row.names=FALSE)

nameout <- paste0(outpath, "/Coeffs/LK.searchlight.coeffs.v", vtx, ".csv")
write.csv(t(data.frame(lm.covs.train$coefficients)), nameout, row.names=FALSE)


# LKmodels <- list(LKfit=LKfit, LKfit.locavg=LKfit.locavg)
# nameout <- paste0(outpath, "/LKmodels/LK.searchlight.LKmdl.v", vtx, ".RData")
# save(LKmodels, file=nameout)


cat("")
cat("Output saved\n")
