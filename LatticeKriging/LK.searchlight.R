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
#'    - Rea#ds# input arguments and sets model parameters (e.g., number of vertices, scale).
#' 3. **Data Preparation**:
#'    - Loa#ds#, partitions, and processes gradient and embedding data.
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
#' @keywor#ds# Lattice Kriging, Spatial Modeling, Gradient Analysis


################################################################################
################################################################################

if(!file.exists("renv.lock")) {
  cat("Running script setup.R to create the environment.\n")
  args = commandArgs(trailingOnly=F)
  setup_path <- dirname(sub("--file=", "", args[grep("--file=", args)]))
  source(paste0(setup_path, "/setup.R"))
}

# Setup environment
renv::activate()
library(caret)
library(reticulate)
library(LatticeKrig)

################################################################################

# Set up parameters

# Input arguments

# H = "R"
# g = 1
# sl.idx = 826
# scale = 50
# side=4

args = commandArgs(trailingOnly=TRUE)
H = as.character(args[[1]])
g = as.integer(args[[2]])
sl.idx = as.integer(args[[3]])
scale = as.integer(args[[4]])
side = as.integer(args[[5]])

# Embeddings-related parameters
algorithm <- paste0("JE_cauchy", scale)
diffusion <- "a05_t0"
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
NC.buffer = 2
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
train_ds <- vgu$dataset("train")
test_ds <- vgu$dataset("test")



train_odir <- train_ds$output_dir
test_odir <- test_ds$output_dir
train_subj_list <- as.matrix(train_ds$subj_list)
test_subj_list <- as.matrix(test_ds$subj_list)
n.train <- train_ds$N
n.test <- test_ds$N

# Load and set up data

# Load data from train and test sets
embeddings.train <- np$load(paste0(train_odir, "/All.",H,".embeddings.npz"))[[algorithm]]
embeddings.train <- embeddings.train[,,1:3]
locations.avg <- colMeans(embeddings.train)
locations.train <- matrix(aperm(embeddings.train, c(2, 1, 3)),
                          nrow = n.train*nvtx, ncol = 3)
embeddings.test <- np$load(paste0(test_odir, "/All.",H,".embeddings.npz"))[[algorithm]]
embeddings.test <- embeddings.test[,,1:3]
locations.test <- matrix(aperm(embeddings.test, c(2, 1, 3)),
                         nrow = n.test*nvtx, ncol = 3)


# Load gradients from the train/test dataset
filename <- train_ds$outpath(paste0(train_ds$id, ".", H, ".FC_embeddings.", diffusion, ".G", g, ".csv"))
gradients.train <- as.matrix(read.csv(filename, header=FALSE))
gradients.avg <- apply(gradients.train, 2, median)
grad.subj.avg.train <- apply(gradients.train, 1, median)
grad.grp.avg.train <- rep(gradients.avg, n.train)
gradients.train  <- matrix(t(gradients.train), ncol=1)

filename <- test_ds$outpath(paste0(test_ds$id, ".", H, ".FC_embeddings.", diffusion, ".G", g, ".csv"))
gradients.test <- as.matrix(read.csv(filename, header=FALSE))
gradients.test <- matrix(t(gradients.test), ncol=1)
grad.subj.avg.test <- apply(gradients.test, 1, median)
grad.grp.avg.test <- rep(gradients.avg, n.test)

################################################################################

# Select data points within searchlight

# Load searchlight IDs
fname <- train_ds$outpath(paste0(
    "train.", H, ".SL_IDs.", algorithm, "_l", side, ".npy"
))
sl.train <- np$load(fname)
sl.train <- matrix(t(sl.train), ncol=1)

fname <- test_ds$outpath(paste0(
    "test.", H, ".SL_IDs.", algorithm, "_l", side, ".npy"
))
sl.test <- np$load(fname)
sl.test <- matrix(t(sl.test), ncol=1)

sl.IDs <- unique(as.vector(sl.train))

# Select searchlight
sl.id <- sl.IDs[sl.idx]

# Create masks of vertices included in this domain
mask.train <- sl.train == sl.id
mask.test <- sl.test == sl.id

# Apply masks
locations.train <- locations.train[mask.train,]
gradients.train <- gradients.train[mask.train]
subj.idx.train <- rep(1:n.train, each=nvtx)[mask.train]
expand.row.train <- match(subj.idx.train, sort(unique(subj.idx.train)))
train.idx.sl <- unique(subj.idx.train)
n.train.sl <- length(train.idx.sl)

locations.test <- locations.test[mask.test,]
gradients.test <- gradients.test[mask.test]
subj.idx.test <- rep(1:n.test, each=nvtx)[mask.test]
expand.row.test <- match(subj.idx.test, sort(unique(subj.idx.test)))
test.idx.sl <- unique(subj.idx.test)
n.test.sl <- length(test.idx.sl)

cat(" ")
cat(" ")
cat(paste0("Processing searchlight N ",  sl.id, "\n"))
cat(" ")
cat(" ")

################################################################################

# Compute subj-wise mean of gradient within the domain

# in the train set
subj.nvtx.train <- numeric()
grad.subj.locavg.train <- numeric()
for (i in 1:n.train.sl) {
    subj.idx = train.idx.sl[i]
    subj.nvtx.train[i] <- sum(subj.idx.train==subj.idx)
    if (subj.nvtx.train[i]>1) {        
        grad.subj.locavg.train[i] <- mean(gradients.train[subj.idx.train==subj.idx])
    }
    else if (subj.nvtx.train[i]==1) {
        grad.subj.locavg.train[i] <- gradients.train[subj.idx.train==subj.idx]
    }
}


# and in the test set
grad.subj.locavg.test <- numeric()
for (i in 1:n.test.sl) {
    subj.idx = test.idx.sl[i]
    subj.nvtx.tmp <- sum(subj.idx.test==subj.idx)
    if (subj.nvtx.tmp>1) {        
        grad.subj.locavg.test[i] <- mean(gradients.test[subj.idx.test==subj.idx])
    }
    else if (subj.nvtx.tmp==1) {
        grad.subj.locavg.test[i] <- gradients.test[subj.idx.test==subj.idx]
    }
}


# Compute group averages for vertex index
grad.grp.avg.train <- grad.grp.avg.train[mask.train]
grad.grp.avg.test <- grad.grp.avg.test[mask.test]

################################################################################
################################################################################

# Calculate residuals 

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
                       grads.train.res.loc,
                       LKinfo = LKinfo,
                       verbose = verbose,
                       normalize = TRUE)

cat(" ")
cat("Model fitting complete\n\n")
################################################################################
################################################################################

# Generate predictions

# Predict train locations
predict.train.locavg.full <- grad.subj.locavg.train[expand.row.train] + LKfit.locavg$fitted.values
predict.train.locavg.fixed <- grad.subj.locavg.train[expand.row.train] + predict.LKrig(LKfit.locavg, xnew = locations.train, just.fixed=TRUE)
predict.train.locavg.avg <- grad.subj.locavg.train[expand.row.train]

# Predict test locations
predict.test <- predict.LKrig(LKfit, xnew = locations.test)

predict.test.locavg.full <- grad.subj.locavg.test[expand.row.test] + predict.LKrig(LKfit.locavg, xnew = locations.test)
predict.test.locavg.fixed <- grad.subj.locavg.test[expand.row.test] + predict.LKrig(LKfit.locavg, xnew = locations.test, just.fixed=TRUE)
predict.test.locavg.avg <- grad.subj.locavg.test[expand.row.test]

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

# On the test set
r2.test <- rsquare(gradients.test,  predict.test)
r2.test.gavg <- rsquare(gradients.test, grad.grp.avg.test)
r2.test.savg <- rsquare(gradients.test, grad.subj.locavg.test[expand.row.test])

r2.test.locavg.full <- rsquare(gradients.test, predict.test.locavg.full)
r2.test.locavg.fixed <- rsquare(gradients.test, predict.test.locavg.fixed)
r2.test.locavg.avg <- rsquare(gradients.test, predict.test.locavg.avg)

cat("")
cat("Computed r square\n\n")
################################################################################

# Calculate R

# On the train set
cor.train <- cor(gradients.train, LKfit$fitted.values)
cor.train.gavg <- cor(gradients.train, grad.grp.avg.train)
cor.train.savg <- cor(gradients.train, predict.train.locavg.avg)

cor.train.locavg.full <- cor(gradients.train, predict.train.locavg.full)
cor.train.locavg.fixed <- cor(gradients.train, predict.train.locavg.fixed)
cor.train.locavg.avg <- cor(gradients.train, predict.train.locavg.avg)

# On the test set
cor.test <- cor(gradients.test,  predict.test)
cor.test.gavg <- cor(gradients.test, grad.grp.avg.test)
cor.test.savg <- cor(gradients.test, predict.test.locavg.avg)

cor.test.locavg.full <- cor(gradients.test, predict.test.locavg.full)
cor.test.locavg.fixed <- cor(gradients.test, predict.test.locavg.fixed)
cor.test.locavg.avg <- cor(gradients.test, predict.test.locavg.avg)

cat(" ")
cat("Computed correlation\n\n")

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
outpath <- paste0(train_odir, "/LKresults.", algorithm, ".G", g)
if (!dir.exists(outpath)) {dir.create(outpath)}

outpath <- paste0(outpath, "/side", side, "/", H)
if (!dir.exists(outpath)) {
    dir.create(outpath, recursive=TRUE)
    dir.create(paste0(outpath, "/Cors"))
    dir.create(paste0(outpath, "/R2"))
    dir.create(paste0(outpath, "/MLEs"))
    dir.create(paste0(outpath, "/LKmodels"))
    dir.create(paste0(outpath, "/NVertex"))
    dir.create(paste0(outpath, "/Coeffs"))
    dir.create(paste0(outpath, "/Predictions"))
}


params <- data.frame(
  searchlight = sl.id,
  scale = scale,
  algorithm = algorithm,
  diffusion = diffusion,
  g = g,
  nvtx = nvtx,
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
                       LKfit.locavg = LKfit.locavg$MLE$summary))
nameout <- paste0(outpath, "/MLEs/LK.searchlight.MLE.v", sl.id, ".csv")
write.csv(MLE.df, nameout)


R2.df <- data.frame(r2.train=r2.train,
                    r2.train.gavg=r2.train.gavg,
                    r2.train.savg=r2.train.savg,
                    r2.train.locavg.full=r2.train.locavg.full,
                    r2.train.locavg.fixed=r2.train.locavg.fixed,
                    r2.train.locavg.avg=r2.train.locavg.avg,
                    r2.test=r2.test,
                    r2.test.gavg=r2.test.gavg,
                    r2.test.savg=r2.test.savg,
                    r2.test.locavg.full=r2.test.locavg.full,
                    r2.test.locavg.fixed=r2.test.locavg.fixed,
                    r2.test.locavg.avg=r2.test.locavg.avg)
nameout <- paste0(outpath, "/R2/LK.searchlight.R2.v", sl.id, ".csv")
write.csv(R2.df, nameout, row.names=FALSE)


cor.df <- data.frame(cor.train=cor.train,
                     cor.train.gavg=cor.train.gavg,
                     cor.train.savg=cor.train.savg,
                     cor.train.locavg.full=cor.train.locavg.full,
                     cor.train.locavg.fixed=cor.train.locavg.fixed,
                     cor.train.locavg.avg=cor.train.locavg.avg,
                     cor.test=cor.test,
                     cor.test.gavg=cor.test.gavg,
                     cor.test.savg=cor.test.savg,
                     cor.test.locavg.full=cor.test.locavg.full,
                     cor.test.locavg.fixed=cor.test.locavg.fixed,
                     cor.test.locavg.avg=cor.test.locavg.avg)
nameout <- paste0(outpath, "/Cors/LK.searchlight.cor.v", sl.id, ".csv")
write.csv(cor.df, nameout, row.names = FALSE)


predict.train.df <- data.frame(subject=subj.idx.train,
                               vertex=rep(1:nvtx, n.train)[mask.train],
                               true.train=gradients.train,
                               predict.train=LKfit$fitted.values,
                               predict.train.locavg.full=predict.train.locavg.full,
                               predict.train.locavg.fixed=predict.train.locavg.fixed,
                               predict.train.locavg.avg=predict.train.locavg.avg)
nameout <- paste0(outpath, "/Predictions/LK.searchlight.train.v", sl.id, ".csv")
write.csv(predict.train.df, nameout, row.names = FALSE)


predict.test.df <- data.frame(subject=subj.idx.test,
                              vertex=rep(1:nvtx, n.test)[mask.test],
                              true.test=gradients.test,
                              predict.test=predict.test,
                              predict.test.locavg.full=predict.test.locavg.full,
                              predict.test.locavg.fixed=predict.test.locavg.fixed,
                              predict.test.locavg.avg=predict.test.locavg.avg)
nameout <- paste0(outpath, "/Predictions/LK.searchlight.test.v", sl.id, ".csv")
write.csv(predict.test.df, nameout, row.names = FALSE)

cat(" ")
cat("Output saved\n\n")
