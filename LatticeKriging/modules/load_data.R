#' Load Covariates for LK Searchlight Analysis
#'
#' Loads a NumPy array with shape (subject, vertex) or
#' (subject, vertex, covariate) and flattens it to the LK observation order:
#' (subject * vertex) x covariate.
#'
load_covariates <- function(covariates_path, n.subj, n.vtx) {
  if (is.null(covariates_path)) {
    return(NULL)
  }

  if (!file.exists(covariates_path)) {
    stop(paste("Error: Covariates file not found:", covariates_path))
  }

  covariates <- np$load(covariates_path)
  covariates.dims <- dim(covariates)

  if (!(length(covariates.dims) %in% c(2, 3))) {
    stop(paste(
      "Error: Covariates must have shape (subject, vertex) or (subject, vertex, covariate).",
      "Found dimensions:",
      paste(covariates.dims, collapse = " x ")
    ))
  }

  expected.dims <- c(n.subj, n.vtx)
  if (!all(covariates.dims[1:2] == expected.dims)) {
    stop(paste(
      "Error: Covariates first two dimensions must match n.subj and n.vtx.",
      "Expected:",
      paste(expected.dims, collapse = " x "),
      "Found:",
      paste(covariates.dims[1:2], collapse = " x ")
    ))
  }

  if (length(covariates.dims) == 2) {
    covariates.dims <- c(covariates.dims, 1)
  }

  n.covariates <- covariates.dims[3]
  covariates <- array(as.numeric(covariates), dim = covariates.dims)
  covariates <- matrix(
    aperm(covariates, c(2, 1, 3)),
    nrow = n.subj * n.vtx,
    ncol = n.covariates
  )
  colnames(covariates) <- paste0("covariate_", seq_len(n.covariates))


  covariates
}

#' Load Data for LK Searchlight Analysis
#'
#' Loads train/test datasets, embeddings, gradients, searchlight IDs, and
#' optional covariates. Returns a list containing all loaded data.
#'
load_data <- function(H, g, scale, side, group = "train", values_path = NULL, covariates_path = NULL) {

  # Check that np and vgu are loaded
  if (!exists("np", envir = .GlobalEnv) || !exists("vgu", envir = .GlobalEnv)) {
      stop("Error: Python modules 'numpy' (np) and 'variograd_utils' (vgu) must be loaded in the global environment. Please run setup_environment() first.")
  }

  algorithm <- paste0("JE_cauchy", scale)
  diffusion <- "a05_t0"

  ds <- vgu$dataset(group)
  odir <- ds$output_dir
  subj.list <- as.matrix(ds$subj_list)
  n.subj <- ds$N

  # Load embeddings
  fname <- paste0(odir, "/All.", H, ".embeddings.npz")
  embeddings <- np$load(fname)[[algorithm]]
  embeddings <- embeddings[,,1:3]
  locations.avg <- colMeans(embeddings)
  n.vtx <- nrow(locations.avg)
  locations <- matrix(aperm(embeddings, c(2, 1, 3)), nrow = n.subj*n.vtx, ncol = 3)

  # Load values
  fname <- values_path
  if (is.null(fname)) {
    fname <- ds$outpath(paste0(ds$id, ".", H, ".FC_embeddings.", diffusion, ".G", g, ".csv"))
  }
  values <- as.matrix(read.csv(fname, header=FALSE))
  values.avg <- colMeans(values)
  values <- matrix(t(values), nrow = n.subj*n.vtx, ncol=1)

  # Load searchlight IDs
  fname <- ds$outpath(paste0(group, ".", H, ".SL_IDs.", algorithm, "_l", side, ".npy"))
  vtx.sl <- np$load(fname)
  vtx.sl <- matrix(t(vtx.sl), ncol=1)

  # Load covariates
  covariates <- load_covariates(covariates_path, n.subj, n.vtx)

  list(
    id = ds$id,
    n.subj = n.subj,
    n.vtx = n.vtx,
    subj.list = subj.list,
    odir = odir,
    locations.avg = locations.avg, # check if it can be removed
    locations = locations,
    values.avg = values.avg,
    values = values,
    covariates = covariates,
    vtx.sl = vtx.sl
  )
}
