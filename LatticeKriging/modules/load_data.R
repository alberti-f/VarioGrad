#' Load Data for LK Searchlight Analysis
#'
#' Loads train/test datasets, embeddings, gradients, and searchlight IDs.
#' Returns a list containing all loaded data.
#'
load_data <- function(H, g, scale, side, group = "train") {

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

  # Load gradients
  fname <- ds$outpath(paste0(ds$id, ".", H, ".FC_embeddings.", diffusion, ".G", g, ".csv"))
  values <- as.matrix(read.csv(fname, header=FALSE))
  values.avg <- colMeans(values)
  values <- matrix(t(values), nrow = n.subj*n.vtx, ncol=1)

  # Load searchlight IDs
  fname <- ds$outpath(paste0(group, ".", H, ".SL_IDs.", algorithm, "_l", side, ".npy"))
  vtx.sl <- np$load(fname)
  vtx.sl <- matrix(t(vtx.sl), ncol=1)

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
    vtx.sl = vtx.sl
  )
}
