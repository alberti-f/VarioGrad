#' Fit Lattice Kriging Model
#'
#' Fits a Lattice Kriging model to the provided locations and gradients.
#' Returns the fitted model object.
#'
#' @param locations Matrix of spatial coordinates (n x 3)
#' @param values Vector of spatial data (n)
#' @param LKparams List of LK model parameters (from JSON)
#' @param covariates Optional matrix of covariates (n x p)
#' @return Fitted LK model object
fit_LK_model <- function(locations, values, LKparams, covariates = NULL) {
  # Load required library
  if (!requireNamespace("LatticeKrig", quietly = TRUE)) {
    stop("The 'LatticeKrig' package is required. Please install it.")
  }

  # Create LKinfo object
  LKinfo <- LKrigSetup(
    locations,
    nlevel = LKparams$nlevel,
    NC = LKparams$NC,
    NC.buffer = LKparams$NC.buffer,
    overlap = LKparams$overlap,
    a.wght = LKparams$a.wght,
    alpha = c(LKparams$alpha),
    LKGeometry = LKparams$LKGeometry,
    max.points = LKparams$max.points,
    mean.neighbor = LKparams$mean.neighbor,
    verbose = LKparams$verbose,
    choleskyMemory = list(nnzR= 2E7)
  )

  # Fit the model
  # Model without covariates
  LKfit <- LatticeKrig(
    x = locations,
    y = values,
    Z = covariates,
    LKinfo = LKinfo,
    verbose = LKparams$verbose,
    normalize = LKparams$normalize)

  print("Lattice Kriging model fitted.")
  q = 1

  return(LKfit)


}
