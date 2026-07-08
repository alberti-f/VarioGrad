#' Fit Lattice Kriging Model
#'
#' Fits a Lattice Kriging model to the provided locations and gradients.
#' Returns the fitted model object.
#'
#' @param locations Matrix of spatial coordinates (n x 3)
#' @param values Vector of spatial data (n)
#' @param LKparams List of LK model parameters (from JSON)
#' @param weights Optional vector of weights (n)
#' @param covariates Optional matrix of covariates (n x p)
#' @return Fitted LK model object
#' 
fit_LK_model <- function(locations, values, LKparams, weights = NULL, covariates = NULL) {
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
    choleskyMemory = list(nnzR= 2E7),
    weights = weights,
    normalize = LKparams$normalize
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

  return(LKfit)


}

#' Fit Covariates-Only Model
#'
#' Fits a linear, fixed effects model using only covariates to predict the provided values.
#' Returns the fitted model object.
#' 
#' @param values Vector of spatial data (n)
#' @param covariates Matrix of covariates (n x p)
#' @return Fitted covariates-only model object
#' 
fit_covariates_only_model <- function(values, covariates) {
  if (is.null(covariates)) {
    stop("Covariates are required for covariates-only model.")
  }
  if (is.null(dim(values))) {
    values <- matrix(values, ncol = 1)
  }
  if (nrow(covariates) != nrow(values)) {
    stop("Covariates and values have different numbers of rows.")
  }

  covariates.center <- colMeans(covariates)
  covariates.scale <- apply(covariates, 2, sd)
  covariates.scale[covariates.scale == 0] <- 1
  covariates.z <- scale(covariates, center = covariates.center, scale = covariates.scale)
  covariates.z <- cbind(1, covariates.z)

  fit <- lm.fit(x = covariates.z, y = values)

  COVfit <- list(
    coefficients = fit$coefficients,
    fitted.values = fit$fitted.values,
    y = values,
    covariates.center = covariates.center,
    covariates.scale = covariates.scale
  )
  return(COVfit)
}
