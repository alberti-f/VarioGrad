#' Evaluate Lattice Kriging Model
#'
#' Evaluates a fitted Lattice Kriging model on training or test data.
#' Performs prediction and computes evaluation metrics:
#' - R-squared (R2)
#' - Correlation between observed and predicted values
#' - Mean Absolute Error (MAE)
#' - Root Mean Squared Error (RMSE)
#' @param LKfit Fitted LK model object
#' @param locations_new Optional matrix of new spatial coordinates (n x 3) for evaluation
#' @param values_new Optional vector of new spatial data (n) for evaluation
#' @param covariates_new Optional matrix of new covariates (n x p)
#' @return List of lists containing observed and predicted values, and evaluation metrics
evaluate_LK_model <- function(LKfit, locations_new = NULL, values_new = NULL, covariates_new = NULL) {
  if (!is.null(locations_new)) {
    y.hat <- predict.LKrig(LKfit, xnew = locations_new, Znew = covariates_new)
    y <- values_new
  } else {
    y.hat <- LKfit$fitted.values[,1]
    y <- LKfit$y
  }

  R2 <- r2score(y, y.hat)
  cor_val <- cor(y, y.hat)[1, 1]
  mae <- mean(abs(y - y.hat))
  rmse <- sqrt(mean((y - y.hat)^2))

  return(list(
    predicted = y.hat,
    observed = y,
    scores = list(
      R2 = R2,
      correlation = cor_val,
      mae = mae,
      rmse = rmse
    )
  ))
}


#' Evaluate Covariates-Only Model
#'
#' Evaluates a fitted covariates-only model on training or test data.
#' Performs prediction and computes evaluation metrics:
#' - R-squared (R2)
#' - Correlation between observed and predicted values
#' - Mean Absolute Error (MAE)
#' - Root Mean Squared Error (RMSE)
#'
#' @param fit Fitted covariates-only model object
#' @param values_new Optional vector of new spatial data (n) for evaluation
#' @param covariates_new Optional matrix of new covariates (n x p)
#' @return List of lists containing observed and predicted values, and evaluation metrics
evaluate_covariates_only_model <- function(fit, values_new = NULL, covariates_new = NULL) {
  if (is.null(covariates_new)) {
    y.hat <- fit$fitted.values
    y <- fit$y
  } else {
    covariates_new.z <- scale(covariates_new, center = fit$covariates.center, scale = fit$covariates.scale)
    covariates_new.z <- cbind(1, covariates_new.z)
    y.hat <- covariates_new.z %*% fit$coefficients
    y <- values_new
  }

  R2 <- r2score(y, y.hat)
  cor_val <- cor(y, y.hat)[1, 1]
  mae <- mean(abs(y - y.hat))
  rmse <- sqrt(mean((y - y.hat)^2))

  return(list(
    predicted = y.hat,
    observed = y,
    scores = list(
      R2 = R2,
      correlation = cor_val,
      mae = mae,
      rmse = rmse
    )
  ))
}

#' Compute R-squared
#'
#' Computes the R-squared value given true and predicted values.
#'
#' @param y_true Vector of true values
#' @param y_pred Vector of predicted values
#' @return R-squared value
#' 
r2score <- function(y_true, y_pred) {
  ss.total <- sum((y_true - mean(y_true))^2)
  ss.resid <- sum((y_true - y_pred)^2)
  r2 = 1 - (ss.resid / ss.total)
  return(r2)
}
