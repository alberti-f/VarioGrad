#' Evaluate Lattice Kriging Model Performance
#'
#' Computes R^2 and correlation between observed and predicted values.
#' If new data is provided, predicts and evaluates on that data; otherwise, evaluates on training data.
#'
#' @param LKfit Fitted LKrig model object
#' @param locations_new Optional matrix of new spatial coordinates (n x 3)
#' @param values_new Optional vector of new observed values (n)
#' @param covariates_new Optional matrix of new covariates (n x p)
#' @return List with R2, correlation, predictions, and observed values

evaluate_LK_model <- function(LKfit, locations_new = NULL, values_new = NULL, covariates_new = NULL) {
  # If new data is provided, predict on new data
  if (!is.null(locations_new)) {
    y.hat <- predict.LKrig(LKfit, xnew = locations_new, Znew = covariates_new)
    y <- values_new
  } else {
    # Evaluate on training data
    y.hat <- LKfit$fitted.values[,1]
    y <- LKfit$y
  }

  # Compute R^2
  R2 <- r2score(y, y.hat)

  # Compute correlation
  cor_val <- cor(y, y.hat)[1, 1]

  # Compute mean absolute error
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


#' Calculates the coefficient of determination (R² score) for regression models.
#'
#' The R² score, also known as the coefficient of determination, measures the proportion of
#' variance in the dependent variable that is predictable from the independent variables.
#' It is computed as:
#' \deqn{R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
#' where \eqn{y_i} are the true values, \eqn{\hat{y}_i} are the predicted values, and \eqn{\bar{y}}
#' is the mean of the true values.
#'
#' @param y_true Numeric vector of true values.
#' @param y_pred Numeric vector of predicted values.
#' @return Numeric value representing the R² score.

r2score <- function(y_true, y_pred) {
  ss.total <- sum((y_true - mean(y_true))^2)
  ss.resid <- sum((y_true - y_pred)^2)
  r2 = 1 - (ss.resid / ss.total)
  return(r2)
}