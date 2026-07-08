#' Parse Command-Line Arguments for LK Searchlight Analysis
#'
#' Returns a named list of arguments:
#' H, g, sl.idx, scale, side, train_values_path, test_values_path,
#' train_covariates_path, test_covariates_path
#'
parse_optional_file_args <- function(args) {
  opts <- list(
    train_values_path = NULL,
    test_values_path = NULL,
    train_covariates_path = NULL,
    test_covariates_path = NULL
  )

  if (length(args) == 0) {
    return(opts)
  }

  for (i in seq.int(1, length(args), by=3)) {
    flag <- args[[i]]
    if (!(flag %in% c("-v", "--values", "-c", "--covariates"))) {
      stop(paste("Error: Unknown optional argument:", flag))
    }

    train_path <- as.character(args[[i + 1]])
    test_path <- as.character(args[[i + 2]])

    if (flag %in% c("-v", "--values")) {
      opts$train_values_path <- train_path
      opts$test_values_path <- test_path
    } else {
      opts$train_covariates_path <- train_path
      opts$test_covariates_path <- test_path
    }
  }

  opts
}

parse_args <- function(args = NULL) {
  if (is.null(args)) args <- commandArgs(trailingOnly = TRUE)

  if (length(args) < 5 || ((length(args) - 5) %% 3 != 0)) {
    stop(paste(
      "Error: Wrong number of arguments provided.",
      "\nUsage: Rscript LK.searchlight.R <H> <g> <sl.idx> <scale> <side> [-v train_values test_values] [-c train_covariates test_covariates]",
      "\nWhere:",
      "  <H>        : Hemisphere ('L' or 'R')",
      "  <g>        : path to CSV with data to predict",
      "  <sl.idx>   : Searchlight index (integer)",
      "  <scale>    : Scale parameter (integer)",
      "  <side>     : Searchlight side length (integer)",
      "  -v, --values     : Optional train/test CSV files with shape (subject, vertex)",
      "  -c, --covariates : Optional train/test NumPy arrays with shape (subject, vertex[, covariate])",
      sep = "\n"
    ))
  }

  optional_args <- parse_optional_file_args(args[-(1:5)])

  c(list(
    H = as.character(args[[1]]),
    g = as.character(args[[2]]),
    sl.idx = as.integer(args[[3]]),
    scale = as.integer(args[[4]]),
    side = as.integer(args[[5]])
  ), optional_args)
}

#' Parse LK-related Parameters from JSON File
#'
#' Reads LK model parameters from a JSON file and returns them as a named list.
#'
#' @param json_path Path to the JSON file containing LK parameters.
#' @return Named list of LK parameters.
#'
read_LK_params <- function(json_path) {
  if (!file.exists(json_path)) {
    stop(paste("Error: JSON file not found:", json_path))
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("The 'jsonlite' package is required to read JSON files. Please install it.")
  }

  # Load JSON parameters
  params <- jsonlite::fromJSON(json_path)

  # Validate required fields
  required_fields <- c("nlevel", "NC", "NC.buffer", "overlap", "a.wght", "alpha", "LKGeometry", "max.points", "mean.neighbor", "verbose")
  missing_fields <- setdiff(required_fields, names(params))
  if (length(missing_fields) > 0) {
    stop(paste("Missing required LK parameter(s) in JSON:", paste(missing_fields, collapse=", ")))
  }
  return(params)
}
