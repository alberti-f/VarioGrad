#' Parse Command-Line Arguments for LK Searchlight Analysis
#'
#' Returns a named list of arguments: H, g, sl.idx, scale, side
#'
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 5) {
    stop(paste(
      "Error: Not enough arguments provided.",
      "\nUsage: Rscript LK.searchlight.R <H> <g> <sl.idx> <scale> <side>",
      "\nWhere:",
      "  <H>        : Hemisphere ('L' or 'R')",
      "  <g>        : Gradient index (integer)",
      "  <sl.idx>   : Searchlight index (integer)",
      "  <scale>    : Scale parameter (integer)",
      "  <side>     : Searchlight side length (integer)",
      sep = "\n"
    ))
  }
  list(
    H = as.character(args[[1]]),
    g = as.integer(args[[2]]),
    sl.idx = as.integer(args[[3]]),
    scale = as.integer(args[[4]]),
    side = as.integer(args[[5]])
  )
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
