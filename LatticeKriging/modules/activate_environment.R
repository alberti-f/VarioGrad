#' Setup R Environment for Lattice Kriging Analysis
#'
#' Activates renv and loads required libraries. If renv.lock is missing, sources setup.R to create the environment.
#'
#' 


activate_environment <- function(module_dir, requirements_file) {

  # Stop if no requirements file is provided
  if (missing(requirements_file)) {
    stop("Error: No requirements file provided. Please specify the path to R_requirements.txt")
  }

  # Check if renv.lock exists
  # if (!file.exists("renv.lock")) {
  cat("Running script setup.R to create the environment.\n")
  source(file.path(module_dir, "setup_environment.R"))
  setup_environment(requirements_file)
  # }

  # Check and install required libraries if missing
  # check_requirements(c("caret", "reticulate", "LatticeKrig", "jsonlite"))

  # Activate renv and load required libraries
  library(caret)
  library(reticulate)
  library(LatticeKrig)
  library(jsonlite)

  # Import Python modules globally for use in other functions
  assign("np", reticulate::import("numpy"), envir = .GlobalEnv)
  assign("vgu", reticulate::import("variograd_utils"), envir = .GlobalEnv)
}
