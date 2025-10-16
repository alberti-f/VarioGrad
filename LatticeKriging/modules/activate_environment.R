#' Setup R Environment for Lattice Kriging Analysis
#'
#' Activates renv and loads required libraries. If renv.lock is missing, sources setup.R to create the environment.
#'
#' 

check_requirements <- function(pkgs) {
  missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    cat("Installing missing packages:", paste(missing, collapse=", "), "\n")
    install.packages(missing, repos = "https://cloud.r-project.org/")
  }
}

activate_environment <- function() {

  # Check if renv.lock exists
  if (!file.exists("renv.lock")) {
    cat("Running script setup.R to create the environment.\n")
    args <- commandArgs(trailingOnly = FALSE)
    setup_path <- dirname(sub("--file=", "", args[grep("--file=", args)]))
    source(paste0(setup_path, "/tmp.modules/setup_environment.R"))
  }

  # Check and install required libraries if missing
  check_requirements(c("caret", "reticulate", "LatticeKrig", "jsonlite"))

  # Activate renv and load required libraries
  renv::activate()
  library(caret)
  library(reticulate)
  library(LatticeKrig)
  library(jsonlite)

  # Import Python modules globally for use in other functions
  assign("np", reticulate::import("numpy"), envir = .GlobalEnv)
  assign("vgu", reticulate::import("variograd_utils"), envir = .GlobalEnv)
}
