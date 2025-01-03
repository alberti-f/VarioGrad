#' Lattice Kriging Analysis Setup Script
#'
#' This R script sets up the environment for performing Lattice Kriging analyses.
#'
#' ## Workflow
#' 1. Checks for the availability of the `renv` package and installs it if necessary.
#' 2. Initializes an `renv` environment if it does not already exist.
#' 3. Activates the `renv` environment and prints the current library paths.
#' 4. Reads a list of required packages from an external `R_requirements.txt` file 
#'    located in the same directory as the script.
#' 5. Identifies and installs any missing packages listed in the requirements file.
#' 6. Updates the `renv.lock` file to reflect the current package environment.
#'
#' ## External File Dependency
#' - The `R_requirements.txt` must be placed in the same directory as the script.
#'
#' ## Output
#' - Updates the `renv.lock` file to capture the package state.
#'
#' ## Note
#' - The script assumes that it is run from the command line using `Rscript`.
#'
#' @keywords Lattice Kriging, renv, R Environment Setup, Dependency Management


# Install renv if not present
renv_avail <- "renv" %in% rownames(installed.packages())
if(!"renv" %in% rownames(installed.packages())) {
  cat("\nInstalling renv in the base library to handle dependencies.\n")
  install.packages("renv", repos = "https://cloud.r-project.org/")
}

# Initialize renv if not existing
if(!file.exists("renv.lock")) {
  cat("\nInitializing renv for Lattice Kriging analyses.\n")
  renv::init()
}

# 
renv::activate()
cat("\nLibraries available in the renv environment:\n")
cat(.libPaths())

# Check for required packages
args = commandArgs(trailingOnly=F)
reqs_path <- dirname(sub("--file=", "", args[grep("--file=", args)]))
reqs_path <- paste0(reqs_path, "/R_requirements.txt")
required_packages <-  as.list(scan(reqs_path, sep="\n", what=character()))
new_packages <- required_packages[
  !(required_packages %in% rownames(installed.packages()))
  ]

# Install missing packages
if(length(new_packages)) {
  cat(do.call(paste, c("\nThe following missing packages will be installed:",
                       new_packages, sep = "\n\t")))
  cat("\n")
  renv::install(new_packages)
}

# Update lock file
renv::snapshot()

cat("\nSetup complete\n\n\n")