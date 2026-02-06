setup_environment <- function(requirements_file) {
  
  # Stop if no requirements file is provided
  if (missing(requirements_file)) {
    stop("No requirements file provided. Specify the path to R_requirements.txt")
  }
  # heck if requirements file exists
  if (!file.exists(requirements_file)) {
    stop("Requirements file not found: ", requirements_file)
  }
  # Check if renv is installed, if not install it
  if (!"renv" %in% rownames(installed.packages())) {
    cat("\nInstalling renv in the base library to handle dependencies.\n")
    install.packages("renv", repos = "https://cloud.r-project.org/")
  }

  # Check if renv.lock exists, if not initiate it
  if (!file.exists("renv.lock")) {
    cat("\nInitializing renv for Lattice Kriging analyses.\n")
    renv::init(project = PROJECT_DIR, bare = TRUE)
  }
  renv::load(project = PROJECT_DIR)
  proj_lib <- renv::paths$library(project = PROJECT_DIR)
  .libPaths(unique(c(proj_lib, SYS_LIBPATHS)))
  cat("\nLibraries available in the renv environment:\n")
  cat(.libPaths(), sep = "\n")

  # Read required packages from the requirements file and install any that are missing
  required_packages <- scan(requirements_file, what = character(), sep = "\n", quiet = TRUE)
  required_packages <- required_packages[nzchar(required_packages)]
  # missing_pkgs <- setdiff(required_packages, rownames(installed.packages()))
  installed_in_proj <- rownames(installed.packages(lib.loc = proj_lib))
  missing_pkgs <- setdiff(required_packages, installed_in_proj)
  if (length(missing_pkgs) > 0) {
    cat("\nThe following missing packages will be installed:\n\t",
        paste(missing_pkgs, collapse = "\n\t"), "\n", sep = "")
    renv::install(missing_pkgs, prompt = FALSE)
  }

  # Hard guarantee
  installed_in_proj <- rownames(installed.packages(lib.loc = proj_lib))
  still_missing <- setdiff(required_packages, installed_in_proj)
  if (length(still_missing) > 0) {
    stop("Packages still missing after install:\n\t", paste(still_missing, collapse = "\n\t"))
  }

  renv::snapshot(project = PROJECT_DIR, prompt=FALSE)
  cat("\nSetup complete\n")
}
