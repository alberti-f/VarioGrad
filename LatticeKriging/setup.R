# Set up R environment for Lattice Kriging analyses

# Install renv if not present
renv_avail <- "renv" %in% rownames(installed.packages())
if(!"renv" %in% rownames(installed.packages())) {
  cat("Installing renv in the base library to handle dependencies.\n")
  install.packages("renv", repos = "https://cloud.r-project.org/")
}

# Initialize renv if not existing
if(!file.exists("renv.lock")) {
  cat("Initializing renv for Lattice Kriging analyses.\n")
  renv::init()
}

# 
renv::activate()
cat("Libraries available in the renv environment:\n")
cat(.libPaths())

# Check for required packages
required_packages <-  as.list(scan("R_requirements.txt", sep="\n", what=character()))
new_packages <- required_packages[
  !(required_packages %in% rownames(installed.packages()))
  ]

# Install missing packages
if(length(new_packages)) {
  cat(do.call(paste, c("The following missing packages will be installed:",
                       new_packages, sep = "\n\t")))
  cat("\n")
  renv::install(new_packages)
}

# Update lock file
renv::snapshot()

cat("\nSetup complete\n\n\n")