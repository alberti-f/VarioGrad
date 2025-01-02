# Set up R environment for Lattice Kriging analyses

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