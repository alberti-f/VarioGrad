SYS_LIBPATHS <- .libPaths()
Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = "FALSE")
options(repos = c(CRAN = "https://cloud.r-project.org"))

get_script_path <- function() {
  cmd <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd, value = TRUE)

  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg)))
  }

  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile))
  }

  stop("Cannot determine script path.")
}

get_script_dir <- function() {
  script_path <- get_script_path()
  return(dirname(script_path))
}

SCRIPT_DIR <- get_script_dir()
PROJECT_DIR <- SCRIPT_DIR 

main <- function(args = NULL) {
    # Main function to run the LK analysis for a single searchlight
    print(paste("Starting LK analysis script", as.character(Sys.time())))

    # Set up the environment and load modules
    module_dir <- file.path(SCRIPT_DIR, "modules")
    source(file.path(module_dir, "setup_environment.R"))
    source(file.path(module_dir, "activate_environment.R"))
    source(file.path(module_dir, "setup_args.R")) 
    activate_environment(module_dir, file.path(module_dir, "R_requirements.txt"))
    module_files <- list.files(module_dir, pattern = "\\.R$", full.names = TRUE)
    invisible(sapply(module_files, source))

    # Read arguments and LK parameters
    LKparams <- read_LK_params(
        file.path(SCRIPT_DIR, "LK_parameters.json")
    )
    args <- parse_args(args)

    print(paste("SL#", args$sl.idx, "    Set up data objects", as.character(Sys.time())))
    # Generate the train and test data objects
    train.data <- load_data(args$H, args$g, args$scale, args$side, group = "train")
    test.data <- load_data(args$H, args$g, args$scale, args$side, group = "test")


    # Get the searchlight ID from the train data
    unique.IDs <- sort(unique(as.vector(train.data$vtx.sl)))
    sl.ID <- unique.IDs[args$sl.idx]

    outpath <- paste0(train.data$odir, "/LKresults", ".S", args$scale, ".G", args$g)
    outpath <- paste0(outpath, "/side", args$side, "/", args$H)
    if (!dir.exists(outpath)) {dir.create(outpath, recursive=TRUE)}
    output <- paste0(outpath, "/LKresults_hdf5_G", args$g, "_sl", sl.ID, ".h5")
    # if (file.exists(output)) {
    #     print(paste("SL#", args$sl.idx, "    Results already exist at:", output))
    #     quit(save = "no")
    # }

    # Select vertices in the searchlight domain
    train.sl <- select_searchlight(train.data, sl.ID)
    test.sl <- select_searchlight(test.data, sl.ID)

    # Center the training and test data
    train.sl.c <- select_searchlight_centered(train.sl)
    test.sl.c <- select_searchlight_centered(test.sl)

    locations.test.avg <- test.data$locations.avg[test.sl$vertex.idx,]
    values.test.avg <- test.data$values.avg[test.sl$vertex.idx]


    print(paste("SL#", args$sl.idx, "    Fitting and evaluating basic LK models", as.character(Sys.time())))
    # Fit the basic LK model
    LKfit <- fit_LK_model(train.sl$locations, train.sl$values, LKparams)
    LKeval.train <- evaluate_LK_model(LKfit)
    LKeval.test <- evaluate_LK_model(LKfit, locations_new = test.sl$locations, values_new = test.sl$values)
    LKeval.avg <- evaluate_LK_model(LKfit, locations_new = locations.test.avg, values_new = test.sl$values)

    print(paste("SL#", args$sl.idx, "    Running permutation test", as.character(Sys.time())))
    set.seed(123)  # for reproducibility
    n.permutations <- 100
    null.data <- test.data
    LKeval.null <- LKeval.test
    LKeval.null$predicted <- matrix(NA, nrow=length(test.sl$vertex.idx), ncol=n.permutations)
    for (name in names(LKeval.null$scores)) {LKeval.null$scores[[name]] <- rep(NA, n.permutations)}
    for (i in 1:n.permutations) {
        perm_idx <- null.data$n.vtx * (rep(sample(null.data$n.subj), each = null.data$n.vtx) - 1)
        perm_idx <- perm_idx + rep(1:null.data$n.vtx, times = null.data$n.subj)
        null.data$locations <- null.data$locations[perm_idx, , drop=FALSE]
        null.data$subj.list <- null.data$subj.list[perm_idx]
        null.data$vtx.sl <- null.data$vtx.sl[perm_idx]
        null.sl <- select_searchlight(null.data, sl.ID)
        LKeval.tmp <- evaluate_LK_model(LKfit, locations_new = null.sl$locations, values_new = null.sl$values)
        LKeval.null$predicted[,i] <- LKeval.tmp$predicted
        for (name in names(LKeval.null$scores)) {LKeval.null$scores[[name]][i] <- LKeval.tmp$scores[[name]]}
    }
    print(paste("SL#", args$sl.idx, "    Permutation test completed", as.character(Sys.time())))


    print(paste("SL#", args$sl.idx, "    Fitting and evaluating centered LK models", as.character(Sys.time())))
    # Fit and evaluate centered model
    LKfit.c <- fit_LK_model(train.sl.c$locations, train.sl.c$values, LKparams)
    LKeval.train.c <- evaluate_LK_model(LKfit.c)
    LKeval.test.c <- evaluate_LK_model(LKfit.c, locations_new = test.sl.c$locations, values_new = test.sl.c$values)
    LKeval.avg.c <- evaluate_LK_model(LKfit.c, locations_new = locations.test.avg, values_new = test.sl.c$values)

    print(paste("SL#", args$sl.idx, "    Running permutation test", as.character(Sys.time())))
    null.data <- test.data
    LKeval.null.c <- LKeval.test.c
    LKeval.null.c$predicted <- matrix(NA, nrow=length(test.sl$vertex.idx), ncol=n.permutations)
    for (name in names(LKeval.null.c$scores)) {LKeval.null.c$scores[[name]] <- rep(NA, n.permutations)}
    for (i in 1:n.permutations) {
        perm_idx <- null.data$n.vtx * (rep(sample(null.data$n.subj), each = null.data$n.vtx) - 1)
        perm_idx <- perm_idx + rep(1:null.data$n.vtx, times = null.data$n.subj)
        null.data$locations <- null.data$locations[perm_idx, , drop=FALSE]
        null.data$subj.list <- null.data$subj.list[perm_idx]
        null.data$vtx.sl <- null.data$vtx.sl[perm_idx]
        null.sl <- select_searchlight_centered(data=null.data, ID=sl.ID)
        LKeval.tmp <- evaluate_LK_model(LKfit.c, locations_new = null.sl$locations, values_new = null.sl$values)
        LKeval.null.c$predicted[,i] <- LKeval.tmp$predicted
        for (name in names(LKeval.null.c$scores)) {LKeval.null.c$scores[[name]][i] <- LKeval.tmp$scores[[name]]}
    }
    print(paste("SL#", args$sl.idx, "    Permutation test completed", as.character(Sys.time())))



    print(paste("SL#", args$sl.idx, "    Saving results", as.character(Sys.time())))
    # Save results to HDF5 after all analyses


    save_results(
        train_sl = train.sl,
        test_sl = test.sl,
        basic_train = LKeval.train,
        basic_test = LKeval.test,
        basic_avg = LKeval.avg,
        basic_null = LKeval.null,
        centered_train = LKeval.train.c,
        centered_test = LKeval.test.c,
        centered_avg = LKeval.avg.c,
        centered_null = LKeval.null.c,
        filename = output
    )

    # Save parameters to CSV
    output <- paste0(outpath, "/parameters.csv")
    save_parameters_csv(LKparams, args, output)
    print(paste("SL#", args$sl.idx, "    Results saved at:", outpath))

    print(paste("SL#", args$sl.idx, "    LK analysis script completed", as.character(Sys.time())))
}

if (identical(environment(), globalenv())) {
  main()
}