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
    cat(paste("Starting LK covariate analysis script", as.character(Sys.time())))

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
    if (is.null(args$train_covariates_path) || is.null(args$test_covariates_path)) {
        stop("Covariate analysis requires -c <train_covariates> <test_covariates>.")
    }

    cat(paste("Set up data objects", as.character(Sys.time())))
    # Generate the train and test data objects
    train.data <- load_data(args$H, args$g, args$scale, args$side, group = "train",
                            values_path = args$train_values_path, covariates_path = args$train_covariates_path)
    test.data <- load_data(args$H, args$g, args$scale, args$side, group = "test",
                           values_path = args$test_values_path, covariates_path = args$test_covariates_path)

    # Get the searchlight ID from the train data
    unique.IDs <- sort(unique(as.vector(train.data$vtx.sl)))
    sl.ID <- unique.IDs[args$sl.idx]

    outpath <- paste0(train.data$odir, "/LKresults_cov", ".S", args$scale, ".G", args$g)
    outpath <- paste0(outpath, "/side", args$side, "/", args$H)
    if (!dir.exists(outpath)) {dir.create(outpath, recursive=TRUE)}
    output <- paste0(outpath, "/LKresults_cov_hdf5_G", args$g, "_sl", sl.ID, ".h5")
    # if (file.exists(output)) {
    #     cat(paste("SL#", args$sl.idx, "    Results already exist at:", output))
    #     quit(save = "no")
    # }


    # Select vertices in the searchlight domain
    train.sl <- select_searchlight(train.data, sl.ID)
    test.sl <- select_searchlight(test.data, sl.ID)
    train.sl$weights <- NULL


    # Center the training and test data
    train.sl.c <- select_searchlight_centered(train.sl)
    test.sl.c <- select_searchlight_centered(test.sl)

    # Fit and evaluate the basic LK model
    cat(paste("SL", sl.ID, "#", args$sl.idx, "    Fitting and evaluating basic LK models", as.character(Sys.time())))
    LKfit <- fit_LK_model(train.sl$locations, train.sl$values, LKparams, weights = train.sl$weights, covariates = train.sl$covariates)
    LKeval.train <- evaluate_LK_model(LKfit)
    LKeval.test <- evaluate_LK_model(LKfit, locations_new = test.sl$locations, values_new = test.sl$values, covariates_new = test.sl$covariates)

    # Fit and evaluate values using covariates only
    cat(paste("SL", sl.ID, "#", args$sl.idx, "    Fitting and evaluating covariates-only model", as.character(Sys.time())))
    cat("Train covariates shape:", dim(train.sl$covariates), "\n")
    covfit <- fit_covariates_only_model(train.sl$values, train.sl$covariates)
    coveval.train <- evaluate_covariates_only_model(covfit)
    coveval.test <- evaluate_covariates_only_model(covfit, values_new = test.sl$values, covariates_new = test.sl$covariates)


    # Fit and evaluate centered model
    cat(paste("SL", sl.ID, "#", args$sl.idx, "    Fitting and evaluating centered LK models", as.character(Sys.time())))
    LKfit.c <- fit_LK_model(train.sl.c$locations, train.sl.c$values, LKparams, weights = train.sl.c$weights, covariates = train.sl.c$covariates)
    LKeval.train.c <- evaluate_LK_model(LKfit.c)
    LKeval.test.c <- evaluate_LK_model(LKfit.c, locations_new = test.sl.c$locations, values_new = test.sl.c$values, covariates_new = test.sl.c$covariates)

    # Fit and evaluate centered values using covariates only
    cat(paste("SL", sl.ID, "#", args$sl.idx, "    Fitting and evaluating centered covariates-only model", as.character(Sys.time())))
    covfit.c <- fit_covariates_only_model(train.sl.c$values, train.sl.c$covariates)
    coveval.train.c <- evaluate_covariates_only_model(covfit.c)
    coveval.test.c <- evaluate_covariates_only_model(covfit.c, values_new = test.sl.c$values, covariates_new = test.sl.c$covariates)



    # Save results to HDF5 after all analyses
    cat(paste("\nSL", sl.ID, "#", args$sl.idx, "    Saving results", as.character(Sys.time())))

    save_results_cov(
        train_sl = train.sl,
        test_sl = test.sl,
        basic_train = LKeval.train,
        basic_test = LKeval.test,
        centered_train = LKeval.train.c,
        centered_test = LKeval.test.c,
        covariates_only_train = coveval.train,
        covariates_only_test = coveval.test,
        covariates_only_centered_train = coveval.train.c,
        covariates_only_centered_test = coveval.test.c,
        filename = output
    )

    # Save parameters to CSV
    output <- paste0(outpath, "/parameters.csv")
    save_parameters_csv(LKparams, args, output)
    cat(paste("SL", sl.ID, "#", args$sl.idx, "\nResults saved at:", outpath))

    cat(paste("\nSL", sl.ID, "#", args$sl.idx, "    LK covariate analysis script completed", as.character(Sys.time())))
}

if (identical(environment(), globalenv())) {
    main()
}
