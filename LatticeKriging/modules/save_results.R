#' Save LK Searchlight Results to HDF5
#'
#' This module provides functions to structure and save Lattice Kriging results
#' in a hierarchical HDF5 file using reticulate and variograd_utils.core_utils.save_hdf5.
#'
#' Each section (parameters, searchlight, models) is built as a list,
#' then combined into a single nested list and saved.
#'
#' Usage:
#'   - Call build_parameters_dict(...), build_searchlight_dict(...), build_models_dict(...)
#'   - Call assemble_LK_results_hdf5(...) to combine and save
#'
#' Dependencies: reticulate, variograd_utils

library(reticulate)
core_utils <- import("variograd_utils.core_utils")


# #' High-level function to save all LK results in one call
# save_results <- function(train_sl, test_sl,
#                         basic_train, basic_test, basic_avg, basic_null,
#                         centered_train, centered_test, centered_avg, centered_null,
#                         filename) {
#   # Extract searchlight info from train_sl (assume train_sl and test_sl have same ID)
#   sl_id <- train_sl$ID
#   vertex_indices <- train_sl$vertex.idx
#   subject_indices <- train_sl$subj.idx
#   searchlight <- build_searchlight_dict(sl_id, vertex_indices, subject_indices)
#   models <- build_models_dict(basic_train, basic_test, basic_avg, basic_null,
#                              centered_train, centered_test, centered_avg, centered_null)

#   results <- assemble_LK_results_hdf5(searchlight, models, filename)
#   core_utils$save_hdf5(results, filename)
# }


#' High-level function to save all LK results in one call
save_results <- function(train_sl, test_sl,
                        basic_train, basic_test, basic_avg, basic_null,
                        centered_train, centered_test, centered_avg, centered_null,
                        filename) {
  results <- build_models_dict(train_sl, test_sl,
                             basic_train, basic_test, basic_avg, basic_null,
                             centered_train, centered_test, centered_avg, centered_null)

  core_utils$save_hdf5(results, filename)
}


save_parameters_csv <- function(params, args, outpath) {
  timestamp <- as.character(Sys.time())
  parameters <- c(as.list(params), as.list(args), list(timestamp = timestamp))
  # Ensure all elements are length 1 and character
  parameters_chr <- lapply(parameters, function(x) {
    if (length(x) == 0 || is.null(x)) return(NA_character_)
    as.character(x)
  })
  df <- as.data.frame(parameters_chr, stringsAsFactors = FALSE)
  # Make a single row with named columns
  df <- as.data.frame(lapply(df, function(x) x[1]), stringsAsFactors = FALSE)
  write.csv(df, outpath, row.names = FALSE)
}


# Build searchlight section
build_searchlight_dict <- function(sl_id, vertex_indices, subject_indices) {
  list(
    id = sl_id,
    vertex_indices = vertex_indices,
    subject_indices = subject_indices
  )
}


# Build model scores sublist
build_scores_dict <- function(scores) {
  list(
    R2 = scores$R2,
    correlation = scores$correlation,
    mae = scores$mae,
    rmse = scores$rmse
  )
}


# Build model evaluation section
build_model_eval_dict <- function(eval_obj) {
  list(
    predicted = eval_obj$predicted,
    observed = eval_obj$observed,
    scores = build_scores_dict(eval_obj$scores)
  )
}


# # Build models section
# build_models_dict <- function(basic_train, basic_test, basic_avg, basic_null,
#                              centered_train, centered_test, centered_avg, centered_null) {
#   list(
#     basic = list(
#       train = build_model_eval_dict(basic_train),
#       test = build_model_eval_dict(basic_test),
#       avg = build_model_eval_dict(basic_avg),
#       null = build_model_eval_dict(basic_null)

#     ),
#     centered = list(
#       train = build_model_eval_dict(centered_train),
#       test = build_model_eval_dict(centered_test),
#       avg = build_model_eval_dict(centered_avg),
#       null = build_model_eval_dict(centered_null)
#     )
#   )
# }


# Build models section
build_models_dict <- function(train_sl, test_sl, basic_train, basic_test, basic_avg, basic_null,
                             centered_train, centered_test, centered_avg, centered_null) {
  list(
    searchlight_id = train_sl$ID,
    train = list(
      vertex_indices = train_sl$vertex.idx,
      subject_indices = train_sl$subj.idx,
      basic = build_model_eval_dict(basic_train),
      centered = build_model_eval_dict(centered_train)

    ),
    test = list(
      vertex_indices = test_sl$vertex.idx,
      subject_indices = test_sl$subj.idx,
      basic = build_model_eval_dict(basic_test),
      centered = build_model_eval_dict(centered_test),
      basic_avg = build_model_eval_dict(basic_avg),
      centered_avg = build_model_eval_dict(centered_avg),
      basic_null = build_model_eval_dict(basic_null),
      centered_null = build_model_eval_dict(centered_null)
    )
  )
}


# Main function to assemble and save all results
assemble_LK_results_hdf5 <- function(searchlight, models, filename) {
  list(
    searchlight = searchlight,
    models = models
  )
}
