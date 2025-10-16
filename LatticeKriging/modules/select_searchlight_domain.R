#' Select Searchlight Domain and Create Masks
#'
#' Extracts the relevant subset of data for a given searchlight index.
#' Returns masked locations, gradients, subject indices, and mask.
#'
#' @param data List returned by load_data()
#' @param sl.idx Index of the searchlight to select
#' @return List with masked locations, gradients, subject indices, mask, and searchlight ID
select_searchlight <- function(data, ID) {
  # Check if ID is present in data$vtx.sl
  if (!(ID %in% data$vtx.sl)) {
    stop(paste("Error: ID", ID, "is not in the vertex.sl array"))
  }

  # Create mask for the selected searchlight
  mask <- data$vtx.sl == ID

  # Apply mask to locations and values
  locations <- data$locations[mask, ]
  values <- data$values[mask, ]
  subj.idx <- rep(1:data$n.subj, each = data$n.vtx)[mask]
  vertex.idx <- rep(1:data$n.vtx, times = data$n.subj)[mask]

  # Return masked data and info
  list(
    ID = ID,
    locations = locations,
    values = values,
    subj.idx = subj.idx,
    vertex.idx = vertex.idx,
    mask = mask
  )
}


select_searchlight_centered <- function(searchlight=NULL, data=NULL, ID=NULL) {
  if (is.null(searchlight) && (is.null(data) || is.null(ID))) {
    stop("Error: Provide either 'searchlight', or both 'data' and 'ID'.")
  }
  if (!is.null(searchlight) && (!is.null(data) || !is.null(ID))) {
    stop("Error: Provide only 'searchlight', or both 'data' and 'ID', not all three.")
  }
  if (!is.null(data) && !is.null(ID)) {
    searchlight <- select_searchlight(data, ID)
  }

  subj.avg <- compute_local_stats(searchlight, mean)
  subj.sd <- compute_local_stats(searchlight, sd)

  centered.values <- searchlight$values - subj.avg[searchlight$subj.idx]

  list(
    ID = searchlight$ID,
    locations = searchlight$locations,
    values = centered.values,
    subj.idx = searchlight$subj.idx,
    vertex.idx = searchlight$vertex.idx,
    mask = searchlight$mask,
    subj.avg = subj.avg[searchlight$subj.idx],
    subj.sd = subj.sd[searchlight$subj.idx]
  )
}


compute_local_stats <- function(domain_data, stat_fun) {
  # browser()
  if (!is.list(domain_data) || is.null(domain_data$values) || is.null(domain_data$subj.idx)) {
    stop("domain_data must be a list with 'values' and 'subj.idx'")
  }

  unique.subj.idx <- unique(domain_data$subj.idx)
  subj.stats <- numeric(length = max(unique.subj.idx))
  for (i in unique.subj.idx) {
      n.vtx <- sum(domain_data$subj.idx==i)
      subj.values <- domain_data$values[domain_data$subj.idx==i]

      # if (n.vtx==1) {
      #     subj.stats[i] <- subj.values
      #     next
      # }

      subj.stats[i] <- stat_fun(subj.values)
  }

  return(subj.stats)
}


replace_values <- function(searchlight, new_values, by="vertex") {
  if (is.null(searchlight) || is.null(new_values)) {
    stop("Error: Provide both 'searchlight' and 'new_values'.")
  }

  if (length(new_values) < max(searchlight$subj.idx)) {
    stop(paste0("Error: ", by, " indices in 'searchlight' exceed length of 'new_values'."))
  }

  if (by == "subject") {
    replaced.values <- new_values[searchlight$subj.idx]
  } else if (by == "vertex") {
    replaced.values <- new_values[searchlight$vertex.idx]
  }

  list(
    ID = searchlight$ID,
    locations = searchlight$locations,
    values = replaced.values,
    subj.idx = searchlight$subj.idx,
    vertex.idx = searchlight$vertex.idx,
    mask = searchlight$mask
  )
}

replace_locations <- function(searchlight, new_locations, by="vertex") {
  if (is.null(searchlight) || is.null(new_locations)) {
    stop("Error: Provide both 'searchlight' and 'new_locations'.")
  }

  if (nrow(new_locations) != nrow(searchlight$locations)) {
    stop(paste0("Error: ", by, " indices in 'searchlight' exceed length of 'new_values'."))
  }

  if(by == "vertex") {
    replaced.locations <- new_locations[searchlight$vertex.idx, ]
  } else if (by == "subject") {
    replaced.locations <- new_locations[searchlight$subj.idx, ]
  }

  list(
    ID = searchlight$ID,
    locations = replaced.locations,
    values = searchlight$values,
    subj.idx = searchlight$subj.idx,
    vertex.idx = searchlight$vertex.idx,
    mask = searchlight$mask
  )
}