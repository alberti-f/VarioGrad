test_replace_values_vertex_match <- function(searchlight, group_values) {
  replaced <- replace_values(searchlight, group_values)
  n_unique_vertex <- length(unique(searchlight$vertex.idx))
  n_unique_values <- length(unique(replaced$values))
  if (n_unique_vertex == n_unique_values) {
    message("Success: Number of unique vertex indices matches number of unique values.")
  } else {
    message(sprintf("Mismatch: %d unique vertex indices, %d unique values.", n_unique_vertex, n_unique_values))
  }
}

compute_95pct_parallelepiped_overlap <- function(coords1, coords2) {
  # coords1 and coords2: matrices (n x d) of coordinates
  if (!is.matrix(coords1) || !is.matrix(coords2)) {
    stop("Both inputs must be coordinate matrices.")
  }
  if (ncol(coords1) != ncol(coords2)) {
    stop("Coordinate sets must have the same number of dimensions.")
  }
  d <- ncol(coords1)

  # Compute 2.5% and 97.5% quantiles for each dimension
  get_bounds <- function(mat) {
    lower <- apply(mat, 2, quantile, probs=0.01)
    upper <- apply(mat, 2, quantile, probs=0.99)
    list(lower=lower, upper=upper)
  }
  b1 <- get_bounds(coords1)
  b2 <- get_bounds(coords2)

  # Compute parallelepiped volumes
  vol <- function(bounds) {
    prod(bounds$upper - bounds$lower)
  }
  vol1 <- vol(b1)
  vol2 <- vol(b2)

  # Compute overlap bounds
  overlap_lower <- pmax(b1$lower, b2$lower)
  overlap_upper <- pmin(b1$upper, b2$upper)
  # If any dimension has negative overlap, volume is zero
  if (any(overlap_upper < overlap_lower)) {
    overlap_vol <- 0
  } else {
    overlap_vol <- prod(overlap_upper - overlap_lower)
  }

  # Percentage overlap relative to each set
  pct1 <- 100 * overlap_vol / vol1
  pct2 <- 100 * overlap_vol / vol2

  return(list(
    volume1 = vol1,
    volume2 = vol2,
    overlap_volume = overlap_vol,
    pct_overlap1 = pct1,
    pct_overlap2 = pct2
  ))
}

matching_idx <- function(data, searchlight) {
  if (is.null(data) || is.null(searchlight)) {
    stop("Error: Provide both 'data' and 'searchlight'.")
  }

  n <- nrow(searchlight$locations)

  matching_indices <- logical(n)
  for (i in 1:n) {
    row <- ((searchlight$subj.idx[i]-1) * data$n.vtx) + searchlight$vertex.idx[i]
    matching_indices[i] <- all(data$locations[row,] == searchlight$locations[i, ])
  }
  if (!all(matching_indices)){
    message("Error: Mismatch between data and searchlight locations.")
  } else {
    message("Success: All location coordinates match between data and searchlight.")
  }

  matching_indices <- logical(n)
  for (i in 1:n) {
    row <- ((searchlight$subj.idx[i]-1) * data$n.vtx) + searchlight$vertex.idx[i]
    matching_indices[i] <- all(data$values[row] == searchlight$values[i])
  }
  if (!all(matching_indices)){
    message("Error: Mismatch between data and searchlight values.")
  } else {
    message("Success: All values match between data and searchlight.")
  }

}


path_walk <- function(x, path = "root") {
  out <- list()
  if (is.list(x)) {
    nms <- names(x)
    for (i in seq_along(x)) {
      key <- if (!is.null(nms) && nzchar(nms[i])) nms[i] else paste0("[[", i, "]]")
      out <- c(out, path_walk(x[[i]], paste0(path, "$", key)))
    }
  } else {
    out <- list(list(path = path, type = class(x), length = length(x), is_char = is.character(x)))
  }
  out
}

debug_scan <- function(obj) {
  hits <- Filter(function(e) isTRUE(e$is_char), path_walk(obj))
  if (length(hits)) {
    message("Character fields (potential string vectors):")
    for (h in hits) message(" - ", h$path, " (len=", h$length, ")")
  } else {
    message("No character fields found.")
  }
  invisible(hits)
}