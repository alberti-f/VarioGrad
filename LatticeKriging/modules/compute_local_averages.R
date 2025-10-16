

#' Compute subject-wise and group-wise averages and residuals for a searchlight domain
#'
#' This function takes the output of select_searchlight_domain and computes:
#' - subject-wise mean within the domain (handling single/multiple vertices per subject)
#' - group mean within the domain
#' - subject-wise residuals (gradient minus subject mean)
#'
#' @param domain_data List. Output from select_searchlight_domain, must contain:
#'   - gradients: vector (masked gradients)
#'   - subj.idx: vector (subject index for each gradient)
#' @return List with:
#'   - grad.subj.locavg: subject-wise mean (vector)
#'   - grad.grp.locavg: group mean (scalar)
#'   - grads.res.loc: residuals (vector)
compute_local_averages <- function(domain_data, n.subj) {

  if (!is.list(domain_data) || is.null(domain_data$values) || is.null(domain_data$subj.idx)) {
    stop("domain_data must be a list with 'values' and 'subj.idx'")
  }

  unique.subj.idx <- unique(domain_data$subj.idx)
  subj.avg <- rep(NA, n.subj)
  for (i in unique.subj.idx) {
      n.vtx <- sum(domain_data$subj.idx==i)
      subj.values <- domain_data$values[domain_data$subj.idx==i]

      if (n.vtx==1) {
          subj.avg[i] <- subj.values
          next
      }

      subj.avg[i] <- mean(subj.values)
  }

  return(subj.avg)
}
