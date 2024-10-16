import numpy as np
from variograd_utils.embed_utils import kernelize
from scipy.spatial.distance import pdist, squareform


def exponential_covariance(dists, correlation_length):
    """Exponential covariance function.
    
    Parameters
    ----------
    dists : array_like
        The pairwise distances between points.
    correlation_length : float
        The correlation length of the covariance function.
    
    Returns
    -------
    covariance_matrix : array_like
        The covariance matrix.
    """
    return np.exp(-dists / correlation_length)


def spherical_covariance(dists, correlation_length):
    """Spherical covariance function.
    
    Parameters
    ----------
    dists : array_like
        The pairwise distances between points.
    correlation_length : float
        The correlation length of the covariance function.
    
    Returns
    -------
    covariance_matrix : array_like
        The covariance matrix.
    """
    cov = np.zeros_like(dists)
    for i in range(len(dists)):
        for j in range(len(dists)):
            if dists[i, j] < correlation_length:
                cov[i, j] = (1.5 * (dists[i, j] / correlation_length)
                             - 0.5 * (dists[i, j] / correlation_length)**3)
            else:
                cov[i, j] = 1.0  # Sill value
    return cov


def gaussian_covariance(dists, correlation_length):
    """Gaussian covariance function.
    
    Parameters
    ----------
    dists : array_like
        The pairwise distances between points.
    correlation_length : float
        The correlation length of the covariance function.
    
    Returns
    -------
    covariance_matrix : array_like
        The covariance matrix.
    """
    return np.exp(- (dists / correlation_length) ** 2)


def generate_covariance_matrix(dists, model, correlation_length):
    """Generate a covariance matrix based on the specified model.
    
    Parameters
    ----------
    dists : array_like
        The pairwise distances between points.
    model : str
        The covariance model to use ('exponential', 'spherical', 'gaussian', 'matern').
    correlation_length : float
        The correlation length of the covariance function.
    nu : float, optional
        The smoothness parameter for the Matérn covariance function.
    
    Returns
    -------
    covariance_matrix : array_like
        The covariance matrix.
    """
    # Select the covariance function
    if model == 'exponential':
        covar_function = exponential_covariance
    elif model == 'spherical':
        covar_function = spherical_covariance
    elif model == 'gaussian':
        covar_function = gaussian_covariance
    else:
        raise ValueError("Unknown covariance model specified.")

    return covar_function(dists, correlation_length)


def generate_spatial_data(num_points, correlation_length, model="spherical",
                          cov_noise=0., abs_noise=0.):
    """Generate spatial data with controllable autocorrelation.
    
    Parameters
    ----------
    num_points : int
        The number of points to generate.
    correlation_length : float
        The correlation length of the spatial data.
    cov_noise : float
        The noise level to add to the covariance matrix.
        Indicates the fraction of covariance standard deviation to add.
    abs_noise : float
        The noise level to add to the data.
        The maximum absolute size of the noise to add.
        Used to generate a normal distribution of noise between [-abs_noise, abs_noise].
    model : str
        The covariance model to use ('exponential', 'spherical', 'gaussian', 'matern').
    
    Returns
    -------
    coordinates : array_like
        The spatial coordinates of the points.
    data : array_like
        The generated spatial data.

    Notes
    -----
    The spatial data is generated using the exponential covariance function.
    The covariance matrix is calculated using the pairwise distances between points.
    The coordinates are generated randomly in a 10x10 grid.
    The data spans the range [0, 1].
    """
    # Generate random coordinates
    x = np.random.uniform(0, 10, num_points)
    y = np.random.uniform(0, 10, num_points)
    coordinates = np.vstack((x, y)).T

    # Calculate pairwise distances
    dists = squareform(pdist(coordinates))

    # Create covariance matrix using the exponential kernel
    covariance_matrix = generate_covariance_matrix(dists, model, correlation_length)
    cov_noise = ((1 - np.random.uniform(0, 1, covariance_matrix.shape))
                 * cov_noise * covariance_matrix.std())
    cov_noise = (cov_noise + cov_noise.T) / 2
    covariance_matrix += cov_noise

    # Generate correlated random data
    mean = np.zeros(num_points)
    data = np.random.multivariate_normal(mean, covariance_matrix + cov_noise * np.eye(num_points))
    data += (1 - np.random.normal(0, 1, num_points)) * abs_noise

    return coordinates, data


def bins_ol(xmin, xmax, nbins=10, overlap=0.25, inclusive=True):
    """
    Define equally spaced, overlapping bins.
    
    Parameters
    ----------
    xmin, xmax: float or scalar
        the extremes of the span of values to bin
    nbins : scalar
        number of bins to divide the values in
    overlap : float
        the fraction of overlap between bins. Must be between -0.5 and 0.5 (Default=0.25)
        Negative values will result in disjoint bins.
    inclusive : bool
        if True, the bounds of the bins will be the centers of the outer bins.
        If False, the bounds will be the edges of the outer bins.
    
    Returns:
    --------
    lower:
        the lower bound of every bin
    upper:
        the upper bound ov every bin

    """
    if overlap < -0.5 or overlap > 0.5:
        raise ValueError("'overlap' should be between -0.5 and 0.5")

    span = xmax - xmin

    if inclusive:
        step = span / nbins
        center = np.arange(xmin, xmax + step, step)
        half_window = step * 0.5  +  step * overlap

        lower = center - half_window
        upper = center + half_window

    else:
        ratio = nbins * (1 - 2 * overlap) + (nbins + 1) * overlap

        window = span / ratio
        step = window * (1 - overlap)

        lower = np.arange(xmin, xmax, step)[:nbins]
        upper = lower + window

    return lower, upper


def digitizedd(x, bins):
    """Digitize data in an ND space.
    
    Parameters
    ----------
    x : array_like
        The data to be digitized. MxN array of M points in N dimensions.
    bins : array_like, tuple, or int
        The bin edges along each dimension.
        A list of N arrays, each with the bin edges for that dimension.
        If a tuple, bins should cspecify the number of bins in each dimension (slower).
        If an int, the same number of bins will be used for each dimension.      
        Bins are defined as bins[i-1] <= x < bins[i].
        A constant of 1e-6 is subtracted7added to the extremes to ensure all
        the bins include all points in the data.
    
    Returns
    -------
    bin_idx : array_like
        The bin index for each point in x. An array of size M.
    """

    # Get the number of dimensions
    ndims = x.shape[1]

    k = 1e-6
    min_ax = x.min(axis=0) - k
    max_ax = x.max(axis=0) + k
    if isinstance(bins, tuple):
        bins = [np.linspace(min_ax[idx], max_ax[idx], bins[idx] + 1) for idx in range(ndims)]
    elif isinstance(bins, int):
        bins = np.linspace(min_ax, max_ax, num=bins+1).T

    # Digitize each dimension
    digitized_indices = np.array([np.digitize(x[:, i], bins[i]) - 1 for i in range(ndims)])

    # Calculate the total number of bins
    nbins = [len(b) - 1 for b in bins]

    # Create a unique index for each combination of bin indices
    return np.ravel_multi_index(digitized_indices, nbins)


def _variogram(distances, differences, lag, weight=None, scale=None):
    """
    Calculate the empirical variogram at a given lag

    Parameters
    ----------
    distances : array_like
        The pairwise distances between points.
    differences : array_like
        The pairwise differences between values.
    lag_edges : tuple
        The edges of the lag bin.
    weight : str
        The weighting function to apply.
        Options are "cauchy", "gauss", "log", "linear"
    scale : float
        The scaling factor for the weighting function.
    
    Returns
    -------
    gamma : float
        The empirical variogram at the given lag.
    """

    N = distances.size

    if weight is None:
        weights = np.ones(N) / N
    else:
        weights = kernelize(np.abs(lag - distances), kernel=weight, scale=scale)
        weights /= weights.sum()

    diffs_sqrd = np.square(differences)
    gamma = np.sum(weights * diffs_sqrd) / 2

    return gamma


def nd_variogram(points, values, lags, overlap=0, min_pairs=10, weight=None, scale=None):
    """
    Calculate the empirical variogram for a set of points and values
    
    Parameters
    ----------
    points : array_like
        The spatial coordinates of the points. NxM array of N points in M dimensions.
    values : array_like
        The values at each point. Nx1 array.
    lags : array_like
        The lags at which to calculate the variogram.
    overlap : float
        The overlap between bins.
    
    Returns
    -------
    variogram : array_like
        The empirical variogram at each lag.
    """

    # Calculate the pairwise distances
    rows, cols = np.triu_indices(points.shape[0], k=1)
    dists = points[rows] - points[cols]
    dists = np.sqrt(np.sum(dists**2, axis=1))
    values = values.squeeze()
    diffs = np.abs(values[rows] - values[cols])
    diffs = np.sqrt(np.sum(diffs**2, axis=1)) if diffs.ndim > 1 else diffs

    # Calculate the variogram
    lag_step = np.diff(lags).mean()
    lag_tolerance = (lag_step * (1 + overlap)) / 2

    variogram = np.full(len(lags), np.nan)
    for lag_i, lag in enumerate(lags):
        mask = np.abs(dists - lag) <= lag_tolerance
        diffs_lag = diffs[mask]
        dists_lag = dists[mask]
        if mask.sum() < min_pairs or np.isnan(diffs_lag).all():
            continue
        else:
            variogram[lag_i] = _variogram(dists_lag, diffs_lag, lag, weight=weight, scale=scale)

    return variogram



# Temporarily not implemented because seemingly slow for small datasets and few lags.
# Might be useful in onther circumstances.
# def nd_variogram_vectorized(points, values, lags, overlap=0, min_pairs=10
#                             , weight=None, scale=None):
#     """
#     Calculate the empirical variogram for a set of points and values

#     Parameters
#     ----------
#     points : array_like
#         The spatial coordinates of the points. NxM array of N points in M dimensions.
#     values : array_like
#         The values at each point. Nx1 array.
#     lags : array_like
#         The lags at which to calculate the variogram.
#     overlap : float
#         The overlap between bins.

#     Returns
#     -------
#     variogram : array_like
#         The empirical variogram at each lag.
#     """

#     # Calculate the pairwise distances
#     rows, cols = np.triu_indices(points.shape[0], k=1)
#     dists = points[rows] - points[cols]
#     dists = np.sqrt(np.sum(dists**2, axis=1))
#     values = values#.squeeze()
#     diffs = np.abs(values[rows] - values[cols])
#     diffs = np.sqrt(np.sum(diffs**2, axis=1)) if diffs.ndim > 1 else diffs

#     # Calculate the variogram
#     lag_step = np.diff(lags).mean()
#     lag_window = lag_step * (1 + overlap)
#     lag_edges =  np.vstack([lags - lag_window / 2, lags + lag_window / 2])
#     lag_masks = np.logical_and(dists >= lag_edges[0].reshape(-1, 1),
#                                dists <= lag_edges[1].reshape(-1, 1)).astype(int)
#     lag_npairs = lag_masks.sum(axis=1).reshape(-1, 1)
#     lag_masks *= (lag_npairs >= min_pairs).astype(int)

#     if weight is None:
#         weights = lag_masks / lag_npairs
#     else:
#         weights = np.abs((dists - lags.reshape(-1, 1)))
#         weights = kernelize(weights, kernel=weight, scale=scale) * lag_masks
#         weights[lag_npairs] /= weights[lag_npairs].sum(axis=1).reshape(-1, 1)

#     diffs_lags = weights * (diffs.reshape(1, -1) ** 2 )
#     variogram = np.sum(diffs_lags, axis=1) / 2
