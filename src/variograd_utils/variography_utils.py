import numpy as np
from variograd_utils.embed_utils import kernelize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import sklearn.metrics.pairwise as pw
from sklearn.metrics import r2_score


class Variogram:
    """
    Class for calculating empirical variograms and fitting variogram models.
    
    Parameters
    ----------
    None

    Attributes
    ----------
    lags : array_like
        The lags at which to calculate the variogram.
    lag_pairs : array_like
        The number of pairs at each lag.
    exp_variogram : array_like
        The empirical variogram at each lag.
    the_variogram : array_like
        The fitted variogram at each lag.   
    variogram_model : dict
        The parameters of the fitted variogram model.
    
    Methods
    -------
    omndir_variogram(points, values, lags, overlap=0, min_pairs=0, weight=None, scale=None)
        Calculate the empirical omnidirectional variogram for a set of points and values.
    directional_variogram()
        Calculate the empirical directional variogram for a set of points and values.
    fit_variogram_model(model="spherical", curve_fit_kwargs={})
        Fit a variogram model to the empirical variogram.
    """

    def __init__(self):
        """
        Initialize the Variogram object.
        """

        self.lags = None
        self.lag_pairs = None
        self.exp_variogram = None
        self.the_variogram = None
        self.variogram_model = None

    def omndir_variogram(self, points, values, lags,
                         overlap=0, min_pairs=0, weight=None, scale=None,
                         metric="euclidean"):
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
        metric : str
            metric used to compute distances in space and the difference between values.
            Options are 'euclidean', 'manhattan', and 'cosine'.
        
        Returns
        -------
        variogram : array_like
            The empirical variogram at each lag.
        """

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        if values.ndim == 1:
            values = values.reshape(-1, 1)

        dist_fun = getattr(pw, f"{metric}_distances")

        # Calculate the pairwise distances
        dists = dist_fun(points)[np.triu_indices(points.shape[0], k=1)]
        diffs = dist_fun(values)[np.triu_indices(values.shape[0], k=1)]

        # Calculate the variogram
        if not np.allclose(np.diff(lags), lags[1] - lags[0], atol=1e-16):
            raise ValueError("Lags must be placed at regular intervals.")

        # Define lag cheracteristics
        lag_step = lags[1] - lags[0]
        lag_tolerance = (lag_step * (1 + overlap)) / 2

        # Compute experimental variogram
        self.lags = lags
        self.lag_pairs = np.full(len(lags), np.nan)
        self.exp_variogram = np.full(len(lags), np.nan)
        for lag_i, lag in enumerate(lags):
            mask = _lag_mask(dists, lag, lag_tolerance=lag_tolerance)
            diffs_lag = diffs[mask]
            dists_lag = dists[mask]
            self.lag_pairs[lag_i] = mask.sum()

            if self.lag_pairs[lag_i] < min_pairs or np.isnan(diffs_lag).all():
                continue

            else:
                self.exp_variogram[lag_i] = _single_lag_variogram(dists_lag, diffs_lag, lag,
                                                                  weight=weight, scale=scale)


    def directional_variogram(self):
        """
        Will calculate the empirical directional variogram for a set of points and values.
        """
        raise NotImplementedError("Directional variograms are not implemented yet.")


    def fit_variogram_model(self, model="spherical", curve_fit_kwargs={}):
        '''
        Fit a variogram model to the empirical variogram

        Parameters
        ----------
        model : str, callable
            The variogram model to fit.
            Predefined variogram functions are 'spherical', 'exponential', 'gaussian'.
            It is alsp possible to pass a custom function with arguments:
            - lags: array_like
                The lags at which to calculate the variogram.
            - nugget: float
                The nugget effect.
            - contribution: float
                The contribution of the variogram.
            - range: float
                The range of the variogram.
        curve_fit_kwargs : dict
            Additional keyword arguments to pass to `scipy.optimize.curve_fit`.
        
        Returns
        -------
        self: Variogram object        
        '''
    
        variogram_models = {
            "spherical": spherical,
            "exponential": exponential,
            "gaussian": gaussian,
            "custom": model
        }

        if callable(model):
            model = "custom"
        elif isinstance(model, str) & (model not in variogram_models):
            raise ValueError("`model` must be 'spherical', 'exponential', 'gaussian',"
                             + "or a callable custom function")

        lags = self.lags[~np.isnan(self.exp_variogram)]
        exp_variogram = self.exp_variogram[~np.isnan(self.exp_variogram)]
        (ngt, cont, rng), _= curve_fit(variogram_models[model], lags,
                                       exp_variogram, **curve_fit_kwargs)

        self.the_variogram = variogram_models[model](lags, ngt, cont, rng)
        self.variogram_model = {"model": model,
                                "function": variogram_models[model],
                                "nugget": ngt,
                                "contribution": cont,
                                "range": rng,
                                "sill": ngt + cont,
                                "r2": r2_score(exp_variogram, self.the_variogram)}


def _single_lag_variogram(distances, differences, lag, weight=None, scale=None):
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


def _lag_mask(dists, lag, lag_tolerance=0):
    """
    Helper to create a mask for the points within a given lag

    Parameters:
    -----------
    dists : array_like
        The pairwise distances between points.
    lag : float
        The lag to consider.
    lag_tolerance : float
        The tolerance for the lag.
    
    Returns:
    --------
    mask : array_like
        A boolean array indexing the points within the given lag.
    """
    return np.abs(dists - lag) <= lag_tolerance


def spherical(x, nugget, contribution, rng):
    '''
    Spherical variogram model
    '''
    gamma = nugget + contribution * (1.5 * (x / rng) - 0.5 * (x / rng) ** 3)
    gamma[x > rng] = nugget + contribution
    return gamma


def exponential(x, nugget, contribution, rng):
    '''
    Exponential variogram model
    '''
    gamma = nugget + contribution * (1 - np.exp(-3 * x / rng))
    return gamma


def gaussian(x, nugget, contribution, rng):
    '''
    Gaussian variogram model
    '''
    gamma = nugget + contribution * (1 - np.exp(-3 * (x / rng) ** 2))
    return gamma


def logistic(x, L, k, x0):
    '''
    Logistic variogram model
    '''
    return L / (1 + np.exp(-k * (x - x0)))


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


def generate_spatial_data(points, correlation_length, model="spherical",
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
    if isinstance(points, (int, np.int32, np.int64)):
        num_points = points
        x = np.random.uniform(0, 10, num_points)
        y = np.random.uniform(0, 10, num_points)
        coordinates = np.vstack((x, y)).T

    elif isinstance(points, np.ndarray) & (points.shape[1] == 2):
        num_points = points.shape[0]
        coordinates = points

    else :
        raise ValueError("`points` must be an integer or an Nx2 numpy array")

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

    if isinstance(points, (int, np.int32, np.int64)):
        return coordinates, data
    else:
        return data
