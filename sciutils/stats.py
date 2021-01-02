"""
Fast, numerically stable implementations of log PDFs and CDFs as well as statistical utility
functions.

.. note::

   Distributions are only guaranteed to be correct within their support. E.g. the behaviour of
   evaluating a Gamma distribution for negative values is undefined.
"""
import hashlib
import logging
import numpy as np
import os
import pickle
from scipy import special, stats


LOGGER = logging.getLogger(__name__)


_LOG2 = np.log(2)
_LOGPI = np.log(np.pi)
_NORMAL_CONST = - (_LOG2 + _LOGPI) / 2
_SQRT2 = np.sqrt(2)


def normal_logpdf(x, mu, sigma):
    """
    Evaluate the log PDF of the normal distribution.
    """
    z = (x - mu) / sigma
    return _NORMAL_CONST - z * z / 2 - np.log(sigma)


def normal_logcdf(x, mu, sigma):
    """
    Evaluate the log CDF of the normal distribution.
    """
    z = (x - mu) / sigma
    return np.log1p(special.erf(z / _SQRT2)) - _LOG2


def cauchy_logpdf(x, mu, sigma):
    """
    Evaluate the log PDF of the Cauchy distribution.
    """
    z = (x - mu) / sigma
    return -_LOGPI - np.log(sigma) - np.log1p(z * z)


def cauchy_logcdf(x, mu, sigma):
    """
    Evaluate the log CDF of the Cauchy distribution.
    """
    z = (x - mu) / sigma
    return np.log1p(2 * np.arctan(z) / np.pi) - _LOG2


def halfcauchy_logpdf(x, mu, sigma):
    """
    Evaluate the log PDF of the half-Cauchy distribution.
    """
    return cauchy_logpdf(x, mu, sigma) + _LOG2


def halfcauchy_logcdf(x, mu, sigma):
    """
    Evaluate the log CDF of the half-Cauchy distribution.
    """
    z = (x - mu) / sigma
    return np.log(2 * np.arctan(z) / np.pi)


def maybe_build_model(model_code, root='.pystan', **kwargs):
    """
    Build a pystan model or retrieve a cached version.

    Parameters
    ----------
    model_code : str
        Stan model code to build.
    root : str
        Root directory at which to cache models.
    **kwargs : dict
        Additional arguments passed to the `pystan.StanModel` constructor.

    Returns
    -------
    model : pystan.StanModel
        Compiled stan model.
    """
    # Construct a filename
    identifier = hashlib.sha1(model_code.encode()).hexdigest()
    filename = os.path.join(root, identifier + '.pkl')

    try:
        with open(filename, 'rb') as fp:
            model = pickle.load(fp)
        LOGGER.info('loaded model from %s', filename)
    except FileNotFoundError:
        import pystan
        model = pystan.StanModel(model_code=model_code, **kwargs)
        os.makedirs(root, exist_ok=True)
        with open(filename, 'wb') as fp:
            pickle.dump(model, fp)
        # Also dump the stan code for reference
        with open(filename.replace('.pkl', '.stan'), 'w') as fp:
            fp.write(model_code)
        LOGGER.info('dumped model to %s', filename)
    return model


def evaluate_mode(x, lin=200, **kwargs):
    """
    Evaluate the mode of a univariate distribution based on samples using a kernel density estimate.

    Parameters
    ----------
    x : array_like
        Univariate samples from the distribution.
    lin : array_like or int
        Sample points at which to evaluate the density estimate or the number of sample points
        across the range of the data.
    **kwargs : dict
        Additional arguments passed to the :class:`scipy.stats.gaussian_kde` constructor.

    Returns
    -------
    mode : float

    """
    kde = stats.gaussian_kde(x, **kwargs)
    if isinstance(lin, int):
        lin = np.linspace(np.min(x), np.max(x), lin)
    y = kde(lin)
    return lin[np.argmax(y)]


def evaluate_hpd_levels(pdf, pvals):
    """
    Evaluate the levels that include a given fraction of the the probability mass.

    Parameters
    ----------
    pdf : array_like
        Probability density function evaluated over a mesh.
    pvals : array_like or int
        Probability mass to be included within the corresponding level or the number of levels.

    Returns
    -------
    levels : array_like
        Contour levels of the probability density function that enclose the desired probability
        mass.
    """
    # Obtain equidistant levels if only the number is given
    if isinstance(pvals, int):
        pvals = (pvals - np.arange(pvals)) / (pvals + 1)
    pvals = np.atleast_1d(pvals)
    # Sort the probability density and evaluate the normalised cumulative distribution
    idx = np.argsort(-pdf.ravel())
    cum = np.cumsum(pdf.ravel()[idx])
    cum /= cum[-1]
    # Find the indices corresponding to the levels
    j = np.argmax(cum[:, None] > pvals, axis=0)
    # Evaluate the levels
    return pdf.ravel()[idx][j]
