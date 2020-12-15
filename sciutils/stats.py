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
from scipy import special


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
