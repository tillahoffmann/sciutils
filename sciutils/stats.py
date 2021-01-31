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
from scipy import integrate, special, stats


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
        Probability density function evaluated over a regular grid.
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
    # Sort the probability density and evaluate the normalised cumulative distribution. We aggregate
    # identical pdf values so we can interpolate.
    pdf, weights = np.unique(-pdf, return_counts=True)
    pdf = - pdf
    cum = integrate.cumtrapz(pdf * weights)
    cum = np.concatenate([np.zeros(1), cum])
    cum /= cum[-1]
    # Find the first index that encloses more than the desired mass
    js = np.argmax(cum[:, None] > pvals, axis=0)
    levels = []
    for j, pval in zip(js, pvals):
        i = j - 1
        # Get the upper and lower bounds and interpolate to find the best level.
        y2 = cum[j]
        y1 = cum[i]
        x2 = pdf[j]
        x1 = pdf[i]
        slope = (y2 - y1) / (x2 - x1)
        offset = y1 - slope * x1
        level = (pval - offset) / slope
        levels.append(level)

    return np.asarray(levels)


def evaluate_hpd_mass(pdf):
    """
    Evaluate the highest posterior density mass excluded from isocontours.

    Parameters
    ----------
    pdf : array_like
        Probability density function evaluated over a regular grid.

    Returns
    -------
    excluded : array_like
        The probability mass excluded at a given isocontour of the `pdf`.
    """
    shape = np.shape(pdf)
    pdf = np.ravel(pdf)
    idx = np.argsort(-pdf)
    cum = np.cumsum(pdf[idx])
    cum /= cum[-1]
    return 1 - np.reshape(cum[np.argsort(idx)], shape)


class TransformedVariable:
    def apply(self, x):
        """
        Transform a variable from an unconstrained space to a possibly constrained space.

        Parameters
        ----------
        x : array_like
            Variable to transform.

        Returns
        -------
        y : array_like
            Transformed variable.
        log_jacobian : array_like
            Logarithm of the Jacobian associated with the transform.
        """
        raise NotImplementedError

    def invert(self, y):
        """
        Transform a variable from a possibly constrained space to an untransformed space.

        Parameters
        ----------
        y : array_like
            Transformed variable.

        Returns
        -------
        x : array_like
            Variable after inverse transform.
        """
        raise NotImplementedError

    @staticmethod
    def apply_transforms(transforms, values):
        """
        Apply transforms to a set of values.

        Parameters
        ----------
        transforms : dict
            Transforms by variable name.
        values : dict
            Variables to transform.

        Returns
        -------
        values : dict
            Variables with transforms applied.
        log_jacobian : float
            Logarithm of the Jacobian associated with all transforms.
        """
        log_jacobian = 0
        for key, transform in transforms.items():
            values[key], contrib = transform.apply(values[key])
            log_jacobian += np.sum(contrib)
        return values, log_jacobian

    @staticmethod
    def invert_transforms(transforms, values):
        """
        Apply inverse transforms to a set of values.

        Parameters
        ----------
        transforms : dict
            Transforms by variable name.
        values : dict
            Transformed variables.

        Returns
        -------
        values : dict
            Variables with inverse transforms applied.
        """
        for key, transform in transforms.items():
            values[key] = transform.invert(values[key])
        return values


class SemiBoundedVariable(TransformedVariable):
    r"""
    A semi-bounded variable :math:`y = loc + scale\times\exp(x)` on the interval :math:`[loc, \inf]`
    if :math:`scale > 0` and :math:`[-\inf, loc]` if :math:`scale < 0`.
    """
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
        self._log_scale = np.log(scale)

    def apply(self, x):
        expx = np.exp(x)
        return self.loc + self.scale * expx, - self._log_scale - x

    def invert(self, y):
        return np.log((y - self.loc) / self.scale)


class BoundedVariable(TransformedVariable):
    r"""
    A bounded variable :math:`y = a + \frac{(b - a)}{1 + \exp(-x)}` on the interval :math:`[a, b]`.
    """
    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b
        self.scale = b - a
        self._log_scale = np.log(self.scale)

    def apply(self, x):
        expitx = special.expit(x)
        return self.a + self.scale * expitx, - 2 * np.log(expitx) - self._log_scale + x

    def invert(self, y):
        return special.logit((y - self.a) / self.scale)
