from collections import abc
import numpy as np


def dict_list_transpose(xs):
    """
    Transpose a dictionary of arrays to an array of dictionaries and vice versa.

    Parameters
    ----------
    xs : mapping or iterable
        Dictionary of arrays or array of dictionaries to convert.

    Returns
    -------
    y : iterable or mapping
        Transposed dictionary of arrays or array of dictionaries to convert.
    """
    if isinstance(xs, abc.Mapping):
        y = None
        for key, values in xs.items():
            if y is None:
                y = [{} for _ in range(len(values))]
            for i, value in enumerate(values):
                y[i][key] = value
    else:
        y = {}
        for x in xs:
            for key, value in x.items():
                y.setdefault(key, []).append(value)
    return y


def bincountnd(xs, weights=None, minshape=None):
    """
    Count number of occurrences of each value in an array of non-negative
    integer tuples.

    Parameters
    ----------
    xs : array_like, 2 dimensions, non-negative integers
        Input array with shape `(p, n)`, where `p` is the number of
        dimensions of the output array and `n` is the number of indices.
    weights : array_like, 1 dimension, optional
        Weight vector of length `n` corresponding to indices in `xs`.
    minshape : tuple, optional
        A minimum shape for the output array.

    Returns
    -------
    out : ndarray
        The result of binning the input indices.
    """
    # Evaluate the shape of the output array.
    if minshape is None:
        minshape = 0
    shape = [np.max(x) + 1 for x in xs]
    shape = np.maximum(shape, minshape)
    minlength = np.prod(shape)
    # Evaluate the compressed index (i.e. the indices in a ravelled array).
    compressed = 0
    a = minlength
    for dim, x in zip(shape, xs):
        a //= dim
        compressed = compressed + a * x

    # Evaluate the bincount.
    if weights is not None:
        weights = weights.ravel()
    bincount = np.bincount(compressed.ravel(), weights, minlength)
    return bincount.reshape(shape)
