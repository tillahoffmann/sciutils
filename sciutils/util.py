from collections import abc


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
