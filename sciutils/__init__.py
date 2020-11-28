import numpy as np


class ParameterReshaper:
    """
    Reshape an array of parameters to a dictionary of named parameters and vice versa.

    Parameters
    ----------
    parameters : dict[str, tuple]
        Mapping from parameter names to shapes.

    Notes
    -----
    The trailing dimensions are considered batch dimensions and are not modified. Inputs are not
    validated to improve performance.

    Examples
    --------
    >>> reshaper = su.ParameterReshaper({'a': 2, 'b': (2, 3)})
    >>> reshaper.to_dict(np.arange(reshaper.size))  # doctest: +NORMALIZE_WHITESPACE
    {'a': array([0, 1]), 'b': array([[2, 3, 4], [5, 6, 7]])}
    """
    def __init__(self, parameters):
        self.parameters = {}
        for key, shape in parameters.items():
            if shape is None:
                shape = ()
            elif isinstance(shape, int):
                shape = (shape,)
            self.parameters[key] = shape

        self.sizes = {key: np.prod(shape, dtype=int) for key, shape in self.parameters.items()}
        self.size = sum(self.sizes.values())

    def to_array(self, values):
        """
        Convert a dictionary of values to a vector.

        Parameters
        ----------
        x : dict[str, np.ndarray]
            Mapping from parameter names to values.

        Returns
        -------
        y : np.ndarray
            Array of parameters encoding the named parameters.
        """
        vector = None
        offset = 0
        for key, shape in self.parameters.items():
            value = values[key]
            if vector is None:
                batch_shape = np.shape(value)[len(shape):]
                vector = np.empty((self.size,) + batch_shape)
            # Just set the scalar element for a 0-rank tensor
            if not shape:
                vector[offset] = value
                offset += 1
                continue
            # Flatten any tensor with rank two or larger
            if len(shape) > 1:
                value = np.reshape(value, (-1,) + batch_shape)
            # Set the vector of parameters
            vector[offset:offset + value.shape[0]] = value
            offset += value.shape[0]
        return vector

    def to_dict(self, vector):
        """
        Convert a vector to a dictionary of values.

        Parameters
        ----------
        x : np.ndarray
            Array of parameters encoding a parameter set.

        Returns
        -------
        y : dict[str, np.ndarray]
            Mapping from parameter names to values.
        """
        values = {}
        offset = 0
        batch_shape = np.shape(vector)[1:]
        for key, shape in self.parameters.items():
            # Handle the scalar case
            if not shape:
                values[key] = vector[offset]
                offset += 1
                continue
            size = self.sizes[key]
            value = vector[offset:offset + size]
            # Reshape to the desired parameter shape
            if len(shape) > 1:
                value = np.reshape(value, shape + batch_shape)
            values[key] = value
            offset += size
        return values
