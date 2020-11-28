import numpy as np


class ParameterReshaper:
    """
    Reshape an array of parameters to a dictionary of named parameters and vice versa.

    Trailing dimensions of each parameter are considered batch dimensions and are left unchanged.

    Parameters
    ----------
    parameters : dict[str, tuple]
        Mapping from parameter names to shapes.

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

    def to_array(self, values, moveaxis=False, validate=True):
        """
        Convert a dictionary of values to an array.

        Parameters
        ----------
        values : dict[str, np.ndarray]
            Mapping from parameter names to values.
        moveaxis : bool
            Move the first axis to the last dimension after reshaping to an array, e.g. if the batch
            dimensions are leading.
        validate : bool
            Validate the input at some cost to performance.

        Returns
        -------
        array : np.ndarray
            Array of parameters encoding the named parameters.
        """
        if validate:
            batch_shapes = set()
            for key, value in values.items():
                shape = self.parameters.get(key)
                if shape is None:
                    raise KeyError(f'extraneous parameter: {key}')
                actual_shape = np.shape(value)[:len(shape)]
                if actual_shape != shape:
                    raise ValueError(f'the leading shape of {key} is {actual_shape}; '
                                     f'expected {shape}')
                batch_shapes.add(np.shape(value)[len(shape):])
            if len(batch_shapes) > 1:
                raise ValueError('parameters have inconsistent batch shapes: ' +
                                 ', '.join(map(str, batch_shapes)))
        array = None
        offset = 0
        for key, shape in self.parameters.items():
            value = values[key]
            if array is None:
                batch_shape = np.shape(value)[len(shape):]
                array = np.empty((self.size,) + batch_shape)
            # Just set the scalar element for a 0-rank tensor
            if not shape:
                array[offset] = value
                offset += 1
                continue
            # Flatten any tensor with rank two or larger
            if len(shape) > 1:
                value = np.reshape(value, (-1,) + batch_shape)
            # Set the array of parameters
            array[offset:offset + value.shape[0]] = value
            offset += value.shape[0]
        if moveaxis:
            array = np.moveaxis(array, 0, -1)
        return array

    def to_dict(self, array, moveaxis=False, validate=True):
        """
        Convert an array to a dictionary of values.

        Trailing dimensions of the array are considered batch dimensions and are left unchanged.

        Parameters
        ----------
        array : np.ndarray
            Array of parameters encoding a parameter set.
        moveaxis : bool
            Move the last axis to the first dimension before reshaping to a dictionary, e.g. if the
            batch dimensions are leading.
        validate : bool
            Validate the input at some cost to performance.

        Returns
        -------
        values : dict[str, np.ndarray]
            Mapping from parameter names to values.
        """
        if moveaxis:
            array = np.moveaxis(array, -1, 0)
        if validate:
            if array.shape[0] != self.size:
                raise ValueError(f'the first dimension of the array has size {array.shape[0]}; '
                                 f'expected {self.size}')
        values = {}
        offset = 0
        batch_shape = np.shape(array)[1:]
        for key, shape in self.parameters.items():
            # Handle the scalar case
            if not shape:
                values[key] = array[offset]
                offset += 1
                continue
            size = self.sizes[key]
            value = array[offset:offset + size]
            # Reshape to the desired parameter shape
            if len(shape) > 1:
                value = np.reshape(value, shape + batch_shape)
            values[key] = value
            offset += size
        return values
