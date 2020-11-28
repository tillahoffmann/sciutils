import itertools as it
import numpy as np
import pytest
import sciutils as su


@pytest.mark.parametrize('batch_shape, moveaxis', it.product([(), (10,), (5, 13)], [True, False]))
def test_parameter_reshaper(batch_shape, moveaxis):
    reshaper = su.ParameterReshaper({
        'a': None,
        'b': (4,),
        'd': 9,
        'c': (9, 7),
    })
    if moveaxis:
        shape = batch_shape + (reshaper.size,)
    else:
        shape = (reshaper.size,) + batch_shape
    x = np.random.normal(0, 1, shape)
    values = reshaper.to_dict(x, moveaxis)
    for key, value in values.items():
        assert value.shape[len(reshaper.parameters[key]):] == batch_shape
    y = reshaper.to_array(values, moveaxis)
    np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('shape', [1, (7, 3), (5, 3, 7)])
def test_invalid_array_size(shape):
    reshaper = su.ParameterReshaper({'a': 3})
    x = np.random.normal(0, 1, shape)
    with pytest.raises(ValueError):
        reshaper.to_dict(x)


def test_inconsistent_batch_shape():
    reshaper = su.ParameterReshaper({'a': 3, 'b': 2})
    y = {
        'a': np.random.normal(0, 1, (3, 4)),
        'b': np.random.normal(0, 1, (2, 5)),
    }
    with pytest.raises(ValueError):
        reshaper.to_array(y)


def test_extraneous_key():
    reshaper = su.ParameterReshaper({'a': 2})
    with pytest.raises(KeyError):
        reshaper.to_array({'a': [0, 1], 'b': 3})


def test_inconsistent_value_shape():
    reshaper = su.ParameterReshaper({'a': 2})
    with pytest.raises(ValueError):
        reshaper.to_array({'a': [0, 1, 2]})
