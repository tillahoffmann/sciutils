import numpy as np
import pytest
import sciutils


@pytest.mark.parametrize('batch_shape', [(), (10,), (5, 13)])
def test_parameter_reshaper(batch_shape):
    reshaper = sciutils.ParameterReshaper({
        'a': None,
        'b': (4,),
        'd': 9,
        'c': (9, 7),
    })
    x = np.random.normal(0, 1, (reshaper.size,) + batch_shape)
    values = reshaper.to_dict(x)
    for key, value in values.items():
        assert value.shape[len(reshaper.parameters[key]):] == batch_shape
    y = reshaper.to_array(values)
    np.testing.assert_array_equal(x, y)
