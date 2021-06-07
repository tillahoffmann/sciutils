from sciutils import util as suu
import itertools as it
import numpy as np
import pytest


def test_transpose_dict_roundtrip():
    list_of_dicts = [
        {'a': 2, 'b': 3},
        {'a': 4, 'b': 7},
    ]
    dict_of_lists = suu.dict_list_transpose(list_of_dicts)
    assert dict_of_lists == {
        'a': [2, 4],
        'b': [3, 7],
    }
    assert list_of_dicts == suu.dict_list_transpose(dict_of_lists)


@pytest.mark.parametrize('weights', [True, False])
def test_bincountnd(weights):
    shape = (3, 5, 7, 11)
    ndim = len(shape)
    n = 100
    xs = (np.random.uniform(0, 1, (n, ndim)) * shape).astype(int)
    weights = np.random.normal(0, 1, n) if weights else None
    counts = suu.bincountnd(xs.T, weights)

    assert counts.shape == shape
    for idx in it.product(*[range(dim) for dim in shape]):
        fltr = np.all(xs == idx, axis=1)
        if weights is None:
            count = fltr.sum()
        else:
            count = weights[fltr].sum()
        assert counts[idx] == count
