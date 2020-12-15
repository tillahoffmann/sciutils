import logging
import numpy as np
import pytest
from scipy import stats
from sciutils import stats as sus
import tempfile


@pytest.mark.parametrize('dist, logpdf, logcdf, args', [
    (
        stats.norm,
        sus.normal_logpdf,
        sus.normal_logcdf,
        (np.random.normal(0, 1, 100), np.random.gamma(1, size=100)),
    ),
    (
        stats.cauchy,
        sus.cauchy_logpdf,
        sus.cauchy_logcdf,
        (np.random.normal(0, 1, 100), np.random.gamma(1, size=100)),
    ),
    (
        stats.halfcauchy,
        sus.halfcauchy_logpdf,
        sus.halfcauchy_logcdf,
        (np.random.normal(0, 1, 100), np.random.gamma(1, size=100)),
    ),
])
def test_logpdf_cdf(dist, logpdf, logcdf, args):
    dist = dist(*args)
    x = dist.rvs()
    # Test the log pdf
    desired = dist.logpdf(x)
    actual = logpdf(x, *args)
    np.testing.assert_allclose(actual, desired)

    # Test the log cdf
    desired = dist.logcdf(x)
    actual = logcdf(x, *args)
    np.testing.assert_allclose(actual, desired)


def test_maybe_build_model(caplog):
    with tempfile.TemporaryDirectory() as tempdir:
        with caplog.at_level(logging.INFO):
            sus.maybe_build_model('data {}', tempdir)
        assert 'dumped model to' in caplog.text

        with caplog.at_level(logging.INFO):
            sus.maybe_build_model('data {}', tempdir)
        assert 'loaded model from' in caplog.text
