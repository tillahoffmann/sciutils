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


@pytest.mark.slow
def test_maybe_build_model(caplog):
    with tempfile.TemporaryDirectory() as tempdir:
        with caplog.at_level(logging.INFO):
            sus.maybe_build_model('data {}', tempdir)
        assert 'dumped model to' in caplog.text

        with caplog.at_level(logging.INFO):
            sus.maybe_build_model('data {}', tempdir)
        assert 'loaded model from' in caplog.text


@pytest.mark.parametrize('lin', [200, np.linspace(3, 5, 357)])
def test_evaluate_mode(lin):
    x = np.random.normal(4, 1, 20000)
    mode = sus.evaluate_mode(x, lin)
    assert -4.25 < mode < 4.25


@pytest.mark.parametrize('pvals', [4, 1 - np.arange(1, 5) / 5])
def test_evaluate_hpd_levels(pvals):
    x = np.linspace(-10, 10, 201)
    pdf = stats.norm.pdf(x)
    actual = sus.evaluate_hpd_levels(pdf, pvals)
    desired = stats.norm.pdf(stats.norm.ppf([0.1, 0.2, 0.3, 0.4]))
    np.testing.assert_array_less(np.abs(actual - desired), 0.01)


def test_evaluate_hpd_mass():
    x = np.linspace(-10, 10, 501)
    pdf = stats.norm.pdf(x)
    mass = sus.evaluate_hpd_mass(pdf)
    (a, b), = np.nonzero(np.diff(mass < 0.5))
    np.testing.assert_array_less(np.abs(x[a] - stats.norm.ppf(.25)), 0.01)
    np.testing.assert_array_less(np.abs(x[b + 1] - stats.norm.ppf(.75)), 0.01)
