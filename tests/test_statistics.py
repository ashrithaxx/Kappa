import numpy as np
import pytest

from app.quant.statistics import (
    confidence_interval,
    descriptive_statistics,
    standard_error,
)


def test_standard_error_known_value():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.std(data, ddof=1) / np.sqrt(5)
    assert standard_error(data) == pytest.approx(expected)


def test_confidence_interval_normal_sample_contains_true_mean_most_of_time():
    rng = np.random.default_rng(0)
    true_mean = 100.0
    contained = 0
    trials = 200
    for i in range(trials):
        sample = rng.normal(loc=true_mean, scale=10, size=200)
        ci = confidence_interval(sample, level=0.95)
        if ci.lower <= true_mean <= ci.upper:
            contained += 1
    # Expect roughly 95% coverage; allow generous slack for randomness.
    assert contained / trials > 0.85


def test_confidence_interval_widens_with_level():
    rng = np.random.default_rng(1)
    sample = rng.normal(size=1000)
    ci90 = confidence_interval(sample, level=0.90)
    ci99 = confidence_interval(sample, level=0.99)
    width90 = ci90.upper - ci90.lower
    width99 = ci99.upper - ci99.lower
    assert width99 > width90


def test_confidence_interval_rejects_bad_level():
    with pytest.raises(ValueError):
        confidence_interval(np.array([1.0, 2.0, 3.0]), level=1.5)


def test_descriptive_statistics_matches_numpy():
    data = np.array([10.0, 12.0, 9.0, 15.0, 11.0, 13.0, 8.0, 14.0])
    report = descriptive_statistics(data)
    assert report.mean == pytest.approx(float(np.mean(data)))
    assert report.median == pytest.approx(float(np.median(data)))
    assert report.std == pytest.approx(float(np.std(data, ddof=1)))
    assert report.variance == pytest.approx(float(np.var(data, ddof=1)))
    assert report.minimum == pytest.approx(float(np.min(data)))
    assert report.maximum == pytest.approx(float(np.max(data)))
    assert 50 in report.percentiles
    assert 95 in report.confidence_intervals


def test_descriptive_statistics_lognormal_positive_skew():
    rng = np.random.default_rng(2)
    # Lognormal data is right-skewed, as GBM terminal prices are.
    data = rng.lognormal(mean=0.0, sigma=0.5, size=5000)
    report = descriptive_statistics(data)
    assert report.skewness > 0
