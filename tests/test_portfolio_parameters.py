import numpy as np
import pytest

from app.models.portfolio_parameters import AssetParameters, PortfolioParameters


def _make_assets(n=3):
    return [
        AssetParameters(name=f"A{i}", initial_price=100, drift=0.08, volatility=0.2)
        for i in range(n)
    ]


def test_valid_portfolio_constructs():
    p = PortfolioParameters(
        assets=_make_assets(3),
        weights=[0.4, 0.3, 0.3],
        correlation_matrix=np.eye(3),
        portfolio_value=1_000_000,
        time_horizon=1.0,
    )
    assert p.n_assets == 3
    np.testing.assert_allclose(p.dollar_allocations, [400_000, 300_000, 300_000])
    assert p.asset_names == ["A0", "A1", "A2"]


def test_rejects_mismatched_weights_length():
    with pytest.raises(ValueError):
        PortfolioParameters(
            assets=_make_assets(3),
            weights=[0.5, 0.5],
            correlation_matrix=np.eye(3),
            portfolio_value=100,
            time_horizon=1.0,
        )


def test_rejects_non_square_correlation():
    with pytest.raises(ValueError):
        PortfolioParameters(
            assets=_make_assets(2),
            weights=[0.5, 0.5],
            correlation_matrix=np.eye(3),
            portfolio_value=100,
            time_horizon=1.0,
        )


def test_rejects_asymmetric_correlation():
    corr = np.array([[1.0, 0.5, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        PortfolioParameters(
            assets=_make_assets(3),
            weights=[0.3, 0.3, 0.4],
            correlation_matrix=corr,
            portfolio_value=100,
            time_horizon=1.0,
        )


def test_rejects_non_unit_diagonal():
    corr = np.array([[1.0, 0.5], [0.5, 0.9]])
    with pytest.raises(ValueError):
        PortfolioParameters(
            assets=_make_assets(2),
            weights=[0.5, 0.5],
            correlation_matrix=corr,
            portfolio_value=100,
            time_horizon=1.0,
        )


def test_rejects_out_of_range_correlation():
    corr = np.array([[1.0, 1.5], [1.5, 1.0]])
    with pytest.raises(ValueError):
        PortfolioParameters(
            assets=_make_assets(2),
            weights=[0.5, 0.5],
            correlation_matrix=corr,
            portfolio_value=100,
            time_horizon=1.0,
        )


def test_rejects_non_psd_correlation():
    # A classic non-positive-semi-definite 3x3 correlation-like matrix.
    corr = np.array(
        [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
    )
    with pytest.raises(ValueError):
        PortfolioParameters(
            assets=_make_assets(3),
            weights=[0.3, 0.3, 0.4],
            correlation_matrix=corr,
            portfolio_value=100,
            time_horizon=1.0,
        )


def test_rejects_nonpositive_portfolio_value():
    with pytest.raises(ValueError):
        PortfolioParameters(
            assets=_make_assets(2),
            weights=[0.5, 0.5],
            correlation_matrix=np.eye(2),
            portfolio_value=0,
            time_horizon=1.0,
        )


def test_asset_parameters_rejects_bad_price():
    with pytest.raises(ValueError):
        AssetParameters(name="X", initial_price=0, drift=0.05, volatility=0.2)


def test_asset_parameters_rejects_negative_vol():
    with pytest.raises(ValueError):
        AssetParameters(name="X", initial_price=100, drift=0.05, volatility=-0.1)
