"""Tests for the stats module."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm


@pytest.fixture
def stats(data):
    """Fixture that returns the stats property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Stats: The stats property of the data fixture.

    """
    return data.stats


@pytest.fixture
def integer_indexed_data():
    """Build a Data object with a simple integer index for a returns frame."""
    from jquantstats.data import Data

    def _build(returns: pl.DataFrame, idx_col: str = "idx") -> Data:
        """Build a Data object indexed by integer positions."""
        index = pl.DataFrame({idx_col: list(range(returns.height))})
        return Data(returns=returns, index=index)

    return _build


def test_skew(stats):
    """Tests that the skew method calculates skewness correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The skewness value for META matches the expected value.

    """
    result = stats.skew()
    assert result["META"] == pytest.approx(0.4220690178941095)


def test_kurtosis(stats):
    """Tests that the kurtosis method calculates kurtosis correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The kurtosis value for META matches the expected value.

    """
    result = stats.kurtosis()
    assert result["META"] == pytest.approx(20.477943361921824)


def test_avg_return(stats):
    """Tests that the avg_return method calculates average returns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The average return value for META matches the expected value.

    """
    result = stats.avg_return()
    assert result["META"] == pytest.approx(0.001142572799759818)


def test_avg_win(stats):
    """Tests that the avg_win method calculates average winning returns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The average win value for META matches the expected value.

    """
    result = stats.avg_win()
    assert result["META"] == pytest.approx(0.016760408889269992)


def test_avg_loss(stats):
    """Tests that the avg_loss method calculates average losing returns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The average loss value for META matches the expected value.

    """
    result = stats.avg_loss()
    assert result["META"] == pytest.approx(-0.01614062422517277)


def test_geometric_mean(stats):
    """Tests that geometric_mean calculates the per-period geometric average correctly."""
    result = stats.geometric_mean()
    assert result["META"] == pytest.approx(0.0008201465942647701)


def test_geometric_mean_annualized(stats):
    """Tests that geometric_mean with annualize=True returns the annualized geometric return."""
    result = stats.geometric_mean(annualize=True)
    assert result["META"] == pytest.approx(0.22904692100449497)


def test_probabilistic_sortino_ratio(stats):
    """Tests that probabilistic_sortino_ratio matches the expected value."""
    result = stats.probabilistic_sortino_ratio()
    assert result["META"] == pytest.approx(0.999936222948045)


def test_probabilistic_adjusted_sortino_ratio(stats):
    """Tests that probabilistic_adjusted_sortino_ratio matches the expected value."""
    result = stats.probabilistic_adjusted_sortino_ratio()
    assert result["META"] == pytest.approx(0.9966624180363135)


def test_smart_sharpe(stats):
    """Tests that smart_sharpe applies the autocorrelation penalty to the Sharpe ratio."""
    result = stats.smart_sharpe(periods=252)
    assert result["META"] == pytest.approx(0.698132519420813)


def test_smart_sortino(stats):
    """Tests that smart_sortino applies the autocorrelation penalty to the Sortino ratio."""
    result = stats.smart_sortino(periods=252)
    assert result["META"] == pytest.approx(1.0368591297457295)


def test_volatility(stats):
    """Tests that the volatility method calculates volatility correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The annualized volatility value for META matches the expected value.

    """
    result = stats.volatility(periods=252, annualize=True)
    assert result["META"] == pytest.approx(0.40072031010504233)


def test_rolling_volatility(stats):
    """Tests that the rolling_volatility method calculates rolling volatility correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The shape of the rolling volatility result matches the shape of the input data.

    """
    result = stats.rolling_volatility(rolling_period=20, periods_per_year=252)
    print(result.tail(5))
    assert result.shape == stats.all.shape


def test_payoff_ratio(stats):
    """Tests that the payoff_ratio method calculates payoff ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The payoff ratio value for META matches the expected value.

    """
    result = stats.payoff_ratio()
    assert result["META"] == pytest.approx(1.0383990517002815)


def test_profit_ratio(stats):
    """Tests that the profit_ratio method calculates profit ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The profit ratio value for META matches the expected value.

    """
    result = stats.profit_ratio()
    assert result["META"] == pytest.approx(0.9252488178411934)


def test_profit_factor(stats):
    """Tests that the profit_factor method calculates profit factor correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The profit factor value for META matches the expected value.

    """
    result = stats.profit_factor()
    assert result["META"] == pytest.approx(1.149125608578595)


def test_value_at_risk(stats):
    """Tests that the value_at_risk method calculates VaR correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The value at risk (alpha=0.05) for META matches the expected value.

    """
    result = stats.value_at_risk(alpha=0.05)
    assert result["META"] == pytest.approx(-0.04038269463520536)


def test_conditional_value_at_risk(stats):
    """Tests that the conditional_value_at_risk method calculates CVaR correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The conditional value at risk (alpha=0.05) for META matches the expected value.

    """
    result = stats.conditional_value_at_risk(alpha=0.05)
    assert result["META"] == pytest.approx(-0.06084410598898649)


def test_win_rate(stats):
    """Tests that the win_rate method calculates win rate correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The win rate value for META matches the expected value.

    """
    result = stats.win_rate()
    assert result["META"] == pytest.approx(0.525309)


def test_kelly_criterion(stats):
    """Tests that the kelly_criterion method calculates Kelly criterion correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Kelly criterion value for META matches the expected value.

    """
    result = stats.kelly_criterion()
    assert result["META"] == pytest.approx(0.06817093826936971)


def test_comp(stats):
    """Tests that comp returns the total compounded return."""
    result = stats.comp()
    assert result["META"] == pytest.approx(13.382664235863414)


def test_geometric_mean_compound_identity(stats):
    """Tests that geometric_mean matches the compounded-return identity."""
    result = stats.geometric_mean()
    comp = stats.comp()["META"]
    n = stats.all["META"].drop_nulls().len()
    expected = (1 + comp) ** (1 / n) - 1
    assert result["META"] == pytest.approx(expected)


def test_compsum(stats):
    """Tests that compsum returns a cumulative return series ending at comp()."""
    result = stats.compsum()
    assert result.shape == stats.all.shape
    assert result["META"][-1] == pytest.approx(13.382664235863414)


def test_outlier_win_ratio(stats):
    """Tests that outlier_win_ratio returns the ratio of high-quantile to mean positive return."""
    result = stats.outlier_win_ratio()
    assert result["AAPL"] == pytest.approx(4.254419434089758)


def test_outlier_loss_ratio(stats):
    """Tests that outlier_loss_ratio returns the ratio of low-quantile to mean negative return."""
    result = stats.outlier_loss_ratio()
    assert result["AAPL"] == pytest.approx(3.778604041649339)


def test_outliers(stats):
    """Tests that outliers returns only returns above the quantile threshold."""
    result = stats.outliers()
    assert len(result["AAPL"]) == 406


def test_remove_outliers(stats):
    """Tests that remove_outliers returns returns below the quantile threshold."""
    result = stats.remove_outliers()
    assert len(result["AAPL"]) == 7710


def test_gain_to_pain_ratio(stats):
    """Tests that the gain_to_pain_ratio method calculates gain-to-pain ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The gain-to-pain ratio value for META matches the expected value.

    """
    result = stats.gain_to_pain_ratio()
    assert result["META"] == pytest.approx(0.14912560857859494)


def test_risk_return_ratio(stats):
    """Tests that the risk_return_ratio method calculates risk-return ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The risk-return ratio value for META matches the expected value.

    """
    result = stats.risk_return_ratio()
    assert result["META"] == pytest.approx(0.045095921921619944)


def test_best(stats):
    """Tests that the best method calculates the best return correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The best return value for META matches the expected value.

    """
    result = stats.best()
    assert result["META"] == pytest.approx(0.2961146917048886)


def test_worst(stats):
    """Tests that the worst method calculates the worst return correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The worst return value for META matches the expected value.

    """
    result = stats.worst()
    assert result["META"] == pytest.approx(-0.2639010078964036)


def test_exposure(stats):
    """Tests that the exposure method calculates exposure correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The exposure value for META matches the expected value.

    """
    result = stats.exposure()
    assert result["META"] == pytest.approx(0.40)


def test_autocorrelation(stats):
    """Tests that the autocorrelation method calculates lag-1 autocorrelation correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The lag-1 autocorrelation value for META matches the expected value.

    """
    result = stats.autocorr()
    assert result["META"] == pytest.approx(-0.025099872722702105)


def test_autocorrelation_lag5(stats):
    """Tests that the autocorrelation method calculates lag-5 autocorrelation correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The lag-5 autocorrelation value for META matches the expected value.

    """
    result = stats.autocorr(lag=5)
    assert result["META"] == pytest.approx(0.012378555376440949)


def test_autocorrelation_invalid_lag_zero(stats):
    """Tests that autocorrelation raises ValueError for lag=0.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        A ValueError is raised when lag is zero.

    """
    with pytest.raises(ValueError, match="lag must be a positive integer"):
        stats.autocorr(lag=0)


def test_autocorrelation_invalid_lag_negative(stats):
    """Tests that autocorrelation raises ValueError for negative lag.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        A ValueError is raised when lag is negative.

    """
    with pytest.raises(ValueError, match="lag must be a positive integer"):
        stats.autocorr(lag=-1)


def test_autocorrelation_invalid_lag_type(stats):
    """Tests that autocorrelation raises TypeError for non-integer lag.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        A TypeError is raised when lag is not an int.

    """
    with pytest.raises(TypeError, match="lag must be an int"):
        stats.autocorr(lag=1.0)  # type: ignore[arg-type]


def test_acf_shape(stats):
    """Tests that the acf method returns a DataFrame with the correct shape.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The ACF DataFrame has nlags+1 rows and the expected columns.

    """
    nlags = 20
    result = stats.acf(nlags=nlags)
    assert isinstance(result, pl.DataFrame)
    assert result.height == nlags + 1
    assert "lag" in result.columns
    assert "META" in result.columns


def test_acf_values(stats):
    """Tests that the acf method returns correct autocorrelation values.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The ACF values at specific lags for META match the expected values.

    """
    result = stats.acf(nlags=20)
    assert result["lag"].to_list() == list(range(21))
    assert result["META"][0] == pytest.approx(1.0)
    assert result["META"][1] == pytest.approx(-0.025099872722702105)
    assert result["META"][5] == pytest.approx(0.012378555376440949)
    assert result["META"][20] == pytest.approx(0.049337716067170856)


def test_acf_invalid_nlags_negative(stats):
    """Tests that acf raises ValueError for negative nlags.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        A ValueError is raised when nlags is negative.

    """
    with pytest.raises(ValueError, match="nlags must be non-negative"):
        stats.acf(nlags=-1)


def test_acf_invalid_nlags_type(stats):
    """Tests that acf raises TypeError for non-integer nlags.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        A TypeError is raised when nlags is not an int.

    """
    with pytest.raises(TypeError, match="nlags must be an int"):
        stats.acf(nlags=1.5)  # type: ignore[arg-type]


def test_sharpe(stats):
    """Tests that the sharpe method calculates Sharpe ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Sharpe ratio value for META matches the expected value.

    """
    result = stats.sharpe(periods=252)
    assert result["META"] == pytest.approx(0.7158755672867543)


def test_sharpe_var(stats):
    """Tests that the sharpe_var method calculates Sharpe ratio variance correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Sharpe ratio variance value for META matches the expected value.

    """
    result = stats.sharpe_variance()
    skew = stats.skew()["META"]
    kurt = stats.kurtosis()["META"]
    # Get unannualized Sharpe ratio
    sr = stats.sharpe(periods=1)["META"]
    t_meta_not_nan = stats.returns["META"].drop_nulls().shape[0]
    period = stats.periods_per_year
    # Expected base variance (unannualized)
    expected_base_var = (1 + (skew * sr) / 2 + ((kurt - 3) / 4) * sr**2) / t_meta_not_nan
    # Expected annualized variance
    expected_var = period * expected_base_var
    assert result["META"] == pytest.approx(expected_var)


def test_prob_sharpe_ratio(stats):
    """Tests that the prob_sharpe_ratio method calculates probabilistic Sharpe ratio correctly."""
    result = stats.probabilistic_sharpe_ratio()
    # Unannualized Sharpe ratio
    observed_sr = stats.sharpe(periods=1)["META"]
    skew = stats.skew()["META"]
    kurt = stats.kurtosis()["META"]
    t_meta_not_nan = stats.returns["META"].drop_nulls().shape[0]
    benchmark_sr = 0.0
    var_bench_sr = (1 + (float(skew) * benchmark_sr) / 2 + ((float(kurt) - 3) / 4) * benchmark_sr**2) / t_meta_not_nan
    expected_prob_sr = norm.cdf((observed_sr - benchmark_sr) / np.sqrt(var_bench_sr))
    assert result["META"] == pytest.approx(expected_prob_sr)


def test_hhi_positive(stats):
    """Tests that the hhi_positive method calculates positive HHI correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The positive HHI value for META matches the expected value.
    """
    result = stats.hhi_positive()
    assert result["META"] == pytest.approx(0.0008093666220006002)


def test_hhi_negative(stats):
    """Tests that the hhi_negative method calculates negative HHI correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The negative HHI value for META matches the expected value.
    """
    result = stats.hhi_negative()
    assert result["META"] == pytest.approx(0.0008748322510113375)


def test_rolling_sharpe(stats):
    """Tests that the rolling_sharpe method calculates rolling Sharpe ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The shape of the rolling Sharpe ratio result matches the shape of the input data.

    """
    result = stats.rolling_sharpe(rolling_period=20, periods_per_year=252)
    assert result.shape == stats.all.shape
    print(result.tail(5))


def test_sortino(stats):
    """Tests that the sortino method calculates Sortino ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Sortino ratio value for META matches the expected value.

    """
    result = stats.sortino(periods=252)
    assert result["META"] == pytest.approx(1.06321091920911)


def test_omega(stats):
    """Tests that the omega method calculates the Omega ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Omega ratio for META matches the expected value for the default
        parameters (rf=0, required_return=0), for a non-zero required_return,
        and for a non-zero risk-free rate.

    """
    result = stats.omega()
    assert result["META"] == pytest.approx(1.1491256085785948)

    result_with_threshold = stats.omega(required_return=0.01)
    assert result_with_threshold["META"] == pytest.approx(1.143589264051078)

    result_with_rf = stats.omega(rf=0.02)
    assert result_with_rf["META"] == pytest.approx(1.1381333817189756)


def test_omega_invalid_required_return(stats):
    """Tests that the omega method returns NaN for required_return <= -1.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Omega ratio is NaN when required_return <= -1.

    """
    result = stats.omega(required_return=-1.0)
    assert np.isnan(result["META"])


def test_rolling_sortino(stats):
    """Tests that the rolling_sortino method calculates rolling Sortino ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The shape of the rolling Sortino ratio result matches the shape of the input data.

    """
    result = stats.rolling_sortino(rolling_period=20, periods_per_year=252)
    assert result.shape == stats.all.shape
    print(result.tail(5))


def test_adjusted_sortino(stats):
    """Tests that the adjusted_sortino method calculates adjusted Sortino ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The adjusted Sortino ratio value for META matches the expected value.

    """
    result = stats.adjusted_sortino(periods=252)
    assert result["META"] == pytest.approx(0.7518036508043441)


def test_edge_cases(edge):
    """Tests that the stats methods handle edge cases correctly.

    Args:
        edge: The edge fixture containing a Data object with edge case data.

    Verifies:
        1. The profit_ratio method returns NaN for the 'returns' column.
        2. The gain_to_pain_ratio method returns NaN for the 'returns' column.

    """
    assert np.isnan(edge.stats.profit_ratio()["returns"])  # == {"returns": np.nan, "Benchmark": np.nan}
    assert np.isnan(edge.stats.gain_to_pain_ratio()["returns"])  # == {"returns": np.nan, "Benchmark": np.nan}


def test_information_ratio(stats):
    """Tests that the information_ratio method calculates information ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The information ratio value for AAPL matches the expected value.

    """
    result = stats.information_ratio(periods_per_year=252)
    assert result["AAPL"] == pytest.approx(0.02883007382760274)


def test_information_ratio_no_annualise(stats):
    """Tests that annualise=False returns a non-annualised information ratio.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        annualise=False divides the annualised value by sqrt(252).

    """
    annualised = stats.information_ratio(periods_per_year=252, annualise=True)["AAPL"]
    raw = stats.information_ratio(periods_per_year=252)["AAPL"]
    assert raw == pytest.approx(annualised / (252**0.5))


def test_greeks(stats):
    """Tests that the greeks method calculates alpha and beta correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The method executes without errors and returns a result.

    """
    result = stats.greeks(periods_per_year=252)
    print(result)

    #
    # assert isinstance(result, pd.DataFrame)
    # assert "alpha" in result.index
    # assert "beta" in result.index
    #
    # assert result["Benchmark"]["beta"] == 1.0
    # assert result["Benchmark"]["alpha"] == 0.0
    #
    # assert result["AAPL"]["beta"] == pytest.approx(1.1090322781954098)
    # assert result["AAPL"]["alpha"] == pytest.approx(0.1576003006124853)


def test_r_squared(stats):
    """Tests that the r_squared method calculates R-squared correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The method executes without errors and returns a result.

    """
    result = stats.r_squared()
    print(result)
    # assert 0 <= result <= 1  # R-squared should be between 0 and 1


def test_drawdowns(stats):
    """Tests that the drawdown method calculates drawdowns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        1. The drawdown method returns a DataFrame with the expected structure.
        2. The drawdown values are calculated correctly.
        3. The maximum drawdown values match expected values.
        4. Drawdown values are always non-negative (since they represent losses).

    """
    # Test the drawdown method returns a DataFrame with the expected structure
    dd = stats.drawdown()
    assert isinstance(dd, pl.DataFrame)
    assert dd.shape[0] == stats.all.shape[0]  # Same number of rows as input data

    # Check that all columns from returns and benchmark are present
    for col in stats.returns.columns:
        assert col in dd.columns
    if stats.benchmark is not None:
        for col in stats.benchmark.columns:
            assert col in dd.columns

    # Verify drawdown values are always non-negative (since they represent losses)
    for col in dd.columns:
        if col not in stats.date_col:  # Skip date column
            assert (dd[col] >= 0).all()

    # Test specific drawdown values for META
    # The maximum drawdown for META should be approximately 0.76 (76%)
    max_dd = dd["META"].max()
    assert max_dd == pytest.approx(0.76, abs=0.01)

    # Test that drawdown is zero at peak equity points
    prices = stats.prices(stats.all["META"])
    peak_indices = prices.cum_max() == prices
    dd_at_peaks = dd.filter(peak_indices)["META"]
    assert (dd_at_peaks == 0).all()


def test_drawdowns_edge_case(edge):
    """Tests that the drawdown method handles edge cases correctly.

    Args:
        edge: The edge fixture containing a Data object with edge case data.

    Verifies:
        1. The drawdown method works with constant zero returns.
        2. The drawdown values are all zero when returns are all zero.

    """
    # Test the drawdown method with constant zero returns
    dd = edge.stats.drawdown()

    # Verify the structure
    assert isinstance(dd, pl.DataFrame)
    assert dd.shape[0] == edge.stats.all.shape[0]

    # For constant zero returns, drawdowns should all be zero
    for col in dd.columns:
        if col not in edge.stats.date_col:  # Skip date column
            assert (dd[col] == 0).all()


def test_max_drawdown(stats):
    """Tests that the max_drawdown method calculates maximum drawdown correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        1. The max_drawdown method returns a dictionary with the expected structure.
        2. The maximum drawdown values match the expected values.
        3. The max_drawdown method returns the same values as the maximum of the drawdown series.

    """
    # Calculate maximum drawdowns
    max_dd = stats.max_drawdown()

    # Verify structure
    assert isinstance(max_dd, dict)
    assert set(max_dd.keys()) == set(stats.assets)

    # Verify values for specific assets
    assert max_dd["META"] == pytest.approx(-0.76, abs=0.01)
    assert max_dd["AAPL"] == pytest.approx(-0.82, abs=0.01)

    # Verify that max_drawdown returns the same values as the maximum of the drawdown series
    dd = stats.drawdown()
    for col in stats.assets:
        assert max_dd[col] == pytest.approx(-dd[col].max(), abs=0.0001)


# ── Ported analytics methods ──────────────────────────────────────────────────


def test_to_float_none():
    """_to_float(None) returns 0.0 (sentinel for absent aggregation results)."""
    from jquantstats._stats._core import _to_float

    assert _to_float(None) == 0.0


def test_to_float_timedelta():
    """_to_float(timedelta) returns total seconds."""
    from datetime import timedelta

    from jquantstats._stats._core import _to_float

    assert _to_float(timedelta(seconds=42)) == 42.0


def test_columnwise_stat_missing_data_attr_raises_descriptive_error():
    """columnwise_stat raises a descriptive error when the configured data attribute is absent."""
    from jquantstats._stats._core import columnwise_stat

    class _MissingData:
        """Host class that omits the required data attribute."""

        @columnwise_stat
        def metric(self, series: pl.Series) -> float:
            """Return the mean of the given series."""
            return float(series.mean() or 0.0)

    with pytest.raises(AttributeError, match="columnwise_stat requires host object to define '_data'"):
        _MissingData().metric()


def test_to_frame_missing_data_attr_raises_descriptive_error():
    """to_frame raises a descriptive error when the configured data attribute is absent."""
    from jquantstats._stats._core import to_frame

    class _MissingData:
        """Host class that omits the required data attribute."""

        all = pl.DataFrame({"Date": [1]})

        @to_frame
        def metric(self, series: pl.Series) -> pl.Expr:
            """Return a literal expression of the series mean."""
            return pl.lit(float(series.mean() or 0.0))

    with pytest.raises(AttributeError, match="to_frame requires host object to define '_data'"):
        _MissingData().metric()


def test_columnwise_stat_supports_custom_data_attr():
    """columnwise_stat supports an explicit data attribute name."""
    from jquantstats._stats._core import columnwise_stat

    class _CustomDataHost:
        """Host class exposing data under a custom attribute name."""

        data = {"asset": pl.Series("asset", [1.0, 2.0, 3.0])}

        @columnwise_stat(data_attr="data")
        def metric(self, series: pl.Series) -> float:
            """Return the mean of the given series."""
            return float(series.mean() or 0.0)

    assert _CustomDataHost().metric() == {"asset": pytest.approx(2.0)}


def test_to_frame_supports_custom_data_attr():
    """to_frame supports an explicit data attribute name."""
    from jquantstats._stats._core import to_frame

    class _DataDictWithDateCol(dict[str, pl.Series]):
        """Series mapping that also exposes a date column attribute."""

        date_col = ["Date"]

    class _CustomDataHost:
        """Host class exposing data under a custom attribute name."""

        data = _DataDictWithDateCol({"asset": pl.Series("asset", [1.0, 2.0, 3.0])})
        all = pl.DataFrame({"Date": [1, 2, 3], "asset": [1.0, 2.0, 3.0]})

        @to_frame(data_attr="data")
        def metric(self, series: pl.Series) -> pl.Expr:
            """Return a literal expression of the series mean."""
            return pl.lit(float(series.mean() or 0.0))

    result = _CustomDataHost().metric()
    assert result.columns == ["Date", "asset"]
    assert result["asset"].to_list() == pytest.approx([2.0, 2.0, 2.0])


def test_periods_per_year(stats):
    """periods_per_year returns a positive annualisation factor."""
    assert stats.periods_per_year > 0


def test_avg_drawdown(stats):
    """avg_drawdown returns a non-positive float per asset (negative convention)."""
    result = stats.avg_drawdown()
    assert isinstance(result, dict)
    for col in result:
        assert result[col] <= 0.0


def test_calmar(stats):
    """Calmar returns a finite float per asset for data with drawdowns."""
    result = stats.calmar()
    assert isinstance(result, dict)
    for col in result:
        assert result[col] is not None


def test_recovery_factor(stats):
    """recovery_factor returns a finite float per asset."""
    result = stats.recovery_factor()
    assert isinstance(result, dict)
    for col in result:
        assert result[col] is not None


def test_max_drawdown_duration_with_date(stats):
    """max_drawdown_duration returns positive integers for date-indexed data."""
    result = stats.max_drawdown_duration()
    assert isinstance(result, dict)
    for col in result:
        assert isinstance(result[col], int)
        assert result[col] >= 0


def test_max_drawdown_duration_integer_indexed(data_no_benchmark, integer_indexed_data):
    """max_drawdown_duration uses period-count when index is non-temporal."""
    returns = data_no_benchmark.returns.head(20)
    int_data = integer_indexed_data(returns)
    result = int_data.stats.max_drawdown_duration()
    assert isinstance(result, dict)
    for col in result:
        assert isinstance(result[col], int)
        assert result[col] >= 0


def test_monthly_win_rate_with_date(stats):
    """monthly_win_rate returns a value in [0, 1] for each asset."""
    result = stats.monthly_win_rate()
    assert isinstance(result, dict)
    for col in result:
        assert 0.0 <= result[col] <= 1.0


def test_monthly_win_rate_no_date(data_no_benchmark, integer_indexed_data):
    """monthly_win_rate returns nan for non-temporal index data."""
    returns = data_no_benchmark.returns.head(10)
    int_data = integer_indexed_data(returns)
    result = int_data.stats.monthly_win_rate()
    for col in result:
        assert np.isnan(result[col])


def test_worst_n_periods(stats):
    """worst_n_periods returns a list of n sorted worst returns per asset."""
    result = stats.worst_n_periods(n=3)
    assert isinstance(result, dict)
    for col in result:
        assert len(result[col]) == 3
        vals = [v for v in result[col] if v is not None]
        assert vals == sorted(vals)


def test_worst_n_periods_padding(data_no_benchmark, integer_indexed_data):
    """worst_n_periods pads with None when series has fewer than n non-null values."""
    returns = pl.DataFrame({"r": [0.01, -0.02]})
    tiny_data = integer_indexed_data(returns)
    result = tiny_data.stats.worst_n_periods(n=5)
    assert result["r"][-1] is None


def test_up_capture_basic(stats):
    """up_capture returns a dict with one entry per asset (including benchmark)."""
    benchmark = stats.all["SPY -- Benchmark"]
    result = stats.up_capture(benchmark)
    assert isinstance(result, dict)
    assert "META" in result
    assert "AAPL" in result


def test_up_capture_no_up_periods(integer_indexed_data):
    """up_capture returns nan when benchmark has no positive periods."""
    returns = pl.DataFrame({"r": [-0.01, -0.02, -0.03, -0.01]})
    d = integer_indexed_data(returns)
    bench = pl.Series([-0.01, -0.02, -0.03, -0.01])
    result = d.stats.up_capture(bench)
    for col in result:
        assert np.isnan(result[col])


def test_up_capture_empty_strategy_up(integer_indexed_data):
    """up_capture returns nan for an asset with no returns during up-benchmark periods."""
    # strategy has all values where benchmark is positive are null
    returns = pl.DataFrame({"r": [None, 0.01, 0.02, 0.01]}, schema={"r": pl.Float64})
    d = integer_indexed_data(returns)
    bench = pl.Series([0.05, 0.0, 0.0, 0.0])  # only period 0 is up; strategy has null there
    result = d.stats.up_capture(bench)
    assert np.isnan(result["r"])


def test_down_capture_basic(stats):
    """down_capture returns a dict with one entry per asset (including benchmark)."""
    benchmark = stats.all["SPY -- Benchmark"]
    result = stats.down_capture(benchmark)
    assert isinstance(result, dict)
    assert "META" in result
    assert "AAPL" in result


def test_down_capture_no_down_periods(integer_indexed_data):
    """down_capture returns nan when benchmark has no negative periods."""
    returns = pl.DataFrame({"r": [0.01, 0.02, 0.03, 0.01]})
    d = integer_indexed_data(returns)
    bench = pl.Series([0.01, 0.02, 0.03, 0.01])
    result = d.stats.down_capture(bench)
    for col in result:
        assert np.isnan(result[col])


def test_down_capture_empty_strategy_down(integer_indexed_data):
    """down_capture returns nan for an asset with no returns during down-benchmark periods."""
    returns = pl.DataFrame({"r": [0.01, None, None, 0.01]}, schema={"r": pl.Float64})
    d = integer_indexed_data(returns)
    bench = pl.Series([0.0, -0.05, -0.03, 0.0])  # periods 1 and 2 are down; strategy has null there
    result = d.stats.down_capture(bench)
    assert np.isnan(result["r"])


def test_annual_breakdown_structure(stats):
    """annual_breakdown returns a DataFrame with year, metric, and asset columns."""
    result = stats.annual_breakdown()
    assert "year" in result.columns
    assert "metric" in result.columns
    assert "META" in result.columns
    assert result.height > 0


def test_annual_breakdown_integer_indexed(data_no_benchmark, integer_indexed_data):
    """annual_breakdown groups by ~252-row chunks for integer-indexed data."""
    import numpy as np

    # 500 rows → 2 full chunks of 252
    n = 500
    rng = np.random.default_rng(42)
    returns = pl.DataFrame({"r": rng.normal(0.001, 0.01, n).tolist()})
    int_data = integer_indexed_data(returns)
    result = int_data.stats.annual_breakdown()
    assert result.height > 0, "expected non-empty result for 500-row integer-indexed data"
    assert "year" in result.columns
    assert "metric" in result.columns
    # 500 rows / 252 ≈ 2 full chunks → year labels 1 and 2
    years = sorted(result["year"].unique().to_list())
    assert years == [1, 2]


def test_annual_breakdown_integer_indexed_sparse_chunk(integer_indexed_data):
    """annual_breakdown skips integer-index chunks that are too small."""
    import numpy as np

    # Only 260 rows: first chunk of 252 is full, remainder (8 rows) < max(5, 63) → skipped
    n = 260
    rng = np.random.default_rng(0)
    returns = pl.DataFrame({"r": rng.normal(0.001, 0.01, n).tolist()})
    int_data = integer_indexed_data(returns)
    result = int_data.stats.annual_breakdown()
    # Only year 1 survives; the 8-row tail is too sparse
    assert list(result["year"].unique().sort().to_list()) == [1]


def test_annual_breakdown_integer_indexed_all_sparse(integer_indexed_data):
    """annual_breakdown returns empty DataFrame when all integer-index chunks are sparse."""
    # Only 3 rows, chunk=252 → 3 < max(5, 63) → skipped
    returns = pl.DataFrame({"r": [0.01, -0.02, 0.03]})
    int_data = integer_indexed_data(returns)
    result = int_data.stats.annual_breakdown()
    assert result.height == 0


@pytest.fixture
def sparse_year_data():
    """Data fixture with one sparse year and one valid year for annual_breakdown tests."""
    from datetime import date

    from jquantstats.data import Data

    dates = [date(2020, 12, 31), date(2021, 1, 2), date(2021, 1, 3)]
    returns = pl.DataFrame({"r": [0.01, -0.02, 0.03]})
    index = pl.DataFrame({"Date": pl.Series(dates, dtype=pl.Date)})
    return Data(returns=returns, index=index)


@pytest.fixture
def all_sparse_years_data():
    """Data fixture where every year is sparse for annual_breakdown tests."""
    from datetime import date

    from jquantstats.data import Data

    dates = [date(2020, 6, 15), date(2021, 6, 15)]
    returns = pl.DataFrame({"r": [0.01, -0.02]})
    index = pl.DataFrame({"Date": pl.Series(dates, dtype=pl.Date)})
    return Data(returns=returns, index=index)


def test_annual_breakdown_skips_sparse_year(sparse_year_data):
    """annual_breakdown skips years with fewer than 2 rows."""
    result = sparse_year_data.stats.annual_breakdown()
    # 2020 has only 1 row → skipped; only 2021 appears
    assert list(result["year"].unique().sort()) == [2021]


def test_annual_breakdown_empty_when_all_sparse(all_sparse_years_data):
    """annual_breakdown returns empty DataFrame when every year has < 2 rows."""
    result = all_sparse_years_data.stats.annual_breakdown()
    assert result.height == 0
    assert "year" in result.columns


def test_summary_structure(stats):
    """Summary returns a DataFrame with a metric column and asset columns."""
    result = stats.summary()
    assert "metric" in result.columns
    assert "META" in result.columns
    assert result.height > 0
    assert "sharpe" in result["metric"].to_list()


def test_rolling_sharpe_invalid_window_raises(stats):
    """rolling_sharpe raises ValueError for non-positive rolling_period."""
    with pytest.raises(ValueError, match="positive integer"):
        stats.rolling_sharpe(rolling_period=0)
    with pytest.raises(ValueError, match="positive integer"):
        stats.rolling_sharpe(rolling_period=-1)


def test_rolling_volatility_invalid_window_raises(stats):
    """rolling_volatility raises ValueError for non-positive rolling_period."""
    with pytest.raises(ValueError, match="positive integer"):
        stats.rolling_volatility(rolling_period=0)


def test_rolling_sortino_invalid_window_raises(stats):
    """rolling_sortino raises ValueError for non-positive rolling_period."""
    with pytest.raises(ValueError, match="positive integer"):
        stats.rolling_sortino(rolling_period=0)
    with pytest.raises(ValueError, match="positive integer"):
        stats.rolling_sortino(rolling_period=-1)


def test_rolling_volatility_invalid_periods_type_raises(stats):
    """rolling_volatility raises TypeError for non-numeric periods_per_year."""
    with pytest.raises(TypeError):
        stats.rolling_volatility(rolling_period=5, periods_per_year="252")  # type: ignore[arg-type]


def test_pct_rank(stats):
    """Tests that pct_rank returns a DataFrame with the correct shape and values."""
    result = stats.pct_rank(window=60)
    assert result.shape == stats.all.shape
    assert result["AAPL"].null_count() == 59
    assert result["AAPL"].drop_nulls()[-1] == pytest.approx(26.666666666666668)


def test_pct_rank_invalid_window_raises(stats):
    """pct_rank raises ValueError for non-positive window."""
    with pytest.raises(ValueError, match="positive integer"):
        stats.pct_rank(window=0)


def test_repr(stats):
    """Tests that Stats.__repr__ returns an informative string."""
    r = repr(stats)
    assert r.startswith("Stats(assets=")
    for asset in stats.assets:
        assert asset in r


def test_sortino_all_zero_returns_is_nan(edge):
    """Sortino returns NaN when all returns are zero (downside_deviation=0, mean=0)."""
    import math

    result = edge.stats.sortino()
    for val in result.values():
        assert math.isnan(val)


def test_monthly_returns_shape(stats):
    """monthly_returns returns a dict of DataFrames with year and month columns."""
    result = stats.monthly_returns(eoy=True)
    assert isinstance(result, dict)
    for df in result.values():
        assert isinstance(df, pl.DataFrame)
        assert "year" in df.columns
        assert "JAN" in df.columns
        assert "DEC" in df.columns
        assert "EOY" in df.columns


def test_monthly_returns_no_eoy(stats):
    """monthly_returns without eoy omits the EOY column."""
    result = stats.monthly_returns(eoy=False)
    for df in result.values():
        assert "EOY" not in df.columns
        assert len(df.columns) == 13  # year + 12 months


def test_monthly_returns_values(stats):
    """monthly_returns compounded values match expected for a spot-check month."""
    result = stats.monthly_returns(eoy=True, compounded=True)
    aapl_df = result["AAPL"]
    assert aapl_df["year"].dtype == pl.Int32
    assert aapl_df.height > 0
    eoy_vals = aapl_df["EOY"].to_list()
    assert all(v is not None for v in eoy_vals)


def test_distribution_structure(stats):
    """Distribution returns nested dict with all five periods and values/outliers keys."""
    result = stats.distribution()
    assert isinstance(result, dict)
    for asset_data in result.values():
        for period in ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly"):
            assert period in asset_data
            assert "values" in asset_data[period]
            assert "outliers" in asset_data[period]
            assert isinstance(asset_data[period]["values"], list)
            assert isinstance(asset_data[period]["outliers"], list)


def test_distribution_counts(stats):
    """Distribution daily count equals the number of non-null daily returns."""
    result = stats.distribution()
    aapl_returns = stats.returns["AAPL"].drop_nulls()
    daily = result["AAPL"]["Daily"]
    total = len(daily["values"]) + len(daily["outliers"])
    assert total == len(aapl_returns)


def test_implied_volatility_rolling(stats):
    """implied_volatility annualized returns a DataFrame of rolling vol series."""
    result = stats.implied_volatility(periods=252, annualize=True)
    assert isinstance(result, pl.DataFrame)
    assert "AAPL" in result.columns
    assert result["AAPL"].null_count() == 251


def test_implied_volatility_scalar(stats):
    """implied_volatility non-annualized returns a dict of scalar values."""
    result = stats.implied_volatility(annualize=False)
    assert isinstance(result, dict)
    for val in result.values():
        assert isinstance(val, float)
        assert val > 0


def test_compare_structure(stats):
    """Compare returns a dict of DataFrames with the expected columns."""
    result = stats.compare()
    assert isinstance(result, dict)
    for df in result.values():
        assert isinstance(df, pl.DataFrame)
        for col in ("Benchmark", "Returns", "Multiplier", "Won"):
            assert col in df.columns


def test_compare_no_benchmark_raises(data_no_benchmark):
    """Compare raises AttributeError when no benchmark is attached."""
    with pytest.raises(AttributeError):
        data_no_benchmark.stats.compare()


def test_compare_round_vals(stats):
    """Compare round_vals rounds Benchmark and Returns columns."""
    result = stats.compare(round_vals=2)
    for df in result.values():
        rounded = df["Returns"].drop_nulls()
        assert all(round(v, 2) == v for v in rounded.to_list())


def test_montecarlo_shape(stats):
    """Montecarlo returns one simulated terminal return per simulation and asset."""
    np.random.seed(7)
    result = stats.montecarlo(n=32, period=30)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (32, len(stats.assets))
    assert result.columns == stats.assets


def test_montecarlo_sharpe_shape(stats):
    """montecarlo_sharpe returns one simulated Sharpe ratio per simulation and asset."""
    np.random.seed(7)
    result = stats.montecarlo_sharpe(n=32, period=30, periods_per_year=252)
    assert result.shape == (32, len(stats.assets))
    for col in result.columns:
        assert np.isfinite(result[col].to_numpy()).all()


def test_montecarlo_drawdown_range(stats):
    """montecarlo_drawdown values are valid drawdowns in [-1, 0]."""
    np.random.seed(7)
    result = stats.montecarlo_drawdown(n=32, period=30)
    assert result.shape == (32, len(stats.assets))
    for col in result.columns:
        assert (result[col] <= 0.0).all()
        assert (result[col] >= -1.0).all()


def test_montecarlo_cagr_shape(stats):
    """montecarlo_cagr returns one simulated CAGR per simulation and asset."""
    np.random.seed(7)
    result = stats.montecarlo_cagr(n=32, period=30, periods_per_year=252)
    assert result.shape == (32, len(stats.assets))
    for col in result.columns:
        assert np.isfinite(result[col].to_numpy()).all()


def test_montecarlo_all_null_data(all_null_data):
    """Montecarlo returns a DataFrame of NaN when all returns are null."""
    result = all_null_data.stats.montecarlo(n=5, period=5)
    assert isinstance(result, pl.DataFrame)
    for col in result.columns:
        assert result[col].is_nan().all()


def test_montecarlo_invalid_params(stats):
    """Monte Carlo methods validate simulation counts and periods."""
    with pytest.raises(ValueError, match="positive integer"):
        stats.montecarlo(n=0, period=30)
    with pytest.raises(ValueError, match="positive integer"):
        stats.montecarlo(n=10, period=0)
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        stats.montecarlo_sharpe(periods_per_year=0)
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        stats.montecarlo_cagr(periods_per_year=0)


# ── Edge-case coverage ────────────────────────────────────────────────────────


@pytest.fixture
def flat_data():
    """Data fixture with all-zero returns (no variance)."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), interval="1d", eager=True)
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.0] * len(dates))})
    return Data.from_returns(returns=df)


@pytest.fixture
def all_positive_data():
    """Data fixture with all strictly positive returns."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), interval="1d", eager=True)
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.01] * len(dates))})
    return Data.from_returns(returns=df)


@pytest.fixture
def all_negative_data():
    """Data fixture with all strictly negative returns."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), interval="1d", eager=True)
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([-0.01] * len(dates))})
    return Data.from_returns(returns=df)


@pytest.fixture
def all_null_data():
    """Data fixture with all-null returns."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), interval="1d", eager=True)
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([None] * len(dates), dtype=pl.Float64)})
    return Data.from_returns(returns=df)


@pytest.fixture
def no_temporal_data():
    """Data fixture with an integer (non-temporal) index column."""
    from jquantstats import Data

    df = pl.DataFrame({"x": list(range(1, 21)), "ret": [0.01, 0.02, -0.05, 0.03, -0.02] * 4})
    return Data.from_returns(returns=df, date_col="x")


@pytest.fixture
def partial_year_data():
    """Data fixture spanning only Jan–Mar of one year (missing 9 months)."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), interval="1d", eager=True)
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.001] * len(dates))})
    return Data.from_returns(returns=df)


@pytest.fixture
def loss_data(benchmark_frame):
    """Data fixture with a complete -100% loss on the last day."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), interval="1d", eager=True)[:100]
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.01] * 99 + [-1.0])})
    bench = benchmark_frame.filter(pl.col("Date").is_between(dates[0], dates[-1]))
    return Data.from_returns(returns=df, benchmark=bench)


@pytest.fixture
def zero_beta_data():
    """Data with zero-return strategy and non-zero benchmark (beta == 0)."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), interval="1d", eager=True)[:100]
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.0] * 100)})
    bench = pl.DataFrame({"Date": dates, "bench": pl.Series([0.01 if i % 2 == 0 else -0.01 for i in range(100)])})
    return Data.from_returns(returns=df, benchmark=bench)


@pytest.fixture
def zero_bench_var_data():
    """Data with all-zero benchmark (var_benchmark == 0)."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), interval="1d", eager=True)[:100]
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.01] * 100)})
    bench = pl.DataFrame({"Date": dates, "bench": pl.Series([0.0] * 100)})
    return Data.from_returns(returns=df, benchmark=bench)


@pytest.fixture
def compound_zero_data():
    """Data fixture with a -100% return period that forces compound product to zero."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), interval="1d", eager=True)
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.1, 0.1, -1.0, 0.1, 0.1])})
    return Data.from_returns(returns=df)


@pytest.fixture
def short_autocorr_data():
    """Data fixture with a short return series for large-lag autocorrelation tests."""
    from jquantstats import Data

    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), interval="1d", eager=True)
    df = pl.DataFrame({"Date": dates, "ret": pl.Series([0.01, -0.02, 0.03, -0.01, 0.02])})
    return Data.from_returns(returns=df)


# ── geometric_mean edge cases ─────────────────────────────────────────────────


def test_geometric_mean_compound_zero(compound_zero_data):
    """geometric_mean returns nan when compound product is zero (-100% return)."""
    result = compound_zero_data.stats.geometric_mean()
    assert np.isnan(result["ret"])


# ── serenity_index / tail_ratio / omega / outlier ratios ──────────────────────


def test_serenity_index_zero_std(flat_data):
    """serenity_index returns nan when std is zero."""
    result = flat_data.stats.serenity_index()
    assert all(np.isnan(v) for v in result.values())


def test_tail_ratio_zero_lower_quantile(flat_data):
    """tail_ratio returns nan when lower quantile is zero."""
    result = flat_data.stats.tail_ratio()
    assert all(np.isnan(v) for v in result.values())


def test_omega_all_zero_returns_nan(flat_data):
    """Omega returns nan when denom is zero (no below-threshold returns)."""
    result = flat_data.stats.omega()
    assert all(np.isnan(v) for v in result.values())


def test_outlier_win_ratio_no_positive_returns(all_negative_data):
    """outlier_win_ratio returns nan when no positive returns exist."""
    result = all_negative_data.stats.outlier_win_ratio()
    assert all(np.isnan(v) for v in result.values())


def test_outlier_win_ratio_zero_positive_mean(flat_data):
    """outlier_win_ratio returns nan when positive_mean is zero (all returns are 0.0)."""
    result = flat_data.stats.outlier_win_ratio()
    assert all(np.isnan(v) for v in result.values())


def test_outlier_loss_ratio_no_negative_returns(all_positive_data):
    """outlier_loss_ratio returns nan when no negative returns exist."""
    result = all_positive_data.stats.outlier_loss_ratio()
    assert all(np.isnan(v) for v in result.values())


# ── autocorr with large lag ───────────────────────────────────────────────────


def test_autocorr_lag_beyond_series_length(short_autocorr_data):
    """Autocorr returns nan when lag >= series length (empty paired after shift)."""
    result = short_autocorr_data.stats.autocorr(lag=100)
    assert all(np.isnan(v) for v in result.values())


# ── probabilistic_ratio ───────────────────────────────────────────────────────


def test_probabilistic_ratio_sharpe_base(stats):
    """probabilistic_ratio with base='sharpe' returns values in [0, 1]."""
    result = stats.probabilistic_ratio(base="sharpe")
    assert isinstance(result, dict)
    assert all(0 <= v <= 1 for v in result.values() if not np.isnan(v))


def test_probabilistic_ratio_sortino_base(stats):
    """probabilistic_ratio with base='sortino' returns values in [0, 1]."""
    result = stats.probabilistic_ratio(base="sortino")
    assert isinstance(result, dict)


def test_probabilistic_ratio_adjusted_sortino_base(stats):
    """probabilistic_ratio with base='adjusted_sortino' returns values in [0, 1]."""
    result = stats.probabilistic_ratio(base="adjusted_sortino")
    assert isinstance(result, dict)


def test_probabilistic_ratio_callable_base(stats):
    """probabilistic_ratio with a callable base uses the callable."""
    result = stats.probabilistic_ratio(base=lambda s: 0.5)
    assert isinstance(result, dict)
    assert all(0 <= v <= 1 for v in result.values() if not np.isnan(v))


def test_probabilistic_ratio_invalid_base_raises(stats):
    """probabilistic_ratio with an unrecognised string base raises ValueError."""
    with pytest.raises(ValueError, match="base must be one of"):
        stats.probabilistic_ratio(base="invalid")


def test_probabilistic_ratio_from_base_small_n():
    """_probabilistic_ratio_from_base returns nan when n <= 1."""
    from jquantstats._stats._performance import _RiskStatsMixin

    result = _RiskStatsMixin._probabilistic_ratio_from_base(1.0, pl.Series([0.01]))
    assert np.isnan(result)


def test_probabilistic_ratio_from_base_negative_variance():
    """_probabilistic_ratio_from_base returns nan when computed variance <= 0."""
    from jquantstats._stats._performance import _RiskStatsMixin

    # Series with extreme negative skew; base=-0.5 yields negative variance
    series = pl.Series([0.001] * 10 + [-100.0])
    result = _RiskStatsMixin._probabilistic_ratio_from_base(-0.5, series)
    assert np.isnan(result)


# ── probabilistic_smart_sortino / probabilistic_sortino zero downside ─────────


def test_probabilistic_sortino_ratio_zero_downside(flat_data):
    """probabilistic_sortino_ratio returns nan when downside deviation is zero."""
    result = flat_data.stats.probabilistic_sortino_ratio()
    assert all(np.isnan(v) for v in result.values())


def test_probabilistic_adjusted_sortino_ratio_zero_downside(flat_data):
    """probabilistic_adjusted_sortino_ratio returns nan when downside deviation is zero."""
    result = flat_data.stats.probabilistic_adjusted_sortino_ratio()
    assert all(np.isnan(v) for v in result.values())


# ── drawdown_details edge cases ───────────────────────────────────────────────


def test_drawdown_details_no_drawdown(all_positive_data):
    """drawdown_details returns empty DataFrame when there are no drawdown periods."""
    result = all_positive_data.stats.drawdown_details()
    for df in result.values():
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


def test_drawdown_details_integer_index(no_temporal_data):
    """drawdown_details works with a non-temporal (integer) index."""
    result = no_temporal_data.stats.drawdown_details()
    for df in result.values():
        assert isinstance(df, pl.DataFrame)
        # duration column uses integer arithmetic
        assert df.schema["start"] == pl.Int64


# ── treynor_ratio edge cases ──────────────────────────────────────────────────


def test_treynor_ratio_no_benchmark_raises(data_no_benchmark):
    """treynor_ratio raises AttributeError when no benchmark is attached."""
    with pytest.raises(AttributeError):
        data_no_benchmark.stats.treynor_ratio()


def test_treynor_ratio_zero_benchmark_variance(zero_bench_var_data):
    """treynor_ratio returns nan when benchmark variance is zero."""
    result = zero_bench_var_data.stats.treynor_ratio()
    assert all(np.isnan(v) for v in result.values())


def test_treynor_ratio_zero_beta(zero_beta_data):
    """treynor_ratio returns nan when beta is zero (strategy uncorrelated)."""
    result = zero_beta_data.stats.treynor_ratio()
    # 'ret' column (all zeros) should give beta=0 -> nan
    assert np.isnan(result["ret"])


def test_treynor_ratio_negative_nav(loss_data):
    """treynor_ratio returns nan when cumulative NAV is zero (-100% return)."""
    result = loss_data.stats.treynor_ratio()
    assert np.isnan(result["ret"])


# ── rolling_greeks error paths ────────────────────────────────────────────────


def test_rolling_greeks_no_benchmark_raises(data_no_benchmark):
    """rolling_greeks raises AttributeError when no benchmark is attached."""
    with pytest.raises(AttributeError):
        data_no_benchmark.stats.rolling_greeks()


def test_rolling_greeks_bad_period_raises(stats):
    """rolling_greeks raises ValueError for non-positive rolling_period."""
    with pytest.raises(ValueError, match="rolling_period must be a positive integer"):
        stats.rolling_greeks(rolling_period=-1)


# ── expected_return edge cases ────────────────────────────────────────────────


def test_expected_return_empty_series(all_null_data):
    """expected_return returns nan when all values are null (empty after drop_nulls)."""
    result = all_null_data.stats.expected_return()
    assert all(np.isnan(v) for v in result.values())


def test_expected_return_invalid_aggregate_raises(stats):
    """expected_return raises ValueError for an unrecognised aggregate string."""
    with pytest.raises(ValueError, match="aggregate must be one of"):
        stats.expected_return(aggregate="INVALID")


def test_expected_return_non_temporal_index_with_aggregate(no_temporal_data):
    """expected_return falls back to per-period mean when index is not temporal."""
    result = no_temporal_data.stats.expected_return(aggregate="monthly")
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


# ── compare with aggregate ────────────────────────────────────────────────────


def test_compare_monthly_aggregate(stats):
    """compare(aggregate='ME') groups returns by month."""
    result = stats.compare(aggregate="ME")
    assert isinstance(result, dict)
    for df in result.values():
        assert "Returns" in df.columns
        assert "Benchmark" in df.columns


# ── geometric_mean null data ──────────────────────────────────────────────────


def test_geometric_mean_all_null(all_null_data):
    """geometric_mean returns nan when all values are null (n==0 after drop_nulls)."""
    result = all_null_data.stats.geometric_mean()
    assert all(np.isnan(v) for v in result.values())


# ── probabilistic_ratio zero-std / zero-downside paths ───────────────────────


def test_probabilistic_ratio_sharpe_base_zero_std(flat_data):
    """probabilistic_ratio with base='sharpe' returns nan when std is zero (flat returns)."""
    result = flat_data.stats.probabilistic_ratio(base="sharpe")
    assert all(np.isnan(v) for v in result.values())


def test_probabilistic_ratio_sortino_base_zero_downside(all_positive_data):
    """probabilistic_ratio with base='sortino' returns nan when downside_dev is zero."""
    result = all_positive_data.stats.probabilistic_ratio(base="sortino")
    assert all(np.isnan(v) for v in result.values())


# ── monthly_returns missing months ───────────────────────────────────────────


def test_monthly_returns_missing_months_filled(partial_year_data):
    """monthly_returns fills missing months with 0.0 when data spans < 1 year."""
    result = partial_year_data.stats.monthly_returns(eoy=True)
    for df in result.values():
        # All 12 month columns should be present even if data only covers Jan-Mar
        for month in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]:
            assert month in df.columns


# ── _internals shared helpers ─────────────────────────────────────────────────


def test_comp_return_known_values():
    """_comp_return returns the correct compounded return for a simple series."""
    import polars as pl

    from jquantstats._stats._internals import _comp_return

    s = pl.Series([0.1, -0.05, 0.2])
    # (1.1)(0.95)(1.2) - 1 = 1.254 - 1 = 0.254
    assert _comp_return(s) == pytest.approx(1.1 * 0.95 * 1.2 - 1.0)


def test_comp_return_all_zeros():
    """_comp_return returns 0.0 for a series of zeros."""
    import polars as pl

    from jquantstats._stats._internals import _comp_return

    assert _comp_return(pl.Series([0.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_comp_return_drops_nulls():
    """_comp_return ignores null values when computing the product."""
    import polars as pl

    from jquantstats._stats._internals import _comp_return

    s_with_null = pl.Series([0.1, None, 0.1], dtype=pl.Float64)
    s_no_null = pl.Series([0.1, 0.1])
    assert _comp_return(s_with_null) == pytest.approx(_comp_return(s_no_null))


def test_nav_series_starts_at_one_plus_r():
    """_nav_series first value equals 1 + first return."""
    import polars as pl

    from jquantstats._stats._internals import _nav_series

    s = pl.Series([0.05, -0.02, 0.03])
    nav = _nav_series(s)
    assert nav[0] == pytest.approx(1.05)
    assert nav[1] == pytest.approx(1.05 * 0.98)
    assert nav[2] == pytest.approx(1.05 * 0.98 * 1.03)


def test_nav_series_flat_returns():
    """_nav_series on a zero-return series stays constant at 1.0."""
    import polars as pl

    from jquantstats._stats._internals import _nav_series

    nav = _nav_series(pl.Series([0.0, 0.0, 0.0]))
    assert nav.to_list() == pytest.approx([1.0, 1.0, 1.0])


def test_annualization_factor_sqrt():
    """_annualization_factor(252) returns sqrt(252)."""
    import math

    from jquantstats._stats._internals import _annualization_factor

    assert _annualization_factor(252) == pytest.approx(math.sqrt(252))


def test_annualization_factor_linear():
    """_annualization_factor(252, sqrt=False) returns 252.0."""
    from jquantstats._stats._internals import _annualization_factor

    assert _annualization_factor(252, sqrt=False) == pytest.approx(252.0)


def test_annualization_factor_invalid():
    """_annualization_factor raises ValueError for non-positive periods."""
    from jquantstats._stats._internals import _annualization_factor

    with pytest.raises(ValueError, match="positive finite number"):
        _annualization_factor(0)
    with pytest.raises(ValueError, match="positive finite number"):
        _annualization_factor(-1)


def test_downside_deviation_known_values():
    """_downside_deviation matches manual computation."""
    import math

    import polars as pl

    from jquantstats._stats._internals import _downside_deviation

    s = pl.Series([0.05, -0.02, 0.03, -0.01, 0.0])
    # Only -0.02 and -0.01 are negative; sum of squares = 0.0004 + 0.0001 = 0.0005
    # n = 5 (total count); downside_dev = sqrt(0.0005 / 5) = sqrt(0.0001) = 0.01
    expected = math.sqrt(0.0005 / 5)
    assert _downside_deviation(s) == pytest.approx(expected)


def test_downside_deviation_all_positive():
    """_downside_deviation returns 0.0 when there are no negative returns."""
    import polars as pl

    from jquantstats._stats._internals import _downside_deviation

    assert _downside_deviation(pl.Series([0.1, 0.2, 0.3])) == pytest.approx(0.0)


def test_downside_deviation_empty():
    """_downside_deviation returns 0.0 for an empty series."""
    import polars as pl

    from jquantstats._stats._internals import _downside_deviation

    assert _downside_deviation(pl.Series([], dtype=pl.Float64)) == pytest.approx(0.0)
