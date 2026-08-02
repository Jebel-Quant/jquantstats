"""The tidy `summary` table and its calendar-year breakdown.

Split out of `_reporting.py`. These are the aggregating metrics: they call
across every other mixin rather than computing anything themselves, which is
why they carry the largest block of cross-mixin type stubs in the package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import polars as pl

if TYPE_CHECKING:
    from ..data import Data


class _SummaryStatsMixin:
    """Mixin providing the `summary` table and `annual_breakdown`.

    Cross-mixin dependencies:
        - _BasicStatsMixin: avg_return, avg_win, avg_loss, win_rate, profit_factor,
          payoff_ratio, best, worst, volatility, skew, kurtosis, value_at_risk,
          conditional_value_at_risk
        - _RiskStatsMixin: sharpe
        - _DrawdownMixin: max_drawdown
        - _ReportingStatsMixin: monthly_win_rate, avg_drawdown,
          max_drawdown_duration, calmar, recovery_factor
    """

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

        def avg_return(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def avg_win(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def avg_loss(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def win_rate(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def profit_factor(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def payoff_ratio(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def best(self) -> dict[str, float | None]:
            """Defined on _BasicStatsMixin."""

        def worst(self) -> dict[str, float | None]:
            """Defined on _BasicStatsMixin."""

        def volatility(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def sharpe(self) -> dict[str, float]:
            """Defined on _RiskStatsMixin."""

        def skew(self) -> dict[str, int | float | None]:
            """Defined on _BasicStatsMixin."""

        def kurtosis(self) -> dict[str, int | float | None]:
            """Defined on _BasicStatsMixin."""

        def value_at_risk(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def conditional_value_at_risk(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def max_drawdown(self) -> dict[str, float]:
            """Defined on _DrawdownMixin."""

        def monthly_win_rate(self) -> dict[str, float]:
            """Defined on _ReportingStatsMixin."""

        def avg_drawdown(self) -> dict[str, float]:
            """Defined on _ReportingStatsMixin."""

        def max_drawdown_duration(self) -> dict[str, float | int | None]:
            """Defined on _ReportingStatsMixin."""

        def calmar(self) -> dict[str, float]:
            """Defined on _ReportingStatsMixin."""

        def recovery_factor(self) -> dict[str, float]:
            """Defined on _ReportingStatsMixin."""

    def annual_breakdown(self) -> pl.DataFrame:
        """Summary statistics broken down by calendar year.

        Groups the data by calendar year using the date index, computes a
        full `summary` for each year, and stacks the results with an
        additional ``year`` column.

        Returns:
            pl.DataFrame: Columns ``year``, ``metric``, one per asset, sorted
            by ``year``.

        Raises:
            ValueError: If the data has no date index.
        """
        all_df = self.all
        date_col_name = self._data.date_col[0] if self._data.date_col else None
        has_temporal = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()

        if not has_temporal:
            return self._annual_breakdown_integer(all_df)
        if date_col_name is None:  # unreachable: has_temporal guarantees non-None  # pragma: no cover
            return pl.DataFrame()  # pragma: no cover
        return self._annual_breakdown_temporal(all_df, date_col_name)

    def _summary_frame(self, sub_all: pl.DataFrame, index_cols: list[str], label: int) -> pl.DataFrame:
        """Compute a `summary` for one sub-period and tag it with a ``year`` label.

        Args:
            sub_all: The combined (index + returns + benchmark) rows for the period.
            index_cols: Column name(s) to use as the sub-period's date index.
            label: Value written to the ``year`` column (calendar year or chunk ordinal).

        Returns:
            The summary DataFrame with an added ``year`` column.
        """
        # Construct the sub-period Data via type(self._data) rather than importing
        # the concrete class: a lazy `from ..data import Data` would put the upper
        # layer back into this subpackage's import graph, which is exactly the
        # coupling _protocol.py exists to prevent. Mirrors the type(self) call below.
        data_factory = cast(Any, type(self._data))
        sub_returns = sub_all.select(self._data.returns.columns)
        sub_benchmark = sub_all.select(self._data.benchmark.columns) if self._data.benchmark is not None else None
        sub_data = data_factory(returns=sub_returns, index=sub_all.select(index_cols), benchmark=sub_benchmark)
        summary: pl.DataFrame = cast(Any, type(self))(sub_data).summary()
        return summary.with_columns(pl.lit(label).alias("year"))

    @staticmethod
    def _order_breakdown(result: pl.DataFrame) -> pl.DataFrame:
        """Reorder breakdown columns so ``year`` and ``metric`` lead."""
        ordered = ["year", "metric", *[c for c in result.columns if c not in ("year", "metric")]]
        return result.select(ordered)

    def _annual_breakdown_integer(self, all_df: pl.DataFrame) -> pl.DataFrame:
        """Break down by fixed row chunks (~one year each) for an integer index."""
        chunk = round(self._data._periods_per_year)
        total = all_df.height
        frames: list[pl.DataFrame] = []
        for i, start in enumerate(range(0, total, chunk), start=1):
            chunk_all = all_df.slice(start, chunk)
            if chunk_all.height < max(5, chunk // 4):
                continue
            frames.append(self._summary_frame(chunk_all, self._data.date_col, i))
        if not frames:
            return pl.DataFrame()
        return self._order_breakdown(pl.concat(frames))

    def _annual_breakdown_temporal(self, all_df: pl.DataFrame, date_col_name: str) -> pl.DataFrame:
        """Break down by calendar year for a temporal index."""
        years = all_df[date_col_name].dt.year().unique().sort().to_list()
        frames: list[pl.DataFrame] = []
        for year in years:
            year_all = all_df.filter(pl.col(date_col_name).dt.year() == year)
            if year_all.height < 2:
                continue
            frames.append(self._summary_frame(year_all, [date_col_name], year))
        if not frames:
            asset_cols = list(self._data.returns.columns)
            schema: dict[str, type[pl.DataType]] = {
                "year": pl.Int32,
                "metric": pl.String,
                **dict.fromkeys(asset_cols, pl.Float64),
            }
            return pl.DataFrame(schema=schema)
        return self._order_breakdown(pl.concat(frames))

    def summary(self) -> pl.DataFrame:
        """Summary statistics for each asset as a tidy DataFrame.

        Each row is one metric; each column beyond ``metric`` is one asset.

        Returns:
            pl.DataFrame: A DataFrame with a ``metric`` column followed by one
            column per asset.

        Returns NaN when:
            Cells are ``float("nan")`` when the underlying metric is unavailable
            for the data (e.g. no temporal index or no benchmark).
        """
        assets = [col for col, _ in self._data.items()]

        def _safe(fn: Any) -> dict[str, Any]:
            """Call *fn()* and return its result; return NaN for each asset on any exception."""
            try:
                result: dict[str, Any] = fn()
            except Exception:
                return dict.fromkeys(assets, float("nan"))
            return result

        metrics: dict[str, dict[str, Any]] = {
            "avg_return": _safe(self.avg_return),
            "avg_win": _safe(self.avg_win),
            "avg_loss": _safe(self.avg_loss),
            "win_rate": _safe(self.win_rate),
            "profit_factor": _safe(self.profit_factor),
            "payoff_ratio": _safe(self.payoff_ratio),
            "monthly_win_rate": _safe(self.monthly_win_rate),
            "best": _safe(self.best),
            "worst": _safe(self.worst),
            "volatility": _safe(self.volatility),
            "sharpe": _safe(self.sharpe),
            "skew": _safe(self.skew),
            "kurtosis": _safe(self.kurtosis),
            "value_at_risk": _safe(self.value_at_risk),
            "conditional_value_at_risk": _safe(self.conditional_value_at_risk),
            "max_drawdown": _safe(self.max_drawdown),
            "avg_drawdown": _safe(self.avg_drawdown),
            "max_drawdown_duration": _safe(self.max_drawdown_duration),
            "calmar": _safe(self.calmar),
            "recovery_factor": _safe(self.recovery_factor),
        }

        rows: list[dict[str, Any]] = [
            {"metric": name, **{asset: values.get(asset) for asset in assets}} for name, values in metrics.items()
        ]
        return pl.DataFrame(rows)
