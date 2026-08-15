"""Portfolio analytics class for quant finance.

This module provides `Portfolio`, a frozen dataclass that stores the
raw portfolio inputs (prices, cash positions, AUM) and exposes both the
derived data series and the full analytics / visualisation suite.

The class is composed from focused mixin modules:

- `PortfolioNavMixin` — NAV & returns chain
- `PortfolioAttributionMixin` — tilt/timing attribution
- `PortfolioTurnoverMixin` — turnover analytics
- `PortfolioCostMixin` — cost analysis
- `PortfolioTransformMixin` — range/lag/smoothing transforms & correlation
- `PortfolioConstructorMixin` — `from_risk_position` / `from_position` factories

Public API is unchanged:

- Derived data series — `profits`, `profit`, `nav_accumulated`,
  `returns`, `monthly`, `nav_compounded`, `highwater`,
  `drawdown`, `all`
- Lazy composition accessors — `stats`, `plots`, `report`
- Portfolio transforms — `truncate`, `lag`, `smoothed_holding`
- Attribution — `tilt`, `timing`, `tilt_timing_decomp`
- Turnover analysis — `turnover`, `turnover_weekly`, `turnover_summary`
- Cost analysis — `cost_adjusted_returns`, `trading_cost_impact`, `deduct_management_fee`
- Utility — `correlation`
"""

import dataclasses
from datetime import date, datetime
from typing import Self, cast

import polars as pl

from ._cache import cached_in_slot
from ._cost_model import CostModel
from ._plots import PortfolioPlots
from ._portfolio_attribution import PortfolioAttributionMixin
from ._portfolio_constructors import PortfolioConstructorMixin, _evaluate_position_expr
from ._portfolio_cost import PortfolioCostMixin
from ._portfolio_nav import PortfolioNavMixin
from ._portfolio_transform import PortfolioTransformMixin
from ._portfolio_turnover import PortfolioTurnoverMixin
from ._portfolio_units import PortfolioUnitsMixin
from ._reports import Report
from ._stats import Stats as Stats
from ._utils import PortfolioUtils as PortfolioUtils
from .data import Data as Data
from .exceptions import (
    InvalidCashPositionTypeError,
    InvalidPricesTypeError,
    NonPositiveAumError,
    RowCountMismatchError,
    UncleanSeriesError,
)

# Slot fields used as lazy caches; __post_init__ initialises each to None and
# `cached_in_slot` fills them on first property access.
_CACHE_SLOTS = (
    "_data_bridge",
    "_stats_cache",
    "_plots_cache",
    "_report_cache",
    "_utils_cache",
    "_profits_cache",
    "_returns_cache",
    "_tilt_cache",
    "_turnover_cache",
)

# The canonical name of the date column throughout the Portfolio internals. Every
# _portfolio_* mixin tests for this exact name to decide whether it has a temporal
# axis, so inputs are normalised to it once at construction rather than each site
# having to cope with an arbitrary caller-chosen name.
_DATE_COLUMN = "date"


def _normalise_date_column(frame: pl.DataFrame) -> pl.DataFrame:
    """Rename *frame*'s temporal column to ``'date'``.

    The Portfolio internals identify the date axis by name, so a frame whose
    dates live under any other label (``'Date'``, ``'timestamp'``, …) would be
    treated as having no temporal axis at all — silently falling back to a
    positional index and a default ``periods_per_year``.  Renaming here makes
    that impossible.

    A frame that already has a ``'date'`` column is returned untouched, so this is
    idempotent and cheap on the internal rebuild paths (`lag`, `truncate`,
    `smoothed_holding`) where the input is already canonical.  When several
    temporal columns are present the first one wins, matching the
    leading-date-column convention used throughout.

    Args:
        frame: A price or cash-position frame, with or without dates.

    Returns:
        *frame* with its date column named ``'date'``, or *frame* unchanged
        when it already has one or has no temporal column to rename.
    """
    if _DATE_COLUMN in frame.columns:
        return frame
    temporal = [name for name, dtype in frame.schema.items() if dtype.is_temporal()]
    if not temporal:
        return frame
    return frame.rename({temporal[0]: _DATE_COLUMN})


@dataclasses.dataclass(frozen=True, slots=True)
class Portfolio(
    PortfolioNavMixin,
    PortfolioAttributionMixin,
    PortfolioTurnoverMixin,
    PortfolioCostMixin,
    PortfolioTransformMixin,
    PortfolioUnitsMixin,
    PortfolioConstructorMixin,
):
    """Portfolio analytics class for quant finance.

    Stores the three raw inputs — cash positions, prices, and AUM — and
    exposes the standard derived data series, analytics facades, transforms,
    and attribution tools.

    Derived data series:

    - `profits` — per-asset daily cash P&L
    - `profit` — aggregate daily portfolio profit
    - `nav_accumulated` — cumulative additive NAV
    - `nav_compounded` — compounded NAV
    - `returns` — daily returns (profit / AUM)
    - `monthly` — monthly compounded returns
    - `highwater` — running high-water mark
    - `drawdown` — drawdown from high-water mark
    - `all` — merged view of all derived series

    - Lazy composition accessors: `stats`, `plots`, `report`
    - Portfolio transforms: `truncate`, `lag`,
      `smoothed_holding`
    - Attribution: `tilt`, `timing`, `tilt_timing_decomp`
    - Turnover: `turnover`, `turnover_weekly`,
      `turnover_summary`
    - Share-count view: `units`, `equity`, `trades_units`,
      `trades_currency`, `weights` — all derived from cash positions and
      prices, so they are available however the portfolio was constructed
    - Cost analysis: `cost_adjusted_returns`,
      `trading_cost_impact`, `deduct_management_fee`
    - Utility: `correlation`

    Attributes:
        cashposition: Polars DataFrame of positions per asset over time.  Any
            temporal column is present as ``'date'`` — see *Date column* below.
        prices: Polars DataFrame of prices per asset over time.  Any temporal
            column is present as ``'date'`` — see *Date column* below.
        aum: Assets under management used as base NAV offset.

    Analytics facades
    -----------------
    - ``.stats``   : delegates to the legacy ``Stats`` pipeline via ``.data``; all 50+ metrics available.
    - ``.plots``   : portfolio-specific ``Plots``; NAV overlays, lead-lag IR, rolling Sharpe/vol, heatmaps.
    - ``.report``  : HTML ``Report``; self-contained portfolio performance report.
    - ``.data``    : bridge to the legacy ``Data`` / ``Stats`` / ``DataPlots`` pipeline.

    ``.plots`` and ``.report`` are intentionally *not* delegated to the legacy path: the legacy
    path operates on a bare returns series, while the analytics path has access to raw prices,
    positions, and AUM for richer portfolio-specific visualisations.

    Cost models
    -----------
    Two independent cost models are provided. They are not interchangeable:

    **Model A — position-delta (stateful, set at construction):**
        ``cost_per_unit: float``  — one-way cost per unit of position change (e.g. 0.01 per share).
        Used by ``.position_delta_costs`` and ``.net_cost_nav``.
        Best for: equity portfolios where cost scales with shares traded.

    **Model B — turnover-bps (stateless, passed at call time):**
        ``cost_bps: float``  — one-way cost in basis points of AUM turnover (e.g. 5 bps).
        Used by ``.cost_adjusted_returns(cost_bps)`` and ``.trading_cost_impact(max_bps)``.
        Best for: macro / fund-of-funds portfolios where cost scales with notional traded.

    **Management fee (flat annual, set at construction or passed at call time):**
        ``annual_fee: float``  — flat annual management fee as a fraction of AUM (e.g. 0.0085 for 85 bps p.a.).
        Used by ``.deduct_management_fee(annual_fee)``.
        The fee accrues pro-rata per calendar day (``annual_fee * days / 365``), so weekends and
        holidays are charged to the next trading day and the deduction sums to ``annual_fee`` over a
        full year.  Cost and fee deductions compose linearly in any order.

    To sweep a range of cost assumptions use ``trading_cost_impact(max_bps=20)`` (Model B).
    To compute a net-NAV curve set ``cost_per_unit`` at construction and read ``.net_cost_nav`` (Model A).

    Date column
    -----------
    The date axis is identified internally by the name ``date``, so a temporal column
    (``pl.Date`` or ``pl.Datetime``) under any other label — ``'Date'``, ``'timestamp'`` —
    is **renamed to ``date`` at construction**.  ``prices`` and ``cashposition`` therefore
    report ``date`` rather than the caller's original name, and every date-dependent
    feature works regardless of what the input column was called.

    Only genuinely temporal columns are normalised: a date column still held as strings
    is left as-is and the portfolio is treated as integer-indexed.  Parse it first
    (``pl.col("Date").str.to_date()``) to get a temporal axis.  When a frame contains
    several temporal columns and none is named ``date``, the first is renamed.

    Most analytics work with or without a date column. The following features require a
    temporal ``date`` column:

    - ``portfolio.plots.correlation_heatmap()``
    - ``portfolio.plots.lead_lag_ir_plot()``
    - ``stats.monthly_win_rate()``      — returns NaN per column when no date is present
    - ``stats.annual_breakdown()``      — raises ``ValueError`` when no date is present
    - ``stats.max_drawdown_duration()`` — returns period count (int) instead of days

    Portfolios without a ``date`` column (integer-indexed) are fully supported for
    NAV, returns, Sharpe, drawdown, cost analytics, and most rolling metrics.

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
        >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
        >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
        >>> pf.assets
        ['A']
    """

    cashposition: pl.DataFrame
    prices: pl.DataFrame
    aum: float
    cost_per_unit: float = 0.0
    cost_bps: float = 0.0
    annual_fee: float = 0.0

    # ── Internal cache fields ─────────────────────────────────────────────────
    # All cache fields are initialised to ``None`` in ``__post_init__`` via
    # ``object.__setattr__`` (required for frozen dataclasses) and populated
    # lazily on first property access.
    #
    # Lifecycle:
    #   - Initialised: ``__post_init__`` sets every field to ``None``.
    #   - Populated: each property computes its value on the first call and
    #     writes it back via ``object.__setattr__``.
    #   - Invalidation: not required — ``Portfolio`` is a *frozen* dataclass,
    #     so its inputs never change and all derived values remain valid for the
    #     lifetime of the instance.
    _data_bridge: "Data | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _stats_cache: "Stats | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _plots_cache: "PortfolioPlots | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _report_cache: "Report | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _utils_cache: "PortfolioUtils | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _profits_cache: "pl.DataFrame | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _returns_cache: "pl.DataFrame | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _tilt_cache: "Portfolio | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _turnover_cache: "pl.DataFrame | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    @staticmethod
    def _build_data_bridge(ret: pl.DataFrame) -> "Data":
        """Build a `Data` bridge from a returns frame.

        Splits out the ``'date'`` column (if present) into an index and passes
        the remaining numeric columns as returns.  Used internally to populate
        ``_data_bridge`` at construction time so the ``data`` property is O(1).

        The name is matched literally, which is safe because ``__post_init__``
        normalises the inputs: the positional-index fallback below is reached
        only by a portfolio that genuinely has no temporal column, never by one
        whose dates simply arrived under a different name.

        Args:
            ret: Returns DataFrame, optionally with a leading ``'date'`` column.

        Returns:
            A `Data` instance backed by *ret*.
        """
        returns_only = ret.select("returns")
        if "date" in ret.columns:
            return Data(returns=returns_only, index=ret.select("date"))
        return Data(returns=returns_only, index=pl.DataFrame({"index": list(range(ret.height))}))

    def __post_init__(self) -> None:
        """Validate input types, shapes, and parameters, and normalise the date column."""
        if not isinstance(self.prices, pl.DataFrame):
            raise InvalidPricesTypeError(type(self.prices).__name__)
        if not isinstance(self.cashposition, pl.DataFrame):
            raise InvalidCashPositionTypeError(type(self.cashposition).__name__)
        # Canonicalise the date axis before anything downstream looks for it; the
        # mixins match ``'date'`` by name, so this must happen on every
        # construction path, including a direct ``Portfolio(...)`` call.
        object.__setattr__(self, "prices", _normalise_date_column(self.prices))
        object.__setattr__(self, "cashposition", _normalise_date_column(self.cashposition))
        if self.cashposition.shape[0] != self.prices.shape[0]:
            raise RowCountMismatchError(self.prices.shape[0], self.cashposition.shape[0])
        if self.aum <= 0.0:
            raise NonPositiveAumError(self.aum)
        for slot in _CACHE_SLOTS:
            object.__setattr__(self, slot, None)

    def _date_range(self) -> tuple[int, date | datetime | None, date | datetime | None]:
        """Return (rows, start, end) for the portfolio's returns series.

        ``start`` and ``end`` are ``None`` when there is no ``'date'`` column.
        """
        ret = self.returns
        rows = ret.height
        if "date" in ret.columns:
            return rows, cast(date | None, ret["date"].min()), cast(date | None, ret["date"].max())
        return rows, None, None

    @property
    def cost_model(self) -> CostModel:
        """Return the active cost model as a `CostModel` instance.

        Returns:
            A `CostModel` whose ``cost_per_unit`` and ``cost_bps`` fields
            reflect the values stored on this portfolio.
        """
        return CostModel(cost_per_unit=self.cost_per_unit, cost_bps=self.cost_bps)

    def __repr__(self) -> str:
        """Return a string representation of the Portfolio object."""
        rows, start, end = self._date_range()
        if start is not None:
            return f"Portfolio(assets={self.assets}, rows={rows}, start={start}, end={end})"
        return f"Portfolio(assets={self.assets}, rows={rows})"

    def describe(self) -> pl.DataFrame:
        """Return a tidy summary of shape, date range and asset names.

        Returns:
        -------
        pl.DataFrame
            One row per asset with columns: asset, start, end, rows.

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
            >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> df = pf.describe()
            >>> list(df.columns)
            ['asset', 'start', 'end', 'rows']
        """
        rows, start, end = self._date_range()
        return pl.DataFrame(
            {
                "asset": self.assets,
                "start": [start] * len(self.assets),
                "end": [end] * len(self.assets),
                "rows": [rows] * len(self.assets),
            }
        )

    # ── Factory classmethods ──────────────────────────────────────────────────

    @classmethod
    def from_cash_position(
        cls,
        prices: pl.DataFrame,
        cash_position: pl.DataFrame | pl.Expr,
        aum: float,
        cost_per_unit: float = 0.0,
        cost_bps: float = 0.0,
        cost_model: CostModel | None = None,
        annual_fee: float = 0.0,
    ) -> Self:
        """Create a Portfolio directly from cash positions aligned with prices.

        Args:
            prices: Price levels per asset over time.  A temporal column is
                normalised to ``'date'`` at construction, whatever it was named.
            cash_position: Cash exposure per asset over time, either as a
                DataFrame or as a Polars expression evaluated against *prices*.
            aum: Assets under management used as the base NAV offset.
            cost_per_unit: One-way trading cost per unit of position change.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_bps: One-way trading cost in basis points of AUM turnover.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_model: Optional `CostModel`
                instance.  When supplied, its ``cost_per_unit`` and
                ``cost_bps`` values take precedence over the individual
                parameters above.
            annual_fee: Flat annual management fee as a fraction of AUM
                (e.g. 0.0085 for 85 bps p.a.).  Defaults to 0.0 (no fee).
                Used as the default by `deduct_management_fee`.

        Returns:
            A Portfolio instance with the provided cash positions.

        Raises:
            PositionExprColumnError: If *cash_position* is an expression that
                creates columns not present in *prices* (e.g. via ``.alias``);
                such expressions leave the original asset columns untouched,
                silently treating raw prices as positions.
        """
        if isinstance(cash_position, pl.Expr):
            cash_position = _evaluate_position_expr(prices, cash_position, "cash_position")
        if cost_model is not None:
            cost_per_unit = cost_model.cost_per_unit
            cost_bps = cost_model.cost_bps
        return cls(
            prices=prices,
            cashposition=cash_position,
            aum=aum,
            cost_per_unit=cost_per_unit,
            cost_bps=cost_bps,
            annual_fee=annual_fee,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _assert_clean_series(series: pl.Series, name: str = "") -> None:
        """Raise `UncleanSeriesError` if *series* contains nulls or non-finite values.

        Args:
            series: The series to validate.
            name: Optional series name included in the error message.

        Raises:
            UncleanSeriesError: If the series contains null or non-finite values.
        """
        if series.null_count() != 0:
            raise UncleanSeriesError(name, "null")
        if not series.is_finite().all():
            raise UncleanSeriesError(name, "non-finite")

    # ── Core data properties ───────────────────────────────────────────────────

    @property
    def assets(self) -> list[str]:
        """List the asset column names from prices (numeric columns).

        Returns:
            list[str]: Names of numeric columns in prices; typically excludes
            ``'date'``.
        """
        return [c for c in self.prices.columns if self.prices[c].dtype.is_numeric()]

    # ── Lazy composition accessors ─────────────────────────────────────────────

    @property
    @cached_in_slot("_data_bridge")
    def data(self) -> "Data":
        """Build a legacy `Data` object from this portfolio's returns.

        This bridges the two entry points: ``Portfolio`` compiles the NAV curve from
        prices and positions; the returned `Data` object
        gives access to the full legacy analytics pipeline (``data.stats``,
        ``data.plots``, ``data.reports``).

        Returns:
            `Data`: A Data object whose ``returns`` column
            is the portfolio's daily return series and whose ``index`` holds the date
            column (or a synthetic integer index for date-free portfolios).

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
            >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> d = pf.data
            >>> "returns" in d.returns.columns
            True
        """
        return Portfolio._build_data_bridge(self.returns)

    @property
    @cached_in_slot("_stats_cache")
    def stats(self) -> "Stats":
        """Return a Stats object built from the portfolio's daily returns.

        Delegates to the legacy `Stats` pipeline via
        `data`, so all analytics (Sharpe, drawdown, summary, etc.) are
        available through the shared implementation.

        The result is cached after first access so repeated calls are O(1).
        """
        return self.data.stats

    @property
    @cached_in_slot("_plots_cache")
    def plots(self) -> PortfolioPlots:
        """Convenience accessor returning a PortfolioPlots facade for this portfolio.

        Use this to create Plotly visualizations such as snapshots, lagged
        performance curves, and lead/lag IR charts.

        Returns:
            `PortfolioPlots`: Helper object with
            plotting methods.

        The result is cached after first access so repeated calls are O(1).
        """
        return PortfolioPlots(self)

    @property
    @cached_in_slot("_report_cache")
    def report(self) -> Report:
        """Convenience accessor returning a Report facade for this portfolio.

        Use this to generate a self-contained HTML performance report
        containing statistics tables and interactive charts.

        Returns:
            `Report`: Helper object with
            report methods.

        The result is cached after first access so repeated calls are O(1).
        """
        return Report(self)

    @property
    @cached_in_slot("_utils_cache")
    def utils(self) -> "PortfolioUtils":
        """Convenience accessor returning a PortfolioUtils facade for this portfolio.

        Use this for common data transformations such as converting returns to
        prices, computing log returns, rebasing, aggregating by period, and
        computing exponential standard deviation.

        Returns:
            `PortfolioUtils`: Helper object with
            utility transform methods.

        The result is cached after first access so repeated calls are O(1).
        """
        return PortfolioUtils(self)
