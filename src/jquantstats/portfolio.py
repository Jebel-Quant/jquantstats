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

Public API is unchanged:

- Derived data series — `profits`, `profit`, `nav_accumulated`,
  `returns`, `monthly`, `nav_compounded`, `highwater`,
  `drawdown`, `all`
- Lazy composition accessors — `stats`, `plots`, `report`
- Portfolio transforms — `truncate`, `lag`, `smoothed_holding`
- Attribution — `tilt`, `timing`, `tilt_timing_decomp`
- Turnover analysis — `turnover`, `turnover_weekly`, `turnover_summary`
- Cost analysis — `cost_adjusted_returns`, `trading_cost_impact`
- Utility — `correlation`
"""

import dataclasses
from datetime import date, datetime
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from ._stats import Stats as Stats
    from ._utils import PortfolioUtils as PortfolioUtils
    from .data import Data as Data

import polars as pl

from ._cache import cached_in_slot
from ._cost_model import CostModel
from ._plots import PortfolioPlots
from ._portfolio_attribution import PortfolioAttributionMixin
from ._portfolio_cost import PortfolioCostMixin
from ._portfolio_nav import PortfolioNavMixin
from ._portfolio_transform import PortfolioTransformMixin
from ._portfolio_turnover import PortfolioTurnoverMixin
from ._reports import Report
from .exceptions import (
    InvalidCashPositionTypeError,
    InvalidPricesTypeError,
    NonPositiveAumError,
    PositionExprColumnError,
    RowCountMismatchError,
    UncleanSeriesError,
)


def _evaluate_position_expr(prices: pl.DataFrame, expr: pl.Expr, param: str) -> pl.DataFrame:
    """Evaluate a position expression against *prices* and validate the result.

    Args:
        prices: Price levels per asset over time.
        expr: Polars expression producing positions, evaluated via
            ``prices.with_columns(expr)``.
        param: Name of the parameter the expression was passed as (used in
            the error message).

    Returns:
        The evaluated positions frame, guaranteed to have the same columns
        as *prices*.

    Raises:
        PositionExprColumnError: If the expression created columns that do
            not exist in *prices* — those would leave the original asset
            columns untouched, silently treating raw prices as positions.
    """
    evaluated = prices.with_columns(expr)
    extra = [c for c in evaluated.columns if c not in prices.columns]
    if extra:
        raise PositionExprColumnError(param, extra)
    return evaluated


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


@dataclasses.dataclass(frozen=True, slots=True)
class Portfolio(
    PortfolioNavMixin,
    PortfolioAttributionMixin,
    PortfolioTurnoverMixin,
    PortfolioCostMixin,
    PortfolioTransformMixin,
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
    - Cost analysis: `cost_adjusted_returns`,
      `trading_cost_impact`
    - Utility: `correlation`

    Attributes:
        cashposition: Polars DataFrame of positions per asset over time
            (includes date column if present).
        prices: Polars DataFrame of prices per asset over time (includes date
            column if present).
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

    To sweep a range of cost assumptions use ``trading_cost_impact(max_bps=20)`` (Model B).
    To compute a net-NAV curve set ``cost_per_unit`` at construction and read ``.net_cost_nav`` (Model A).

    Date column requirement
    -----------------------
    Most analytics work with or without a ``date`` column. The following features require a
    temporal ``date`` column (``pl.Date`` or ``pl.Datetime``):

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

        Args:
            ret: Returns DataFrame, optionally with a leading ``'date'`` column.

        Returns:
            A `Data` instance backed by *ret*.
        """
        from .data import Data

        returns_only = ret.select("returns")
        if "date" in ret.columns:
            return Data(returns=returns_only, index=ret.select("date"))
        return Data(returns=returns_only, index=pl.DataFrame({"index": list(range(ret.height))}))

    def __post_init__(self) -> None:
        """Validate input types, shapes, and parameters post-initialization."""
        if not isinstance(self.prices, pl.DataFrame):
            raise InvalidPricesTypeError(type(self.prices).__name__)
        if not isinstance(self.cashposition, pl.DataFrame):
            raise InvalidCashPositionTypeError(type(self.cashposition).__name__)
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
    def from_risk_position(
        cls,
        prices: pl.DataFrame,
        risk_position: pl.DataFrame | pl.Expr,
        aum: float,
        vola: int | dict[str, int] = 32,
        vol_cap: float | None = None,
        cost_per_unit: float = 0.0,
        cost_bps: float = 0.0,
        cost_model: CostModel | None = None,
    ) -> Self:
        """Create a Portfolio from per-asset risk positions.

        De-volatizes each risk position using an EWMA volatility estimate
        derived from the corresponding price series.

        Args:
            prices: Price levels per asset over time (may include a date column).
            risk_position: Risk units per asset aligned with prices.
            vola: EWMA lookback (span-equivalent) used to estimate volatility.
                Pass an ``int`` to apply the same span to every asset, or a
                ``dict[str, int]`` to set a per-asset span (assets absent from
                the dict default to ``32``).  Every span value must be a
                positive integer; a ``ValueError`` is raised otherwise.  Dict
                keys that do not correspond to any numeric column in *prices*
                also raise a ``ValueError``.
            vol_cap: Optional lower bound for the EWMA volatility estimate.
                When provided, the vol series is clipped from below at this
                value before dividing the risk position, preventing
                position blow-up in calm, low-volatility regimes.  For
                example, ``vol_cap=0.05`` ensures annualised vol is never
                estimated below 5%.  Must be positive when not ``None``.
            aum: Assets under management used as the base NAV offset.
            cost_per_unit: One-way trading cost per unit of position change.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_bps: One-way trading cost in basis points of AUM turnover.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_model: Optional `CostModel`
                instance.  When supplied, its ``cost_per_unit`` and
                ``cost_bps`` values take precedence over the individual
                parameters above.

        Returns:
            A Portfolio instance whose cash positions are risk_position
            divided by EWMA volatility.

        Raises:
            ValueError: If any span value in *vola* is ≤ 0, or if a key in a
                *vola* dict does not match any numeric column in *prices*, or
                if *vol_cap* is provided but is not positive.
            PositionExprColumnError: If *risk_position* is an expression that
                creates columns not present in *prices*.
        """
        if isinstance(risk_position, pl.Expr):
            risk_position = _evaluate_position_expr(prices, risk_position, "risk_position")
        if cost_model is not None:
            cost_per_unit = cost_model.cost_per_unit
            cost_bps = cost_model.cost_bps
        assets = [col for col, dtype in prices.schema.items() if dtype.is_numeric()]

        # ── Validate vol_cap ──────────────────────────────────────────────────
        if vol_cap is not None and vol_cap <= 0:
            raise ValueError(f"vol_cap must be a positive number when provided, got {vol_cap!r}")  # noqa: TRY003

        # ── Validate vola ─────────────────────────────────────────────────────
        if isinstance(vola, dict):
            unknown = set(vola.keys()) - set(assets)
            if unknown:
                raise ValueError(  # noqa: TRY003
                    f"vola dict contains keys that do not match any numeric column in prices: {sorted(unknown)}"
                )
            for asset, span in vola.items():
                if int(span) <= 0:
                    raise ValueError(f"vola span for '{asset}' must be a positive integer, got {span!r}")  # noqa: TRY003
        else:
            if int(vola) <= 0:
                raise ValueError(f"vola span must be a positive integer, got {vola!r}")  # noqa: TRY003

        def _span(asset: str) -> int:
            """Return the EWMA span for *asset*, falling back to 32 if not specified."""
            if isinstance(vola, dict):
                return int(vola.get(asset, 32))
            return int(vola)

        def _vol(asset: str) -> pl.Series:
            """Return the EWMA volatility series for *asset*, optionally clipped from below."""
            vol = prices[asset].pct_change().ewm_std(com=_span(asset) - 1, adjust=True, min_samples=_span(asset))
            if vol_cap is not None:
                vol = vol.clip(lower_bound=vol_cap)
            return vol

        cash_position = risk_position.with_columns((pl.col(asset) / _vol(asset)).alias(asset) for asset in assets)
        return cls(prices=prices, cashposition=cash_position, aum=aum, cost_per_unit=cost_per_unit, cost_bps=cost_bps)

    @classmethod
    def from_position(
        cls,
        prices: pl.DataFrame,
        position: pl.DataFrame | pl.Expr,
        aum: float,
        cost_per_unit: float = 0.0,
        cost_bps: float = 0.0,
        cost_model: CostModel | None = None,
    ) -> Self:
        """Create a Portfolio from share/unit positions.

        Converts *position* (number of units held per asset) to cash exposure
        by multiplying element-wise with *prices*, then delegates to
        :py`from_cash_position`.

        Args:
            prices: Price levels per asset over time (may include a date column).
            position: Number of units held per asset over time, aligned with
                *prices*.  Non-numeric columns (e.g. ``'date'``) are passed
                through unchanged.
            aum: Assets under management used as the base NAV offset.
            cost_per_unit: One-way trading cost per unit of position change.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_bps: One-way trading cost in basis points of AUM turnover.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_model: Optional `CostModel` instance.
                When supplied, its ``cost_per_unit`` and ``cost_bps`` values
                take precedence over the individual parameters above.

        Returns:
            A Portfolio instance whose cash positions equal *position* x *prices*.

        Raises:
            PositionExprColumnError: If *position* is an expression that
                creates columns not present in *prices*.

        Examples:
            >>> import polars as pl
            >>> prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
            >>> pos = pl.DataFrame({"A": [10.0, 10.0, 10.0]})
            >>> pf = Portfolio.from_position(prices=prices, position=pos, aum=1e6)
            >>> pf.cashposition["A"].to_list()
            [1000.0, 1100.0, 1050.0]
        """
        if isinstance(position, pl.Expr):
            position = _evaluate_position_expr(prices, position, "position")
        assets = [col for col, dtype in prices.schema.items() if dtype.is_numeric()]
        cash_position = position.with_columns((pl.col(asset) * prices[asset]).alias(asset) for asset in assets)
        return cls.from_cash_position(
            prices=prices,
            cash_position=cash_position,
            aum=aum,
            cost_per_unit=cost_per_unit,
            cost_bps=cost_bps,
            cost_model=cost_model,
        )

    @classmethod
    def from_cash_position(
        cls,
        prices: pl.DataFrame,
        cash_position: pl.DataFrame | pl.Expr,
        aum: float,
        cost_per_unit: float = 0.0,
        cost_bps: float = 0.0,
        cost_model: CostModel | None = None,
    ) -> Self:
        """Create a Portfolio directly from cash positions aligned with prices.

        Args:
            prices: Price levels per asset over time (may include a date column).
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
        return cls(prices=prices, cashposition=cash_position, aum=aum, cost_per_unit=cost_per_unit, cost_bps=cost_bps)

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
        from ._utils import PortfolioUtils

        return PortfolioUtils(self)
