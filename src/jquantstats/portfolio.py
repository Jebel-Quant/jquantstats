"""Portfolio analytics class for quant finance.

This module provides `Portfolio`, a frozen dataclass that stores the
raw portfolio inputs (prices, cash positions, AUM) and exposes both the
derived data series and the full analytics / visualisation suite.

The class is composed from four focused mixin modules:

- `PortfolioNavMixin` — NAV & returns chain
- `PortfolioAttributionMixin` — tilt/timing attribution
- `PortfolioTurnoverMixin` — turnover analytics
- `PortfolioCostMixin` — cost analysis

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
import polars.selectors as cs

from ._cost_model import CostModel
from ._plots import PortfolioPlots
from ._portfolio_attribution import PortfolioAttributionMixin
from ._portfolio_cost import PortfolioCostMixin
from ._portfolio_nav import PortfolioNavMixin
from ._portfolio_turnover import PortfolioTurnoverMixin
from ._reports import Report
from .exceptions import (
    IntegerIndexBoundError,
    InvalidCashPositionTypeError,
    InvalidPricesTypeError,
    NonPositiveAumError,
    RowCountMismatchError,
)


@dataclasses.dataclass(frozen=True, slots=True)
class Portfolio(
    PortfolioNavMixin,
    PortfolioAttributionMixin,
    PortfolioTurnoverMixin,
    PortfolioCostMixin,
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
        object.__setattr__(self, "_data_bridge", None)
        object.__setattr__(self, "_stats_cache", None)
        object.__setattr__(self, "_plots_cache", None)
        object.__setattr__(self, "_report_cache", None)
        object.__setattr__(self, "_utils_cache", None)
        object.__setattr__(self, "_profits_cache", None)
        object.__setattr__(self, "_returns_cache", None)
        object.__setattr__(self, "_tilt_cache", None)
        object.__setattr__(self, "_turnover_cache", None)

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
        """
        if isinstance(risk_position, pl.Expr):
            risk_position = prices.with_columns(risk_position)
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

        Examples:
            >>> import polars as pl
            >>> prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
            >>> pos = pl.DataFrame({"A": [10.0, 10.0, 10.0]})
            >>> pf = Portfolio.from_position(prices=prices, position=pos, aum=1e6)
            >>> pf.cashposition["A"].to_list()
            [1000.0, 1100.0, 1050.0]
        """
        if isinstance(position, pl.Expr):
            position = prices.with_columns(position)
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
        """
        if isinstance(cash_position, pl.Expr):
            cash_position = prices.with_columns(cash_position)
        if cost_model is not None:
            cost_per_unit = cost_model.cost_per_unit
            cost_bps = cost_model.cost_bps
        return cls(prices=prices, cashposition=cash_position, aum=aum, cost_per_unit=cost_per_unit, cost_bps=cost_bps)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _assert_clean_series(series: pl.Series, name: str = "") -> None:
        """Raise ValueError if *series* contains nulls or non-finite values."""
        if series.null_count() != 0:
            raise ValueError
        if not series.is_finite().all():
            raise ValueError

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
        if self._data_bridge is not None:
            return self._data_bridge
        bridge = Portfolio._build_data_bridge(self.returns)
        object.__setattr__(self, "_data_bridge", bridge)
        return bridge

    @property
    def stats(self) -> "Stats":
        """Return a Stats object built from the portfolio's daily returns.

        Delegates to the legacy `Stats` pipeline via
        `data`, so all analytics (Sharpe, drawdown, summary, etc.) are
        available through the shared implementation.

        The result is cached after first access so repeated calls are O(1).
        """
        if self._stats_cache is None:
            object.__setattr__(self, "_stats_cache", self.data.stats)
        return cast("Stats", self._stats_cache)

    @property
    def plots(self) -> PortfolioPlots:
        """Convenience accessor returning a PortfolioPlots facade for this portfolio.

        Use this to create Plotly visualizations such as snapshots, lagged
        performance curves, and lead/lag IR charts.

        Returns:
            `PortfolioPlots`: Helper object with
            plotting methods.

        The result is cached after first access so repeated calls are O(1).
        """
        if self._plots_cache is None:
            object.__setattr__(self, "_plots_cache", PortfolioPlots(self))
        return cast(PortfolioPlots, self._plots_cache)

    @property
    def report(self) -> Report:
        """Convenience accessor returning a Report facade for this portfolio.

        Use this to generate a self-contained HTML performance report
        containing statistics tables and interactive charts.

        Returns:
            `Report`: Helper object with
            report methods.

        The result is cached after first access so repeated calls are O(1).
        """
        if self._report_cache is None:
            object.__setattr__(self, "_report_cache", Report(self))
        return cast(Report, self._report_cache)

    @property
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
        if self._utils_cache is None:
            from ._utils import PortfolioUtils

            object.__setattr__(self, "_utils_cache", PortfolioUtils(self))
        return cast("PortfolioUtils", self._utils_cache)

    # ── Portfolio transforms ───────────────────────────────────────────────────

    def truncate(
        self,
        start: date | datetime | str | int | None = None,
        end: date | datetime | str | int | None = None,
    ) -> "Portfolio":
        """Return a new Portfolio truncated to the inclusive [start, end] range.

        When a ``'date'`` column is present in both prices and cash positions,
        truncation is performed by comparing the ``'date'`` column against
        ``start`` and ``end`` (which should be date/datetime values or strings
        parseable by Polars).

        When the ``'date'`` column is absent, integer-based row slicing is
        used instead.  In this case ``start`` and ``end`` must be non-negative
        integers representing 0-based row indices.  Passing non-integer bounds
        to an integer-indexed portfolio raises `TypeError`.

        In all cases the ``aum`` value is preserved.

        Args:
            start: Optional lower bound (inclusive). A date/datetime or
                Polars-parseable string when a ``'date'`` column exists; a
                non-negative int row index when the data has no ``'date'``
                column.
            end: Optional upper bound (inclusive). Same type rules as
                ``start``.

        Returns:
            A new Portfolio instance with prices and cash positions filtered
            to the specified range.

        Raises:
            TypeError: When the portfolio has no ``'date'`` column and a
                non-integer bound is supplied.
        """
        has_date = "date" in self.prices.columns
        if has_date:
            cond = pl.lit(True)
            if start is not None:
                cond = cond & (pl.col("date") >= pl.lit(start))
            if end is not None:
                cond = cond & (pl.col("date") <= pl.lit(end))
            pr = self.prices.filter(cond)
            cp = self.cashposition.filter(cond)
        else:
            if start is not None and not isinstance(start, int):
                raise IntegerIndexBoundError("start", type(start).__name__)
            if end is not None and not isinstance(end, int):
                raise IntegerIndexBoundError("end", type(end).__name__)
            row_start = int(start) if start is not None else 0
            row_end = int(end) + 1 if end is not None else self.prices.height
            length = max(0, row_end - row_start)
            pr = self.prices.slice(row_start, length)
            cp = self.cashposition.slice(row_start, length)
        return Portfolio(
            prices=pr,
            cashposition=cp,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    def lag(self, n: int) -> "Portfolio":
        """Return a new Portfolio with cash positions lagged by ``n`` steps.

        This method shifts the numeric asset columns in the cashposition
        DataFrame by ``n`` rows, preserving the ``'date'`` column and any
        non-numeric columns unchanged.  Positive ``n`` delays weights (moves
        them down); negative ``n`` leads them (moves them up); ``n == 0``
        returns the current portfolio unchanged.

        Notes:
            Missing values introduced by the shift are left as nulls;
            downstream profit computation already guards and treats nulls as
            zero when multiplying by returns.

        Args:
            n: Number of rows to shift (can be negative, zero, or positive).

        Returns:
            A new Portfolio instance with lagged cash positions and the same
            prices/AUM as the original.
        """
        if not isinstance(n, int):
            raise TypeError
        if n == 0:
            return self

        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        cp_lagged = self.cashposition.with_columns(pl.col(c).shift(n) for c in assets)
        return Portfolio(
            prices=self.prices,
            cashposition=cp_lagged,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    def smoothed_holding(self, n: int) -> "Portfolio":
        """Return a new Portfolio with cash positions smoothed by a rolling mean.

        Applies a trailing window average over the last ``n`` steps for each
        numeric asset column (excluding ``'date'``). The window length is
        ``n + 1`` so that:

        - n=0 returns the original weights (no smoothing),
        - n=1 averages the current and previous weights,
        - n=k averages the current and last k weights.

        Args:
            n: Non-negative integer specifying how many previous steps to
                include.

        Returns:
            A new Portfolio with smoothed cash positions and the same
            prices/AUM.
        """
        if not isinstance(n, int):
            raise TypeError
        if n < 0:
            raise ValueError
        if n == 0:
            return self

        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        window = n + 1
        cp_smoothed = self.cashposition.with_columns(
            pl.col(c).rolling_mean(window_size=window, min_samples=1).alias(c) for c in assets
        )
        return Portfolio(
            prices=self.prices,
            cashposition=cp_smoothed,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    def correlation(self, frame: pl.DataFrame, name: str = "portfolio") -> pl.DataFrame:
        """Compute a correlation matrix of asset returns plus the portfolio.

        Computes percentage changes for all numeric columns in ``frame``,
        appends the portfolio profit series under the provided ``name``, and
        returns the Pearson correlation matrix across all numeric columns.

        Args:
            frame: A Polars DataFrame containing at least the asset price
                columns (and a date column which will be ignored if
                non-numeric).
            name: The column name to use when adding the portfolio profit
                series to the input frame.

        Returns:
            A square Polars DataFrame where each cell is the correlation
            between a pair of series (values in [-1, 1]).
        """
        p = frame.with_columns(cs.by_dtype(pl.Float32, pl.Float64).pct_change())
        p = p.with_columns(pl.Series(name, self.profit["profit"]))
        corr_matrix = p.select(cs.numeric()).fill_null(0.0).corr()
        return corr_matrix
