"""Factory classmethods for constructing Portfolio objects.

`PortfolioConstructorMixin` provides the `from_risk_position`,
`from_position` and `from_weights` entry points (plus the shared
`_evaluate_position_expr` helper). All funnel through
``cls.from_cash_position`` — defined on the `Portfolio` dataclass — so the
actual instantiation happens in one place.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Self

import numpy as np
import polars as pl

from ._cost_model import CostModel
from ._portfolio_base import _PortfolioMembers
from ._rebalance import simulate_rebalancing
from .exceptions import NegativeCostBpsError, PositionExprColumnError, RowCountMismatchError, WeightsColumnError


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


def _validate_cost_bps(cost_bps: float) -> float:
    """Validate a one-way trading cost in basis points.

    Args:
        cost_bps: Candidate cost.

    Returns:
        The cost as a float.

    Raises:
        TypeError: If *cost_bps* is not a number.
        ValueError: If *cost_bps* is not finite.
        NegativeCostBpsError: If *cost_bps* is negative.
    """
    if isinstance(cost_bps, bool) or not isinstance(cost_bps, int | float):
        raise TypeError(f"cost_bps must be a number, got {type(cost_bps).__name__}")  # noqa: TRY003
    if not math.isfinite(cost_bps):
        raise ValueError(f"cost_bps must be finite, got {cost_bps}")  # noqa: TRY003
    if cost_bps < 0:
        raise NegativeCostBpsError(cost_bps)
    return float(cost_bps)


def _weights_matrix(prices: pl.DataFrame, weights: pl.DataFrame, assets: list[str]) -> np.ndarray:
    """Align *weights* to the asset columns of *prices* as a ``(t, n)`` array.

    Args:
        prices: Price levels per asset over time.
        weights: Target weights, null on rows that do not rebalance.
        assets: Numeric asset columns of *prices*, in order.

    Returns:
        Float array with NaN wherever no weight is given, including for an
        asset that has no column in *weights* at all.

    Raises:
        RowCountMismatchError: If *weights* and *prices* differ in length.
        WeightsColumnError: If *weights* has a numeric column that is not an
            asset of *prices*.
    """
    if weights.height != prices.height:
        raise RowCountMismatchError(prices.height, weights.height, "weights")
    numeric = [c for c, dtype in weights.schema.items() if dtype.is_numeric()]
    unknown = [c for c in numeric if c not in assets]
    if unknown:
        raise WeightsColumnError(unknown)
    aligned = pl.select(
        (weights[a] if a in numeric else pl.repeat(None, prices.height, eager=True)).cast(pl.Float64).alias(a)
        for a in assets
    )
    return aligned.fill_nan(None).to_numpy().astype(float)


def _validate_vol_cap(vol_cap: float | None) -> None:
    """Validate the optional ``vol_cap`` lower bound.

    Args:
        vol_cap: Candidate lower bound for the EWMA volatility estimate.

    Raises:
        ValueError: If *vol_cap* is provided but not strictly positive.
    """
    if vol_cap is not None and vol_cap <= 0:
        raise ValueError(f"vol_cap must be a positive number when provided, got {vol_cap!r}")  # noqa: TRY003


def _validate_vola(vola: int | dict[str, int], assets: list[str]) -> None:
    """Validate the EWMA ``vola`` span specification against the asset columns.

    Args:
        vola: A single span applied to every asset, or a per-asset span dict.
        assets: Numeric column names available in the price frame.

    Raises:
        ValueError: If a dict key matches no numeric column, or any span is
            not a positive integer.
    """
    if isinstance(vola, dict):
        unknown = set(vola.keys()) - set(assets)
        if unknown:
            raise ValueError(  # noqa: TRY003
                f"vola dict contains keys that do not match any numeric column in prices: {sorted(unknown)}"
            )
        for asset, span in vola.items():
            if int(span) <= 0:
                raise ValueError(f"vola span for '{asset}' must be a positive integer, got {span!r}")  # noqa: TRY003
    elif int(vola) <= 0:
        raise ValueError(f"vola span must be a positive integer, got {vola!r}")  # noqa: TRY003


class PortfolioConstructorMixin(_PortfolioMembers):
    """Mixin providing the risk- and notional-position factory classmethods."""

    if TYPE_CHECKING:

        @classmethod
        def from_cash_position(
            cls,
            prices: pl.DataFrame,
            cash_position: pl.DataFrame,
            aum: float,
            cost_per_unit: float = 0.0,
            cost_bps: float = 0.0,
            cost_model: CostModel | None = None,
            annual_fee: float = 0.0,
            compounding: bool = False,
            realized_costs: pl.DataFrame | None = None,
        ) -> Self:
            """Create a Portfolio directly from cash positions aligned with prices."""
            ...

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
        annual_fee: float = 0.0,
    ) -> Self:
        """Create a Portfolio from per-asset risk positions.

        De-volatizes each risk position using an EWMA volatility estimate
        derived from the corresponding price series.

        Args:
            prices: Price levels per asset over time.  A temporal column is
                normalised to ``'date'`` at construction, whatever it was named.
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
            annual_fee: Flat annual management fee as a fraction of AUM
                (e.g. 0.0085 for 85 bps p.a.).  Defaults to 0.0 (no fee).
                Used as the default by `deduct_management_fee`.

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

        _validate_vol_cap(vol_cap)
        _validate_vola(vola, assets)

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
        return cls.from_cash_position(
            prices=prices,
            cash_position=cash_position,
            aum=aum,
            cost_per_unit=cost_per_unit,
            cost_bps=cost_bps,
            annual_fee=annual_fee,
        )

    @classmethod
    def from_position(
        cls,
        prices: pl.DataFrame,
        position: pl.DataFrame | pl.Expr,
        aum: float,
        cost_per_unit: float = 0.0,
        cost_bps: float = 0.0,
        cost_model: CostModel | None = None,
        annual_fee: float = 0.0,
    ) -> Self:
        """Create a Portfolio from share/unit positions.

        Converts *position* (number of units held per asset) to cash exposure
        by multiplying element-wise with *prices*, then delegates to
        :py`from_cash_position`.

        Args:
            prices: Price levels per asset over time.  A temporal column is
                normalised to ``'date'`` at construction, whatever it was named.
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
            annual_fee: Flat annual management fee as a fraction of AUM
                (e.g. 0.0085 for 85 bps p.a.).  Defaults to 0.0 (no fee).
                Used as the default by `deduct_management_fee`.

        Returns:
            A Portfolio instance whose cash positions equal *position* x *prices*.

        Raises:
            PositionExprColumnError: If *position* is an expression that
                creates columns not present in *prices*.

        Examples:
            >>> import polars as pl
            >>> from jquantstats.portfolio import Portfolio
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
            annual_fee=annual_fee,
        )

    @classmethod
    def from_weights(
        cls,
        prices: pl.DataFrame,
        weights: pl.DataFrame | pl.Expr,
        aum: float,
        cost_bps: float = 0.0,
        compounding: bool = True,
        annual_fee: float = 0.0,
    ) -> Self:
        """Create a Portfolio that rebalances to target weights on sparse dates.

        *weights* gives the target fraction of NAV per asset, but only on the
        rows where the portfolio rebalances — every other row is null.  Between
        two rebalancings the book holds its **units**, so the realised weights
        drift with prices rather than staying at the targets.  On a rebalancing
        row the units are reset to ``NAV * w / price``, where NAV is the value
        the drifted book has reached, and the trade pays ``cost_bps`` on its
        traded notional out of cash.  Because each rebalancing is sized off a
        NAV that every earlier cost has already lowered, the book is simulated
        row by row rather than vectorised.

        Weights are applied at the price of the row they appear on, matching
        the convention that a position set on row *t* earns the return from
        *t* to *t + 1*.  Shift the weights first (``weights.shift(n)``) to
        trade them *n* rows later.  Weights that sum to less than one leave the
        remainder in cash, which earns nothing; weights above one borrow it.

        The trading costs are realised inside the simulation and stored as
        ``realized_costs``, so `profits`, NAV and `returns` are already net of
        them.  The portfolio's own ``cost_bps`` is therefore left at zero:
        `cost_adjusted_returns` would otherwise charge the same trades twice.

        Args:
            prices: Price levels per asset over time.  A temporal column is
                normalised to ``'date'`` at construction, whatever it was named.
            weights: Target weights per asset, aligned row for row with
                *prices*, as a DataFrame or as an expression evaluated against
                *prices*.  A row with at least one non-null weight is a
                rebalancing; within it, a null weight or a missing asset
                column means a target of zero.  Non-numeric columns (e.g.
                ``'date'``) are ignored.
            aum: Starting capital, held in cash until the first rebalancing.
            cost_bps: One-way trading cost in basis points of traded notional.
                Defaults to 0.0 (no cost).
            compounding: When True (the default) each rebalancing is sized off
                the current NAV and `returns` are measured against the
                previous period's NAV.  When False every rebalancing is sized
                off the constant *aum*, as a fixed-notional book, and
                `returns` are measured against *aum*.
            annual_fee: Flat annual management fee as a fraction of AUM, used
                as the default by `deduct_management_fee`.  Defaults to 0.0.

        Returns:
            A Portfolio whose cash positions are the simulated units times
            prices, with ``compounding`` set as requested and the costs paid
            on every trade in ``realized_costs``.

        Raises:
            RowCountMismatchError: If *weights* and *prices* differ in length.
            WeightsColumnError: If *weights* has a numeric column that is not
                an asset of *prices*.
            PositionExprColumnError: If *weights* is an expression that creates
                columns not present in *prices*.
            RebalancePriceError: If an asset that is held or targeted has no
                finite, positive price on a rebalancing row.
            NegativeCostBpsError: If *cost_bps* is negative.

        Examples:
            >>> import polars as pl
            >>> from jquantstats.portfolio import Portfolio
            >>> prices = pl.DataFrame({"A": [100.0, 110.0, 110.0], "B": [100.0, 90.0, 90.0]})
            >>> weights = pl.DataFrame({"A": [0.5, None, 0.5], "B": [0.5, None, 0.5]})
            >>> pf = Portfolio.from_weights(prices=prices, weights=weights, aum=1000.0)
            >>> [round(u, 4) for u in pf.units["A"].to_list()]  # held through row 1, reset on row 2
            [5.0, 5.0, 4.5455]
            >>> [round(w, 2) for w in pf.weights["A"].to_list()]  # drifted on row 1
            [0.5, 0.55, 0.5]
        """
        if isinstance(weights, pl.Expr):
            weights = _evaluate_position_expr(prices, weights, "weights")
        cost_bps = _validate_cost_bps(cost_bps)
        assets = [col for col, dtype in prices.schema.items() if dtype.is_numeric()]
        target = _weights_matrix(prices, weights, assets)
        price_matrix = prices.select(pl.col(a).cast(pl.Float64) for a in assets).fill_nan(None).to_numpy().astype(float)

        path = simulate_rebalancing(
            prices=price_matrix,
            weights=np.nan_to_num(target),
            is_rebalancing=np.asarray(~np.isnan(target).all(axis=1)),
            aum=aum,
            assets=assets,
            cost_bps=cost_bps,
            compounding=compounding,
        )
        cash_position = prices.with_columns(pl.Series(a, path.units[:, i]) * pl.col(a) for i, a in enumerate(assets))
        realized_costs = prices.with_columns(pl.Series(a, path.costs[:, i]) for i, a in enumerate(assets))
        return cls.from_cash_position(
            prices=prices,
            cash_position=cash_position,
            aum=aum,
            annual_fee=annual_fee,
            compounding=compounding,
            realized_costs=realized_costs,
        )
