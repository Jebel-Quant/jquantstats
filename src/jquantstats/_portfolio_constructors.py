"""Factory classmethods for constructing Portfolio objects.

`PortfolioConstructorMixin` provides the `from_risk_position` and
`from_position` entry points (plus the shared `_evaluate_position_expr`
helper). Both funnel through ``cls.from_cash_position`` — defined on the
`Portfolio` dataclass — so the actual instantiation happens in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import polars as pl

from ._cost_model import CostModel
from ._portfolio_base import _PortfolioMembers
from .exceptions import PositionExprColumnError


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
        return cls.from_cash_position(
            prices=prices,
            cash_position=cash_position,
            aum=aum,
            cost_per_unit=cost_per_unit,
            cost_bps=cost_bps,
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
