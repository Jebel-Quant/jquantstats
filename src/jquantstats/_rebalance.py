"""Path-dependent simulation of a portfolio rebalanced to sparse target weights.

`Portfolio.from_weights` takes target weights that are set on rebalancing dates
only, and this module turns them into the units actually held. Between two
rebalancings the book holds *units*, not weights, so the realised weights drift
with prices. At a rebalancing the units are reset to ``nav * w / price``, where
``nav`` is the value of the drifted book on that row, and the trade is charged
a cost proportional to its traded notional, paid out of cash.

Because the units set at a rebalancing depend on the nav reached there, and the
cost paid there feeds into every nav after it, this is a genuine recursion in
nav and cash and is simulated one row at a time. The design follows
``qis.backtest_rebalanced_portfolio``: target weights are applied to the
pre-trade nav, and costs are ``cost_bps / 10_000 * price * |Δunits|`` per asset.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .exceptions import RebalancePriceError


class RebalancingPath(NamedTuple):
    """Row-by-row state of a simulated rebalanced portfolio.

    Attributes:
        units: Units held per asset after trading on each row, shape ``(t, n)``.
        costs: Trading cost charged per asset on each row, shape ``(t, n)``.
        cash: Cash balance after trading on each row, shape ``(t,)``.
        nav: Net asset value after trading on each row, shape ``(t,)``.
    """

    units: np.ndarray
    costs: np.ndarray
    cash: np.ndarray
    nav: np.ndarray


def simulate_rebalancing(
    prices: np.ndarray,
    weights: np.ndarray,
    is_rebalancing: np.ndarray,
    aum: float,
    assets: list[str],
    cost_bps: float = 0.0,
    compounding: bool = True,
) -> RebalancingPath:
    """Simulate a book that trades to target weights on rebalancing rows only.

    On a rebalancing row ``t`` the target units are ``base * weights[t] /
    prices[t]``, where ``base`` is the pre-trade nav when *compounding* and the
    constant *aum* otherwise. On every other row the previous units are held.
    The trade cost ``cost_bps / 10_000 * prices[t] * |Δunits|`` is paid out of
    cash, so it lowers the nav every later rebalancing is sized off.

    A price that is missing between two rebalancings drops that leg from the
    nav for the row, as the units are still held. On a rebalancing row,
    however, every asset that is held or targeted must have a finite, positive
    price: the trade could not be valued otherwise.

    Args:
        prices: Price per asset, shape ``(t, n)``.
        weights: Target weights, shape ``(t, n)``. Rows where *is_rebalancing*
            is False are ignored.
        is_rebalancing: Boolean flag per row, shape ``(t,)``.
        aum: Starting capital, held as cash until the first rebalancing.
        assets: Asset names in column order, used in error messages.
        cost_bps: One-way cost in basis points of traded notional.
        compounding: Size trades off the current nav when True, off the
            constant *aum* when False.

    Returns:
        RebalancingPath: Units, per-asset costs, cash and nav for every row.

    Raises:
        RebalancePriceError: If an asset that is held or targeted has no
            finite, positive price on a rebalancing row.
    """
    n_rows, n_assets = prices.shape
    rate = cost_bps / 10_000.0
    units = np.zeros((n_rows, n_assets))
    costs = np.zeros((n_rows, n_assets))
    cash = np.zeros(n_rows)
    nav = np.zeros(n_rows)

    held = np.zeros(n_assets)
    balance = float(aum)
    for t in range(n_rows):
        price = prices[t]
        if is_rebalancing[t]:
            involved = (weights[t] != 0.0) | (held != 0.0)
            unpriced = involved & ~(np.isfinite(price) & (price > 0.0))
            if unpriced.any():
                raise RebalancePriceError(t, [assets[i] for i in np.flatnonzero(unpriced)])
            pre_trade_nav = balance + np.nansum(held * price)
            base = pre_trade_nav if compounding else aum
            target = np.divide(base * weights[t], price, out=np.zeros(n_assets), where=weights[t] != 0.0)
            traded = np.multiply(target - held, price, out=np.zeros(n_assets), where=involved)
            costs[t] = rate * np.abs(traded)
            balance -= traded.sum() + costs[t].sum()
            held = target
        units[t] = held
        cash[t] = balance
        nav[t] = balance + np.nansum(held * price)

    return RebalancingPath(units=units, costs=costs, cash=cash, nav=nav)
