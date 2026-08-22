"""jQuantStats: Portfolio analytics for quants.

Two entry points
----------------
**Entry point 1 — prices + positions (recommended for active portfolios):**

Use `Portfolio` when you have price series and
position sizes.  Portfolio compiles the NAV curve from raw inputs and exposes
the full analytics suite via ``.stats``, ``.plots``, and ``.report``.

    >>> import polars as pl
    >>> from jquantstats import Portfolio
    >>> prices = pl.DataFrame(
    ...     {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "A": [100.0, 101.0, 99.0]}
    ... ).with_columns(pl.col("date").str.to_date())
    >>> positions = pl.DataFrame(
    ...     {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "A": [1000.0, 1000.0, 1000.0]}
    ... ).with_columns(pl.col("date").str.to_date())
    >>> pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1_000_000)
    >>> pf.assets
    ['A']
    >>> sorted(pf.stats.sharpe())
    ['returns']
    >>> type(pf.plots.snapshot()).__name__
    'Figure'

**Entry point 2 — returns series (for arbitrary return streams):**

Use `Data` when you already have a returns series
(e.g. downloaded from a data vendor) and want benchmark comparison or
factor analytics.

    >>> from jquantstats import Data
    >>> returns = pl.DataFrame(
    ...     {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "Asset1": [0.01, -0.02, 0.03]}
    ... ).with_columns(pl.col("date").str.to_date())
    >>> benchmark = pl.DataFrame(
    ...     {"date": ["2023-01-01", "2023-01-02", "2023-01-03"], "Market": [0.005, -0.01, 0.02]}
    ... ).with_columns(pl.col("date").str.to_date())
    >>> data = Data.from_returns(returns=returns, benchmark=benchmark)
    >>> data.benchmark.columns
    ['Market']
    >>> type(data.plots.snapshot(title="Performance")).__name__
    'Figure'

The two APIs are layered: ``portfolio.data`` returns a `Data`
object so you can always drop into the returns-series API from a Portfolio.

    >>> type(pf.data).__name__
    'Data'

**Choosing a rendering backend:**

Charts render with Plotly by default.  Install ``jquantstats[mpl]`` and select
matplotlib for static figures instead — appreciably cheaper when a script builds
many charts at once.  See `set_plot_backend`, `plot_backend` and
`get_plot_backend`.

    >>> from jquantstats import get_plot_backend
    >>> get_plot_backend()
    'plotly'

For more information, visit the `jQuantStats Documentation <https://jebel-quant.github.io/jquantstats/book>`_.
"""

import importlib.metadata

from ._cost_model import CostModel as CostModel
from ._plots._backend import get_plot_backend as get_plot_backend
from ._plots._backend import plot_backend as plot_backend
from ._plots._backend import set_plot_backend as set_plot_backend
from ._types import NativeFrame as NativeFrame
from ._types import NativeFrameOrScalar as NativeFrameOrScalar
from .data import Data as Data
from .data import interpolate as interpolate
from .portfolio import Portfolio as Portfolio
from .result import Result as Result

__version__ = importlib.metadata.version("jquantstats")

__all__ = [
    "CostModel",
    "Data",
    "NativeFrame",
    "NativeFrameOrScalar",
    "Portfolio",
    "Result",
    "get_plot_backend",
    "interpolate",
    "plot_backend",
    "set_plot_backend",
]
