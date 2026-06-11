"""Result container for system/experiment outputs."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .exceptions import MuSchemaError
from .portfolio import Portfolio


@dataclass(frozen=True)
class Result:
    """Lightweight container for system outputs.

    Attributes:
        portfolio: The portfolio constructed by a system/experiment.
        mu: Optional per-asset expected-returns surface used by some systems.
    """

    portfolio: Portfolio
    mu: pl.DataFrame | None = None

    def __post_init__(self) -> None:
        """Validate that mu (when given) is a DataFrame covering every portfolio asset.

        Raises:
            TypeError: If ``mu`` is neither ``None`` nor a `polars.DataFrame`.
            MuSchemaError: If ``mu`` lacks a column for one or more portfolio assets.
        """
        if self.mu is None:
            return
        if not isinstance(self.mu, pl.DataFrame):
            raise TypeError(f"mu must be a polars DataFrame or None, got {type(self.mu).__name__}")  # noqa: TRY003
        missing = [asset for asset in self.portfolio.assets if asset not in self.mu.columns]
        if missing:
            raise MuSchemaError(missing)

    def create_reports(self, output_dir: Path) -> None:
        """Generate CSV exports and interactive HTML plots for this result.

        Args:
            output_dir: Destination directory where two subfolders will be created:
                - data/: CSV exports of prices, profit, returns, positions, and signal (if mu present).
                - plots/: Plotly HTML reports (snapshot, lead/lag IR, lagged performance,
                  smoothed holdings performance).
        """
        data = output_dir / "data"
        plots = output_dir / "plots"

        data.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)

        self.portfolio.prices.write_csv(file=data / "prices.csv")
        self.portfolio.profit.write_csv(file=data / "profit.csv")
        self.portfolio.returns.write_csv(file=data / "returns.csv")
        self.portfolio.tilt_timing_decomp.write_csv(file=data / "tilt_timing_decomp.csv")

        if self.mu is not None:
            self.mu.write_csv(file=data / "signal.csv")

        self.portfolio.cashposition.write_csv(file=data / "position.csv")

        fig = self.portfolio.plots.snapshot()
        fig.write_html(file=plots / "snapshot.html", auto_open=False, include_plotlyjs="cdn")
        fig = self.portfolio.plots.lead_lag_ir_plot()
        fig.write_html(file=plots / "lag_ir.html", auto_open=False, include_plotlyjs="cdn")
        fig = self.portfolio.plots.lagged_performance_plot()
        fig.write_html(file=plots / "lagged_perf.html", auto_open=False, include_plotlyjs="cdn")
        fig = self.portfolio.plots.smoothed_holdings_performance_plot()
        fig.write_html(file=plots / "smooth_perf.html", auto_open=False, include_plotlyjs="cdn")
