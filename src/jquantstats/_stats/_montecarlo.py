"""Monte Carlo simulation statistics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ..exceptions import NonPositivePeriodsPerYearError, NonPositiveWindowError

if TYPE_CHECKING:
    from ..data import Data


class _MonteCarloStatsMixin:
    """Mixin providing Monte Carlo simulation statistics for return series."""

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> int:
        """Validate that *value* is a positive integer."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NonPositiveWindowError(name)
        return value

    @staticmethod
    def _block_bootstrap_paths(values: np.ndarray, n: int, period: int, block_size: int) -> np.ndarray:
        """Generate *n* return paths via block bootstrap, returning an ``(n, period)`` array.

        All *n* paths are sampled simultaneously using fully vectorised numpy
        indexing — no Python-level loop is required.

        Args:
            values: 1-D array of historical returns.
            n: Number of simulation paths.
            period: Length of each simulated path (number of return observations).
            block_size: Block length for the block bootstrap resampling.

        Returns:
            np.ndarray: Float64 array of shape ``(n, period)``.

        """
        n_obs = values.size
        if n_obs == 0:  # pragma: no cover
            return np.full((n, period), np.nan, dtype=np.float64)

        n_blocks = math.ceil(period / block_size)
        max_start = max(1, n_obs - block_size + 1)
        # Draw all block starting indices at once: shape (n, n_blocks)
        starts = np.random.randint(0, max_start, size=(n, n_blocks))
        # Build full index array: (n, n_blocks, block_size)
        idx = starts[:, :, np.newaxis] + np.arange(block_size)[np.newaxis, np.newaxis, :]
        idx = np.clip(idx, 0, n_obs - 1)
        # Flatten and trim to the requested period length
        return np.asarray(values[idx].reshape(n, -1)[:, :period])

    def _simulate_distribution(self, n: int, period: int) -> dict[str, np.ndarray]:
        """Prepare validated inputs and sample *n* block-bootstrap paths per asset.

        Returns a dict mapping each asset column name to an ``(n, period)``
        float64 array of simulated return paths.  Assets with no usable data
        map to an ``(n, period)`` array of NaN.

        Args:
            n: Number of simulation paths (must be a positive integer).
            period: Path length in return observations (must be a positive integer).

        Returns:
            dict[str, np.ndarray]: Asset name → ``(n, period)`` paths array.

        """
        n = self._validate_positive_integer("n", n)
        period = self._validate_positive_integer("period", period)
        block_size = max(1, round(period**0.5))

        paths: dict[str, np.ndarray] = {}
        for col, series in self._data.items():
            clean = series.cast(pl.Float64).drop_nulls().drop_nans()
            values = np.asarray(clean.to_numpy(), dtype=np.float64)
            if values.size == 0:
                paths[col] = np.full((n, period), np.nan, dtype=np.float64)
            else:
                block = min(block_size, values.size)
                paths[col] = self._block_bootstrap_paths(values, n, period, block)
        return paths

    def montecarlo(self, n: int = 1000, period: int = 252) -> pl.DataFrame:
        """Simulate cumulative returns across *n* block-bootstrap paths.

        Args:
            n: Number of Monte Carlo paths. Defaults to 1000.
            period: Simulation horizon in return observations. Defaults to 252.

        Returns:
            pl.DataFrame: Shape ``(n, n_assets)`` — one simulated terminal
            cumulative return per path and asset.


        Returns NaN when:
            Entries are NaN for assets with no usable (non-null, non-NaN)
            observations.
        """
        paths = self._simulate_distribution(n=n, period=period)
        result = {col: np.prod(1.0 + arr, axis=1) - 1.0 for col, arr in paths.items()}
        return pl.DataFrame(result)

    def montecarlo_sharpe(
        self,
        n: int = 1000,
        period: int = 252,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Simulate the Sharpe-ratio distribution across block-bootstrap paths.

        Args:
            n: Number of Monte Carlo paths. Defaults to 1000.
            period: Simulation horizon in return observations. Defaults to 252.
            periods_per_year: Annualisation factor. Defaults to the value
                inferred from the data.

        Returns:
            pl.DataFrame: Shape ``(n, n_assets)`` — one simulated annualised
            Sharpe ratio per path and asset.


        Returns NaN when:
            Entries are NaN when a path's standard deviation is zero or the asset
            has no usable observations.
        """
        ppy = self._data._periods_per_year if periods_per_year is None else periods_per_year
        if ppy <= 0:
            raise NonPositivePeriodsPerYearError
        scale = math.sqrt(ppy)
        paths = self._simulate_distribution(n=n, period=period)
        result: dict[str, np.ndarray] = {}
        for col, arr in paths.items():
            means = arr.mean(axis=1)
            stds = arr.std(axis=1, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                result[col] = np.where(stds == 0.0, np.nan, means / stds * scale)
        return pl.DataFrame(result)

    def montecarlo_drawdown(self, n: int = 1000, period: int = 252) -> pl.DataFrame:
        """Simulate the maximum-drawdown distribution across block-bootstrap paths.

        Args:
            n: Number of Monte Carlo paths. Defaults to 1000.
            period: Simulation horizon in return observations. Defaults to 252.

        Returns:
            pl.DataFrame: Shape ``(n, n_assets)`` — one simulated maximum
            drawdown per path and asset (values in ``[-1, 0]``).


        Returns NaN when:
            Entries are NaN for assets with no usable (non-null, non-NaN)
            observations.
        """
        paths = self._simulate_distribution(n=n, period=period)
        result: dict[str, np.ndarray] = {}
        for col, arr in paths.items():
            nav = np.cumprod(1.0 + arr, axis=1)
            hwm = np.maximum.accumulate(nav, axis=1)
            result[col] = np.min(nav / hwm - 1.0, axis=1)
        return pl.DataFrame(result)

    def montecarlo_cagr(
        self,
        n: int = 1000,
        period: int = 252,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Simulate the CAGR distribution across block-bootstrap paths.

        Args:
            n: Number of Monte Carlo paths. Defaults to 1000.
            period: Simulation horizon in return observations. Defaults to 252.
            periods_per_year: Annualisation factor. Defaults to the value
                inferred from the data.

        Returns:
            pl.DataFrame: Shape ``(n, n_assets)`` — one simulated annualised
            CAGR per path and asset.


        Returns NaN when:
            Entries are NaN when a path's terminal compound return is non-positive
            or the asset has no usable observations.
        """
        ppy = self._data._periods_per_year if periods_per_year is None else periods_per_year
        if ppy <= 0:
            raise NonPositivePeriodsPerYearError
        years = period / ppy
        paths = self._simulate_distribution(n=n, period=period)
        result: dict[str, np.ndarray] = {}
        for col, arr in paths.items():
            totals = np.prod(1.0 + arr, axis=1)
            with np.errstate(invalid="ignore"):
                result[col] = np.where(totals > 0, totals ** (1.0 / years) - 1.0, np.nan)
        return pl.DataFrame(result)
