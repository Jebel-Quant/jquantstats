"""Up- and down-market capture ratios.

Split out of `_reporting.py`: capture ratios are the only metrics in that
module that take an explicit benchmark series as an argument rather than
reading the benchmark off `Data`, and they share a computation that is worth
stating once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ..data import Data


class _CaptureStatsMixin:
    """Mixin providing up-market and down-market capture ratios."""

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from .._protocol import DataLike

        data: DataLike

    @staticmethod
    def _geometric_mean(series: pl.Series) -> float:
        """Geometric mean return of *series*: ``prod(1 + r)^(1/n) - 1``.

        Args:
            series: A non-empty return series.

        Returns:
            The per-period geometric mean return.
        """
        return float(float((series + 1.0).product()) ** (1.0 / len(series)) - 1.0)

    def _capture_ratio(self, benchmark: pl.Series, mask: pl.Series) -> dict[str, float]:
        """Ratio of each asset's geometric mean to the benchmark's, over *mask*.

        Shared by `up_capture` and `down_capture`, which differ only in the
        sign of the benchmark periods they select.

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.
            mask: Boolean series selecting the periods to measure over.

        Returns:
            dict[str, float]: Capture ratio per asset; ``float("nan")`` where
            the benchmark or the asset has nothing usable in the selected
            periods.
        """
        bench_selected = benchmark.filter(mask).drop_nulls()
        # A benchmark with no periods of this sign makes capture undefined for every asset.
        if bench_selected.is_empty():
            return {col: float("nan") for col, _ in self._data.items()}
        bench_geom = self._geometric_mean(bench_selected)
        if bench_geom == 0.0:  # pragma: no cover
            return {col: float("nan") for col, _ in self._data.items()}

        result: dict[str, float] = {}
        for col, series in self._data.items():
            strat_selected = series.filter(mask).drop_nulls()
            # An asset may have no usable returns during the selected periods after null filtering.
            if strat_selected.is_empty():
                result[col] = float("nan")
            else:
                result[col] = self._geometric_mean(strat_selected) / bench_geom
        return result

    def up_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Up-market capture ratio relative to an explicit benchmark series.

        Measures the fraction of the benchmark's upside that the strategy
        captures.  A value greater than 1.0 means the strategy outperformed
        the benchmark in rising markets.

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Up capture ratio per asset.

        Returns NaN when:
            Entries are ``float("nan")`` when the benchmark has no positive
            periods, its up-market geometric mean is zero, or an asset has no
            usable returns during those periods.
        """
        return self._capture_ratio(benchmark, benchmark > 0)

    def down_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Down-market capture ratio relative to an explicit benchmark series.

        A value less than 1.0 means the strategy lost less than the benchmark
        in falling markets (a desirable property).

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Down capture ratio per asset.

        Returns NaN when:
            Entries are ``float("nan")`` when the benchmark has no negative
            periods, its down-market geometric mean is zero, or an asset has no
            usable returns during those periods.
        """
        return self._capture_ratio(benchmark, benchmark < 0)
