"""Concentration metrics for the signed distribution of returns.

Split out of :mod:`jquantstats._stats._performance`, which had grown to 714 lines
across four unrelated concerns. This module owns the Herfindahl-Hirschman Index
applied to returns; the risk-adjusted-ratio family stays in ``_performance`` and
the benchmark-relative metrics live in ``_benchmark``.
"""

from __future__ import annotations

import polars as pl

from ._core import columnwise_stat

# ── Concentration metrics (HHI) mixin ────────────────────────────────────────


class _ConcentrationStatsMixin:
    """Mixin providing temporal concentration metrics for gains and losses.

    ``hhi_positive`` and ``hhi_negative`` apply the Herfindahl-Hirschman Index to
    the signed distribution of returns, measuring *temporal* concentration of
    gains and losses respectively — a value near 0 means returns are spread evenly
    across periods; a value near 1 means a single period dominates.

    **Intentionally public, optional use.** These are not included in
    ``summary()`` by default because they are supplemental diagnostics rather than
    standard risk-adjusted-return measures, but they are fully supported as part
    of the public ``Stats`` API.

    Both metrics depend only on the passed series — no benchmark, no cross-mixin
    state.
    """

    @columnwise_stat
    def hhi_positive(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for positive returns.

        This quantifies how concentrated the positive returns are in a series.

        .. math::
            w^{\plus} = \frac{r_{t}^{\plus}}{\sum{r_{t}^{\plus}}} \\
            HHI^{\plus} = \frac{N_{\plus} \sum{(w^{\plus})^2} - 1}{N_{\plus} - 1}

        where:
            - \(r_{t}^{\plus}\) are the positive returns
            - \(N_{\plus}\) is the number of positive returns
            - \(w^{\plus}\) are the weights of positive returns

        Args:
            series (pl.Series): The series to calculate HHI for.

        Returns:
            float: The HHI value for positive returns. Returns NaN if fewer than 3
                positive returns are present.

        Note:
            Values range from 0 (perfectly diversified gains) to 1 (all gains
            concentrated in a single period).
        """
        positive_returns = series.filter(series > 0).drop_nans()
        if positive_returns.len() <= 2:
            return float("nan")  # indeterminate: fewer than 3 positive returns
        weight = positive_returns / positive_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))

    @columnwise_stat
    def hhi_negative(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for negative returns.

        This quantifies how concentrated the negative returns are in a series.

        .. math::
            w^{\minus} = \frac{r_{t}^{\minus}}{\sum{r_{t}^{\minus}}} \\
            HHI^{\minus} = \frac{N_{\minus} \sum{(w^{\minus})^2} - 1}{N_{\minus} - 1}

        where:
            - \(r_{t}^{\minus}\) are the negative returns
            - \(N_{\minus}\) is the number of negative returns
            - \(w^{\minus}\) are the weights of negative returns

        Args:
            series (pl.Series): The returns series to calculate HHI for.

        Returns:
            float: The HHI value for negative returns. Returns NaN if fewer than 3
                negative returns are present.

        Note:
            Values range from 0 (perfectly diversified losses) to 1 (all losses
            concentrated in a single period).
        """
        negative_returns = series.filter(series < 0).drop_nans()
        if negative_returns.len() <= 2:
            return float("nan")  # indeterminate: fewer than 3 negative returns
        weight = negative_returns / negative_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))
