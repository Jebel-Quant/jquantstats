"""Multi-panel performance dashboard figure builder."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ._styling import _apply_base_layout, _hex_to_rgba, _ticker_colors


def _plot_performance_dashboard(returns: pl.DataFrame, log_scale: bool = False) -> go.Figure:
    """Build a multi-panel performance dashboard figure for the given returns.

    Args:
        returns: A Polars DataFrame with a date column followed by one column per asset.
        log_scale: Whether to use a logarithmic y-axis for cumulative returns.

    Returns:
        A Plotly Figure containing cumulative returns, drawdowns, and monthly returns panels.

    """
    # Get the date column name from the first column of the DataFrame
    date_col = returns.columns[0]

    # Get the tickers (all columns except the date column)
    tickers = [col for col in returns.columns if col != date_col]

    # Calculate cumulative returns (prices)
    prices = returns.with_columns([((1 + pl.col(ticker)).cum_prod()).alias(f"{ticker}_price") for ticker in tickers])

    colors = _ticker_colors(tickers)
    colors.update({f"{ticker}_light": _hex_to_rgba(colors[ticker]) for ticker in tickers})

    # Resample to monthly returns
    monthly_returns = returns.group_by_dynamic(
        index_column=date_col, every="1mo", period="1mo", closed="right", label="right"
    ).agg([((pl.col(ticker) + 1.0).product() - 1.0).alias(ticker) for ticker in tickers])

    # Create subplot grid with domain for stats table
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Cumulative Returns", "Drawdowns", "Monthly Returns"],
        vertical_spacing=0.05,
    )

    # --- Row 1: Cumulative Returns
    for ticker in tickers:
        price_col = f"{ticker}_price"
        fig.add_trace(
            go.Scatter(
                x=prices[date_col],
                y=prices[price_col],
                mode="lines",
                name=ticker,
                legendgroup=ticker,
                line={"color": colors[ticker], "width": 2},
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{ticker}: %{{y:.2f}}x",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # --- Row 2: Drawdowns
    for ticker in tickers:
        price_col = f"{ticker}_price"
        # Calculate drawdowns using polars
        price_series = prices[price_col]
        cummax = prices.select(pl.col(price_col).cum_max().alias("cummax"))
        dd_values = ((price_series - cummax["cummax"]) / cummax["cummax"]).to_list()

        fig.add_trace(
            go.Scatter(
                x=prices[date_col],
                y=dd_values,
                mode="lines",
                fill="tozeroy",
                fillcolor=colors[f"{ticker}_light"],
                line={"color": colors[ticker], "width": 1},
                name=ticker,
                legendgroup=ticker,
                hovertemplate=f"{ticker} Drawdown: %{{y:.2%}}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=1)

    # --- Row 3: Monthly Returns
    for ticker in tickers:
        # Get monthly returns values as a list for coloring
        monthly_values = monthly_returns[ticker].to_list()

        # If there's only one ticker, use green for positive returns and red for negative returns
        if len(tickers) == 1:
            bar_colors = ["green" if val > 0 else "red" for val in monthly_values]
        else:
            bar_colors = [colors[ticker] if val > 0 else colors[f"{ticker}_light"] for val in monthly_values]

        fig.add_trace(
            go.Bar(
                x=monthly_returns[date_col],
                y=monthly_returns[ticker],
                name=ticker,
                legendgroup=ticker,
                marker={
                    "color": bar_colors,
                    "line": {"width": 0},
                },
                opacity=0.8,
                hovertemplate=f"{ticker} Monthly Return: %{{y:.2%}}",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    _apply_base_layout(fig, f"{' vs '.join(tickers)} Performance Dashboard", height=1200)

    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1, tickformat=".2f")
    fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".0%")
    fig.update_yaxes(title_text="Monthly Return", row=3, col=1, tickformat=".0%")

    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)

    return fig
