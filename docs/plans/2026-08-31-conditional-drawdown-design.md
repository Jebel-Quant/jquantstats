# Design: Conditional Drawdown Metrics (Issue #862)

**Date:** 2026-08-31
**Issue:** #862 - "conditional drawdown and friends"

## Overview

Add a comprehensive suite of conditional drawdown risk metrics to jQuantStats, implementing the drawdown analogs of Value-at-Risk and Expected Shortfall. These metrics are widely used in quantitative risk management for tail risk assessment.

## Metrics to Implement

### 1. Drawdown Value at Risk (DD_VaR)
- The α-quantile of the drawdown distribution (when underwater)
- Example: DD_VaR(95%) = the drawdown level that only 5% of underwater periods exceed
- Returns positive fraction (e.g., 0.15 for 15%)

### 2. Conditional Drawdown at Risk (CDaR / Expected Drawdown Shortfall)
- The expected drawdown given that drawdown exceeds DD_VaR
- The drawdown analog of CVaR / Expected Shortfall
- CDaR(α) = E[DD | DD ≤ DD_VaR(α)] where DD is negative
- Returns positive fraction

### 3. Expected Drawdown (when underwater)
- Average drawdown during underwater periods only
- Currently `avg_drawdown` averages over ALL periods (including 0 drawdown)
- This metric only averages over periods where drawdown < 0

### 4. Tail Drawdown Ratio
- Ratio of CDaR to Expected Drawdown (when underwater)
- Analog of tail ratio for drawdowns
- Higher values indicate fatter tail in drawdown distribution

### 5. Ulcer Performance Index (UPI)
- Already exists as `ulcer_performance_index`
- (CAGR - rf) / Ulcer Index
- Ulcer Index = RMS of drawdowns = sqrt(mean(dd²))

## Architecture

### Location
Add to `_DrawdownMixin` in `src/jquantstats/_stats/_drawdown.py` - follows existing pattern for drawdown metrics.

### Method Signatures

```python
@columnwise_stat
def drawdown_value_at_risk(self, series: pl.Series, alpha: float = 0.05) -> float:
    """Drawdown VaR at confidence level alpha (default 95%)."""

@columnwise_stat
def conditional_drawdown_at_risk(self, series: pl.Series, alpha: float = 0.05) -> float:
    """CDaR / Expected Drawdown Shortfall at confidence level alpha."""

@columnwise_stat
def expected_drawdown(self, series: pl.Series) -> float:
    """Average drawdown during underwater periods only."""

@columnwise_stat
def tail_drawdown_ratio(self, series: pl.Series, alpha: float = 0.05) -> float:
    """CDaR / Expected Drawdown ratio."""
```

### Parameters
- `alpha`: Tail probability (e.g., 0.05 for 95% confidence). Default 0.05.
- Follows same convention as `value_at_risk` and `conditional_value_at_risk`

### Return Values
- All return positive fractions (e.g., 0.15 for 15% drawdown)
- NaN when no underwater periods exist or insufficient data

## Implementation Details

### Helper Functions
- Reuse `_drawdown_with_baseline` from `_basic_core.py` for consistent drawdown series
- Need a `_drawdown_underwater` helper to get only negative drawdown values

### Edge Cases
- Series with no drawdown (monotonically increasing) → return NaN
- Series with insufficient underwater observations → return NaN
- Alpha validation: must be in (0, 1)

## Testing Strategy

### Unit Tests
- Test each metric against known values
- Test with monotonic series (no drawdown)
- Test with constant series
- Test with various alpha values (0.05, 0.01, 0.10)
- Test edge cases (all positive returns, all negative returns)

### Integration Tests
- Verify metrics appear in `summary()` table
- Verify metrics appear in HTML reports
- Test both Data and Portfolio routes

### Parity Tests
- Compare against manual calculation
- No direct QuantStats equivalent, so validate mathematically

## Reporting Integration

Add to `_add_drawdown_rows` in `src/jquantstats/_reports/_metrics.py`:
```python
rows.append(("Drawdown VaR (95%)", _pct(_safe(s.drawdown_value_at_risk))))
rows.append(("CDaR (95%)", _pct(_safe(s.conditional_drawdown_at_risk))))
rows.append(("Expected Drawdown", _pct(_safe(s.expected_drawdown))))
rows.append(("Tail DD Ratio", _safe(s.tail_drawdown_ratio)))
```

## Type Annotations

Follow existing patterns:
- Input: `pl.Series`
- Output: `float` for columnwise methods, `dict[str, float]` for DataFrame methods
- Use `@columnwise_stat` decorator for per-asset computation

## Acceptance Criteria

1. All 4 new metrics implemented and accessible via `data.stats.*` and `portfolio.stats.*`
2. 100% test coverage (line and branch)
3. All existing tests pass
4. Metrics appear in `summary()` and HTML reports
5. `make test`, `make fmt`, `make typecheck`, `make docs-coverage` all pass