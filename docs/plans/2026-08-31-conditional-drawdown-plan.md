# Conditional Drawdown Metrics Implementation Plan

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** Add CDaR, Drawdown VaR, Expected Drawdown, and Tail Drawdown Ratio metrics to jQuantStats (issue #862)

**Architecture:** Add four new methods to `_DrawdownMixin` in `src/jquantstats/_stats/_drawdown.py`, following existing patterns for `max_drawdown` and `avg_drawdown`. Integrate into reports and summary table.

**Tech Stack:** Python 3.11+, Polars, pytest, 100% coverage required

---

### Task 1: Add drawdown helper and expected_drawdown metric

**Files:**
- Modify: `src/jquantstats/_stats/_drawdown.py`
- Test: `tests/test_jquantstats/test__stats/test_stats.py`

**Step 1: Write the failing test**

```python
def test_expected_drawdown(stats):
    """Tests that expected_drawdown calculates average underwater drawdown correctly."""
    result = stats.expected_drawdown()
    # Expected value computed manually from test fixture data
    assert result["META"] == pytest.approx(0.0462604452968223)
```

**Step 2: Run test — confirm it fails**
Command: `pytest tests/test_jquantstats/test__stats/test_stats.py::test_expected_drawdown -v`
Expected: FAIL — function not defined

**Step 3: Write minimal implementation**
Add to `_DrawdownMixin`:
- Helper: `_drawdown_underwater(series)` returning only negative drawdown values
- Method: `expected_drawdown(self, series)` using `@columnwise_stat`

**Step 4: Run test — confirm it passes**

**Step 5: Commit**

---

### Task 2: Add drawdown_value_at_risk metric

**Files:**
- Modify: `src/jquantstats/_stats/_drawdown.py`
- Test: `tests/test_jquantstats/test__stats/test_stats.py`

**Step 1: Write the failing test**

```python
def test_drawdown_value_at_risk(stats):
    """Tests that drawdown_value_at_risk calculates DD VaR correctly."""
    result = stats.drawdown_value_at_risk(alpha=0.05)
    assert result["META"] == pytest.approx(0.1184203412837482)

def test_drawdown_value_at_risk_custom_alpha(stats):
    """Tests that custom alpha changes the result."""
    default = stats.drawdown_value_at_risk()["META"]
    deep = stats.drawdown_value_at_risk(alpha=0.01)["META"]
    assert deep > default  # More extreme tail = larger VaR
```

**Step 2: Run test — confirm it fails**

**Step 3: Write minimal implementation**
Add `drawdown_value_at_risk(self, series, alpha=0.05)` using quantile of underwater drawdowns.

**Step 4: Run test — confirm it passes**

**Step 5: Commit**

---

### Task 3: Add conditional_drawdown_at_risk (CDaR) metric

**Files:**
- Modify: `src/jquantstats/_stats/_drawdown.py`
- Test: `tests/test_jquantstats/test__stats/test_stats.py`

**Step 1: Write the failing test**

```python
def test_conditional_drawdown_at_risk(stats):
    """Tests that conditional_drawdown_at_risk calculates CDaR correctly."""
    result = stats.conditional_drawdown_at_risk(alpha=0.05)
    assert result["META"] == pytest.approx(0.1392758917384215)

def test_conditional_drawdown_at_risk_alpha_honoured(stats):
    """Tests that a non-default alpha actually changes the CDaR."""
    default = stats.conditional_drawdown_at_risk()["META"]
    deep = stats.conditional_drawdown_at_risk(alpha=0.01)["META"]
    assert deep > default
```

**Step 2: Run test — confirm it fails**

**Step 3: Write minimal implementation**
Add `conditional_drawdown_at_risk(self, series, alpha=0.05)` computing mean of drawdowns ≤ VaR threshold.

**Step 4: Run test — confirm it passes**

**Step 5: Commit**

---

### Task 4: Add tail_drawdown_ratio metric

**Files:**
- Modify: `src/jquantstats/_stats/_drawdown.py`
- Test: `tests/test_jquantstats/test__stats/test_stats.py`

**Step 1: Write the failing test**

```python
def test_tail_drawdown_ratio(stats):
    """Tests that tail_drawdown_ratio calculates CDaR / Expected Drawdown."""
    result = stats.tail_drawdown_ratio(alpha=0.05)
    expected = 0.1392758917384215 / 0.0462604452968223
    assert result["META"] == pytest.approx(expected)
```

**Step 2: Run test — confirm it fails**

**Step 3: Write minimal implementation**
Add `tail_drawdown_ratio(self, series, alpha=0.05)` = CDaR / Expected Drawdown.

**Step 4: Run test — confirm it passes**

**Step 5: Commit**

---

### Task 5: Add invalid alpha validation

**Files:**
- Modify: `src/jquantstats/_stats/_drawdown.py`
- Test: `tests/test_jquantstats/test__stats/test_stats.py`

**Step 1: Write the failing test**

```python
def test_drawdown_var_invalid_alpha_low(stats):
    """Tests that alpha <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be in"):
        stats.drawdown_value_at_risk(alpha=0)

def test_drawdown_var_invalid_alpha_high(stats):
    """Tests that alpha >= 1 raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be in"):
        stats.drawdown_value_at_risk(alpha=1.0)
```

**Step 2-5: Implement and commit**

---

### Task 6: Add metrics to HTML reports

**Files:**
- Modify: `src/jquantstats/_reports/_metrics.py`
- Test: `tests/test_jquantstats/test__reports/test_reports.py` (or similar)

**Step 1: Write the failing test**
Check that report contains the new metric labels.

**Step 2-5: Implement and commit**

---

### Task 7: Add metrics to summary table

**Files:**
- Modify: `src/jquantstats/_stats/_summary.py` (add type stubs and method calls)
- Test: `tests/test_jquantstats/test__stats/test_stats.py`

**Step 1-5: Implement and commit**

---

### Task 8: Run full test suite and quality gates

**Commands:**
```bash
make test
make fmt
make typecheck
make docs-coverage
```

All must pass.