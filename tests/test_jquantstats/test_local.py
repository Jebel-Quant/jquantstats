"""Pytest integration tests for the jquantstats FastAPI app (app.py).

Uses FastAPI's TestClient (in-process, no server subprocess required).
"""

import csv
import io
import math
import random
from datetime import date, timedelta

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed (pip install jquantstats[api])")

from starlette.testclient import TestClient

from api.app import app

# ── Constants ─────────────────────────────────────────────────────────────────

ASSETS = ["AAPL", "MSFT", "SPY"]
AUM = 1_000_000
SEED = 42
N_DAYS = 252


# ── Helpers ───────────────────────────────────────────────────────────────────


def _trading_days(n: int) -> list[date]:
    """Return the first n weekday dates starting from 2023-01-02."""
    days = []
    d = date(2023, 1, 2)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _simulate_prices(n: int, start: float, daily_vol: float, seed_offset: int) -> list[float]:
    """Generate n synthetic daily prices via a log-normal random walk."""
    random.seed(SEED + seed_offset)
    prices = [start]
    for _ in range(n - 1):
        ret = random.gauss(0.0003, daily_vol)
        prices.append(round(prices[-1] * math.exp(ret), 4))
    return prices


def _to_csv_bytes(rows: list[dict]) -> bytes:
    """Serialize a list of dicts to CSV-encoded bytes."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Create a session-scoped TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(scope="session")
def csv_files() -> dict[str, bytes]:
    """Generate synthetic prices and cash-position CSVs."""
    days = _trading_days(N_DAYS)

    price_paths = {
        "AAPL": _simulate_prices(N_DAYS, start=150.0, daily_vol=0.015, seed_offset=0),
        "MSFT": _simulate_prices(N_DAYS, start=280.0, daily_vol=0.013, seed_offset=1),
        "SPY": _simulate_prices(N_DAYS, start=400.0, daily_vol=0.010, seed_offset=2),
    }

    random.seed(SEED + 99)
    cash_paths: dict[str, list[float]] = {a: [] for a in ASSETS}
    weights = {a: AUM / len(ASSETS) for a in ASSETS}
    for i in range(N_DAYS):
        if i % 63 == 0 and i > 0:
            noise = {a: random.uniform(-0.05, 0.05) for a in ASSETS}  # noqa: S311  # nosec B311
            total = sum(1 + noise[a] for a in ASSETS)
            weights = {a: AUM * (1 + noise[a]) / total for a in ASSETS}
        for a in ASSETS:
            cash_paths[a].append(round(weights[a], 2))

    prices_csv = _to_csv_bytes(
        [{"date": days[i].isoformat(), **{a: price_paths[a][i] for a in ASSETS}} for i in range(N_DAYS)]
    )
    positions_csv = _to_csv_bytes(
        [{"date": days[i].isoformat(), **{a: cash_paths[a][i] for a in ASSETS}} for i in range(N_DAYS)]
    )

    return {"prices": prices_csv, "positions": positions_csv}


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_health_check(client: TestClient) -> None:
    """GET / returns 200 with expected status payload."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "jquantstats API running"}


def test_report_returns_html(client: TestClient, csv_files: dict[str, bytes]) -> None:
    """POST /report returns a 200 HTML response."""
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_report_html_is_non_empty(client: TestClient, csv_files: dict[str, bytes]) -> None:
    """POST /report returns a non-trivially-sized HTML report."""
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
    )
    assert len(response.text) > 1_000


def test_report_accepts_custom_aum(client: TestClient, csv_files: dict[str, bytes]) -> None:
    """POST /report accepts an aum form field instead of the hardcoded default."""
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
        data={"aum": "500000"},
    )
    assert response.status_code == 200


def test_report_rejects_nonpositive_aum(client: TestClient, csv_files: dict[str, bytes]) -> None:
    """POST /report with aum <= 0 fails form validation with 422."""
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
        data={"aum": "0"},
    )
    assert response.status_code == 422


def test_report_invalid_csv_returns_400(client: TestClient) -> None:
    """POST /report with unparseable CSV content returns 400, not 500."""
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", b"", "text/csv"),
            "positions": ("positions.csv", b"", "text/csv"),
        },
    )
    assert response.status_code == 400


def test_report_inconsistent_data_returns_400(client: TestClient, csv_files: dict[str, bytes]) -> None:
    """POST /report with CSVs that cannot form a portfolio returns 400, not 500."""
    positions = b"date,UNKNOWN\n2023-01-02,100\n2023-01-03,100\n"
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", positions, "text/csv"),
        },
    )
    assert response.status_code == 400


def test_report_oversized_upload_returns_413(
    client: TestClient, csv_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /report with a file above the size limit returns 413."""
    import api.app as app_module

    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 16)
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
    )
    assert response.status_code == 413


def test_report_too_many_rows_returns_413(
    client: TestClient, csv_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /report with a CSV above the row limit returns 413."""
    import api.app as app_module

    monkeypatch.setattr(app_module, "MAX_ROWS", 10)
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
    )
    assert response.status_code == 413
    assert "rows" in response.json()["detail"]


def test_report_too_many_columns_returns_413(
    client: TestClient, csv_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /report with a CSV above the column limit returns 413."""
    import api.app as app_module

    monkeypatch.setattr(app_module, "MAX_COLUMNS", 2)
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
    )
    assert response.status_code == 413
    assert "columns" in response.json()["detail"]


@pytest.mark.parametrize("bad_aum", ["inf", "-inf", "nan", "1e308"])
def test_report_rejects_nonfinite_or_oversized_aum(
    client: TestClient, csv_files: dict[str, bytes], bad_aum: str
) -> None:
    """POST /report with non-finite or absurdly large aum fails validation with 422."""
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", csv_files["positions"], "text/csv"),
        },
        data={"aum": bad_aum},
    )
    assert response.status_code == 422


def test_report_parse_errors_do_not_echo_filename_or_internals(client: TestClient) -> None:
    """CSV parse errors identify uploads by field name, not client-supplied filename or parser internals."""
    response = client.post(
        "/report",
        files={
            "prices": ("../secret/path.csv", b"", "text/csv"),
            "positions": ("positions.csv", b"date,A\n2023-01-02,1.0\n", "text/csv"),
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "prices: file is not valid CSV"


def test_report_build_errors_are_generic(client: TestClient, csv_files: dict[str, bytes]) -> None:
    """Report-build failures return a generic message without exception internals."""
    positions = b"date,UNKNOWN\n2023-01-02,100\n"
    response = client.post(
        "/report",
        files={
            "prices": ("prices.csv", csv_files["prices"], "text/csv"),
            "positions": ("positions.csv", positions, "text/csv"),
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "failed to build a report from the uploaded data"


def test_report_rate_limit_returns_429(
    client: TestClient, csv_files: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /report returns 429 once a client exceeds the rate limit."""
    import api.app as app_module

    monkeypatch.setattr(app_module, "RATE_LIMIT_MAX_REQUESTS", 1)
    app_module._request_log.clear()
    files = {
        "prices": ("prices.csv", csv_files["prices"], "text/csv"),
        "positions": ("positions.csv", csv_files["positions"], "text/csv"),
    }
    first = client.post("/report", files=files)
    second = client.post("/report", files=files)
    app_module._request_log.clear()
    assert first.status_code == 200
    assert second.status_code == 429


def test_rate_limit_window_evicts_old_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests older than the window are evicted, so the client is no longer throttled."""
    import api.app as app_module

    monkeypatch.setattr(app_module, "RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr(app_module, "RATE_LIMIT_WINDOW_SECONDS", 0.0)
    app_module._request_log.clear()
    app_module._request_log["testclient"].append(-1.0)  # stale entry, outside any window

    class _FakeClient:
        """Stand-in for starlette's request.client address."""

        host = "testclient"

    class _FakeRequest:
        """Stand-in for a starlette Request exposing only .client."""

        client = _FakeClient()

    app_module._enforce_rate_limit(_FakeRequest())  # stale entry evicted, no 429
    app_module._request_log.clear()


def test_cross_origin_requests_denied_by_default(client: TestClient) -> None:
    """Without JQS_CORS_ORIGINS configured, responses carry no CORS allow header."""
    response = client.get("/", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers
