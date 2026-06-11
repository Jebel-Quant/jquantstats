"""FastAPI web application for jquantstats portfolio analysis."""

import logging
import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

import polars as pl
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from jquantstats import Portfolio

logger = logging.getLogger("jquantstats.api")

app = FastAPI(title="jquantstats API")

# Optional API-key auth: set JQS_API_KEY to require clients to send the same
# value in the X-API-Key header. Empty (default) leaves the API open.
API_KEY = os.environ.get("JQS_API_KEY", "")


@app.middleware("http")
async def _log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Log every request with method, path, status, duration, and client IP."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    client = request.client.host if request.client else "unknown"
    logger.info(
        "%s %s -> %d (%.1f ms, client=%s)", request.method, request.url.path, response.status_code, duration_ms, client
    )
    return response


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB per CSV upload
MAX_ROWS = 100_000  # max rows per parsed CSV
MAX_COLUMNS = 1_000  # max columns per parsed CSV
MAX_AUM = 1e15  # upper bound on the aum form field

# Cross-origin browser requests are denied unless origins are explicitly
# allowed via the JQS_CORS_ORIGINS env var (comma-separated list).
_cors_origins = [origin.strip() for origin in os.environ.get("JQS_CORS_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Sliding-window, per-client-IP, in-process rate limit. Sufficient for the
# single-instance Railway deployment; multi-instance deployments need a
# shared limiter (reverse proxy or Redis-backed) in front instead.
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("JQS_RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("JQS_RATE_LIMIT_WINDOW_SECONDS", "60"))

_request_log: dict[str, deque[float]] = defaultdict(deque)


def _require_api_key(request: Request) -> None:
    """Reject the request with 401 when JQS_API_KEY is set and the header doesn't match.

    Args:
        request: The incoming request; the key is read from the ``X-API-Key`` header.

    Raises:
        HTTPException: 401 when authentication is enabled and the key is wrong or absent.

    """
    if API_KEY and not secrets.compare_digest(request.headers.get("x-api-key", ""), API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _validate_date_alignment(prices_df: pl.DataFrame, positions_df: pl.DataFrame) -> None:
    """Reject the request with 422 when prices and positions dates don't line up.

    Portfolio construction validates row counts but aligns rows by position —
    two frames with matching row counts but different dates would silently
    produce nonsensical analytics.

    Args:
        prices_df: Parsed prices upload.
        positions_df: Parsed positions upload.

    Raises:
        HTTPException: 422 when only one frame has a ``date`` column, or both
            have one but the date values differ.

    """
    has_prices_date = "date" in prices_df.columns
    has_positions_date = "date" in positions_df.columns
    if has_prices_date != has_positions_date:
        raise HTTPException(
            status_code=422,
            detail="prices and positions must both have a 'date' column, or neither",
        )
    if has_prices_date and prices_df["date"].to_list() != positions_df["date"].to_list():
        raise HTTPException(status_code=422, detail="prices and positions must cover the same dates")


def _enforce_rate_limit(request: Request) -> None:
    """Reject the request with 429 when the client exceeds the rate limit.

    Tracks request timestamps per client IP in a sliding window of
    ``RATE_LIMIT_WINDOW_SECONDS``; at most ``RATE_LIMIT_MAX_REQUESTS``
    requests are allowed per window.

    Args:
        request: The incoming request, used to identify the client.

    Raises:
        HTTPException: 429 when the limit is exceeded.

    """
    now = time.monotonic()
    client = request.client.host if request.client else "unknown"
    window = _request_log[client]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="rate limit exceeded, retry later")
    window.append(now)


async def _read_csv(upload: UploadFile, label: str) -> pl.DataFrame:
    """Read an uploaded CSV into a DataFrame, rejecting oversized or invalid files.

    Error responses identify the upload only by *label* (the form field name),
    never by the client-supplied filename, and never include parser internals —
    those are logged server-side instead.

    Args:
        upload: The uploaded file to parse.
        label: Form-field name used to identify the upload in error responses.

    Returns:
        pl.DataFrame: The parsed CSV content.

    Raises:
        HTTPException: 413 if the file exceeds ``MAX_UPLOAD_BYTES`` or the
            parsed frame exceeds ``MAX_ROWS``/``MAX_COLUMNS``,
            400 if the content is not valid CSV.

    """
    raw = await upload.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        limit_mib = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"{label}: file exceeds {limit_mib} MiB limit")
    try:
        frame = pl.read_csv(raw)
    except Exception as exc:
        logger.warning("rejected %s upload: CSV parse failed: %s", label, exc)
        raise HTTPException(status_code=400, detail=f"{label}: file is not valid CSV") from exc
    if frame.height > MAX_ROWS or frame.width > MAX_COLUMNS:
        raise HTTPException(
            status_code=413,
            detail=f"{label}: CSV exceeds {MAX_ROWS} rows or {MAX_COLUMNS} columns",
        )
    logger.info("%s upload accepted: %d bytes, %d rows, %d columns", label, len(raw), frame.height, frame.width)
    return frame


@app.get("/")
def root() -> dict:
    """Health check endpoint."""
    return {"status": "jquantstats API running"}


@app.post("/report", response_class=HTMLResponse)
async def generate_report(
    request: Request,
    prices: UploadFile,
    positions: UploadFile,
    aum: float = Form(default=1_000_000, gt=0, le=MAX_AUM, allow_inf_nan=False),
) -> str:
    """Accept prices and positions CSVs and return a full HTML analytics report."""
    _require_api_key(request)
    _enforce_rate_limit(request)
    prices_df = await _read_csv(prices, "prices")
    positions_df = await _read_csv(positions, "positions")
    _validate_date_alignment(prices_df, positions_df)
    try:
        pf = Portfolio.from_cash_position(
            prices=prices_df,
            cash_position=positions_df,
            aum=aum,
        )
        return pf.report.to_html()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("report build failed: %s", exc)
        raise HTTPException(status_code=400, detail="failed to build a report from the uploaded data") from exc
