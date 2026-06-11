"""FastAPI web application for jquantstats portfolio analysis."""

import logging
import os
import time
from collections import defaultdict, deque

import polars as pl
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from jquantstats import Portfolio

logger = logging.getLogger("jquantstats.api")

app = FastAPI(title="jquantstats API")

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
    _enforce_rate_limit(request)
    prices_df = await _read_csv(prices, "prices")
    positions_df = await _read_csv(positions, "positions")
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
