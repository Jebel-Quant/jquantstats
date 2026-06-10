"""FastAPI web application for jquantstats portfolio analysis."""

import polars as pl
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from jquantstats import Portfolio

app = FastAPI(title="jquantstats API")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB per CSV upload


async def _read_csv(upload: UploadFile) -> pl.DataFrame:
    """Read an uploaded CSV into a DataFrame, rejecting oversized or invalid files.

    Args:
        upload: The uploaded file to parse.

    Returns:
        pl.DataFrame: The parsed CSV content.

    Raises:
        HTTPException: 413 if the file exceeds ``MAX_UPLOAD_BYTES``,
            400 if the content is not valid CSV.

    """
    raw = await upload.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        limit_mib = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"{upload.filename}: file exceeds {limit_mib} MiB limit")
    try:
        return pl.read_csv(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{upload.filename}: invalid CSV ({exc})") from exc


@app.get("/")
def root() -> dict:
    """Health check endpoint."""
    return {"status": "jquantstats API running"}


@app.post("/report", response_class=HTMLResponse)
async def generate_report(
    prices: UploadFile,
    positions: UploadFile,
    aum: float = Form(default=1_000_000, gt=0),
) -> str:
    """Accept prices and positions CSVs and return a full HTML analytics report."""
    prices_df = await _read_csv(prices)
    positions_df = await _read_csv(positions)
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
        raise HTTPException(status_code=400, detail=f"failed to build report: {exc}") from exc
