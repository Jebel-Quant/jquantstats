"""Recipe-backed portfolio context for the jquantstats Claude plugin.

There is no live Python process shared between tool calls, so a portfolio
cannot be handed from one skill invocation to the next. What travels instead is
a *recipe*: the paths and constructor arguments needed to rebuild the object
deterministically, stored in ``.jquantstats/context.json``.

Two files, deliberately split by lifecycle:

* ``.jquantstats/context.json`` — the recipe. Authoritative, tiny,
  hand-editable, worth committing.
* ``.jquantstats/digest.json`` — derived facts (shape, nulls, anchor metrics,
  warnings) plus content fingerprints of every input. Regenerable, and stale
  the moment an input file changes, so it is gitignored and fingerprint-checked
  before its numbers are quoted.

Typical use from an analysis snippet::

    import sys; sys.path.insert(0, os.environ["CLAUDE_PLUGIN_ROOT"] + "/scripts")
    from jqs_context import load

    pf = load()          # active context
    pf = load("lag1")    # a named variant
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

SCHEMA = 1
"""Version of the on-disk recipe/digest format."""

CONTEXT_DIRNAME = ".jquantstats"
CONTEXT_FILENAME = "context.json"
DIGEST_FILENAME = "digest.json"

PORTFOLIO_CONSTRUCTORS = {
    "from_cash_position": "cash_position",
    "from_position": "position",
    "from_risk_position": "risk_position",
}
"""Portfolio constructor -> the name of its position input."""

DATA_CONSTRUCTORS = {"from_returns": "returns", "from_prices": "prices"}
"""Data constructor -> the name of its primary input."""

# The Portfolio -> Data bridge keeps the date axis only when the price frame's
# date column is literally named "date"; any other name is silently dropped and
# replaced by a positional integer index, which changes every annualised metric.
# All Portfolio inputs are therefore normalised to this name.
PORTFOLIO_DATE_COL = "date"

_ANCHOR_METRICS = ("comp", "cagr", "sharpe", "volatility", "max_drawdown")


class ContextError(RuntimeError):
    """A recipe could not be read, validated, or rebuilt."""


# ── locating the context ──────────────────────────────────────────────────────


def find_root(start: Path | str | None = None) -> Path:
    """Locate the directory that owns (or should own) the context files.

    Walks upward looking for an existing ``.jquantstats`` directory first, then
    for a repository marker, so the same context is found from any subdirectory.

    Args:
        start: Directory to search from. Defaults to the current directory.

    Returns:
        The resolved project root.
    """
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    candidates = [here, *here.parents]
    for parent in candidates:
        if (parent / CONTEXT_DIRNAME / CONTEXT_FILENAME).exists():
            return parent
    for parent in candidates:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return here


def context_path(root: Path) -> Path:
    """Return the path of the recipe file under *root*."""
    return root / CONTEXT_DIRNAME / CONTEXT_FILENAME


def digest_path(root: Path) -> Path:
    """Return the path of the digest file under *root*."""
    return root / CONTEXT_DIRNAME / DIGEST_FILENAME


# ── reading and writing the recipe ────────────────────────────────────────────


def _ensure_dir(root: Path) -> Path:
    """Create ``.jquantstats/`` and mark the derived files as not-for-commit.

    The recipe is worth committing; the digest is derived and goes stale the
    moment an input changes. A self-contained ignore file keeps that distinction
    without needing to edit the host project's ``.gitignore``.

    Args:
        root: Project root.

    Returns:
        The context directory.
    """
    directory = root / CONTEXT_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    ignore = directory / ".gitignore"
    if not ignore.exists():
        ignore.write_text(
            f"# Derived from context.json and the input files — regenerate, don't commit.\n{DIGEST_FILENAME}\n",
            encoding="utf-8",
        )
    return directory


def read_container(root: Path) -> dict[str, Any]:
    """Read the whole recipe file, which may hold several named contexts.

    Args:
        root: Project root holding ``.jquantstats/``.

    Returns:
        A mapping with ``schema``, ``active`` and ``contexts`` keys.

    Raises:
        ContextError: If the file is missing, malformed, or a newer schema.
    """
    path = context_path(root)
    if not path.exists():
        raise ContextError(f"no context at {path} — run jqs_load.py to create one")
    try:
        container = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContextError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(container, dict) or "contexts" not in container:
        raise ContextError(f"{path} has no 'contexts' block")
    schema = container.get("schema", 0)
    if schema > SCHEMA:
        raise ContextError(f"{path} uses schema {schema}; this plugin understands {SCHEMA}")
    return container


def resolve(root: Path, name: str | None = None) -> tuple[str, dict[str, Any]]:
    """Pick one recipe out of the container.

    Args:
        root: Project root.
        name: Context name. Defaults to the container's ``active`` entry.

    Returns:
        The ``(name, recipe)`` pair.

    Raises:
        ContextError: If the requested name is not present.
    """
    container = read_container(root)
    contexts = container["contexts"]
    chosen = name or container.get("active")
    if chosen is None and len(contexts) == 1:
        chosen = next(iter(contexts))
    if not isinstance(chosen, str) or chosen not in contexts:
        known = ", ".join(sorted(contexts)) or "none"
        raise ContextError(f"unknown context {chosen!r} (known: {known})")
    return chosen, contexts[chosen]


def upsert(root: Path, recipe: dict[str, Any], *, activate: bool = True) -> Path:
    """Write *recipe* into the container under its own name.

    Args:
        root: Project root.
        recipe: A recipe carrying a ``name`` key.
        activate: Whether to make it the active context.

    Returns:
        The path written.
    """
    path = context_path(root)
    _ensure_dir(root)
    container: dict[str, Any] = {"schema": SCHEMA, "active": None, "contexts": {}}
    if path.exists():
        # a corrupt file is replaced rather than blocking a fresh load
        with contextlib.suppress(ContextError):
            container = read_container(root)
    name = recipe["name"]
    container["schema"] = SCHEMA
    container["contexts"][name] = recipe
    if activate or container.get("active") is None:
        container["active"] = name
    path.write_text(json.dumps(container, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_digests(root: Path) -> dict[str, dict[str, Any]]:
    """Read every stored digest, keyed by context name.

    Args:
        root: Project root.

    Returns:
        Mapping of context name to digest, empty when no digest file exists.
    """
    path = digest_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("digests", {})
    except json.JSONDecodeError:
        return {}


def write_digest(root: Path, payload: dict[str, Any]) -> Path:
    """Merge a digest into the digest file under its context's name.

    Keyed by context so building a variant cannot overwrite — and silently
    invalidate the freshness check of — a sibling context's fingerprints.

    Args:
        root: Project root.
        payload: The digest mapping, carrying a ``context`` key.

    Returns:
        The path written.
    """
    path = digest_path(root)
    _ensure_dir(root)
    digests = read_digests(root)
    digests[payload.get("context") or "base"] = payload
    path.write_text(json.dumps({"schema": SCHEMA, "digests": digests}, indent=2) + "\n", encoding="utf-8")
    return path


# ── inputs ────────────────────────────────────────────────────────────────────


def _read_table(path: Path) -> pl.DataFrame:
    """Read a tabular file, dispatching on its suffix.

    Args:
        path: File to read.

    Returns:
        The frame.

    Raises:
        ContextError: If the suffix is not a supported table format.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".ipc", ".arrow", ".feather"}:
        return pl.read_ipc(path)
    if suffix == ".ndjson":
        return pl.read_ndjson(path)
    if suffix == ".json":
        return pl.read_json(path)
    raise ContextError(f"unsupported table format {suffix!r} for {path}")


def _from_script(spec: dict[str, Any], root: Path) -> pl.DataFrame:
    """Build a frame by calling a function in a repo-owned Python file.

    This is the escape hatch for inputs a path cannot express — positions
    derived from an expression, a query, or a model. The logic stays in a
    normal reviewable ``.py`` file rather than as source embedded in JSON.

    Args:
        spec: Input spec with ``script`` and optional ``callable`` keys.
        root: Project root that ``script`` is relative to.

    Returns:
        The frame returned by the callable.

    Raises:
        ContextError: If the module, callable, or return type is wrong.
    """
    script = (root / spec["script"]).resolve()
    func_name = spec.get("callable", "build")
    if not script.exists():
        raise ContextError(f"builder script not found: {script}")
    module_spec = importlib.util.spec_from_file_location(f"_jqs_builder_{script.stem}", script)
    if module_spec is None or module_spec.loader is None:
        raise ContextError(f"cannot import builder script: {script}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    func = getattr(module, func_name, None)
    if func is None:
        raise ContextError(f"{script} has no callable named {func_name!r}")
    frame = func()
    if not isinstance(frame, pl.DataFrame):
        raise ContextError(f"{script}:{func_name} returned {type(frame).__name__}, expected pl.DataFrame")
    return frame


def read_input(spec: dict[str, Any], root: Path) -> pl.DataFrame:
    """Materialise one recipe input.

    Args:
        spec: Input spec — either ``{"path": ...}`` or
            ``{"script": ..., "callable": ...}``, optionally with ``columns``.
        root: Project root that relative paths resolve against.

    Returns:
        The frame, restricted to ``columns`` when given.

    Raises:
        ContextError: If the spec names neither a path nor a script.
    """
    if "path" in spec:
        path = (root / spec["path"]).resolve()
        if not path.exists():
            raise ContextError(f"input file not found: {path}")
        frame = _read_table(path)
    elif "script" in spec:
        frame = _from_script(spec, root)
    else:
        raise ContextError(f"input spec needs a 'path' or a 'script': {spec}")

    columns = spec.get("columns")
    if columns:
        date_col = spec.get("date_col") or frame.columns[0]
        keep = [date_col, *[c for c in columns if c != date_col]]
        missing = [c for c in keep if c not in frame.columns]
        if missing:
            raise ContextError(f"columns {missing} not in {spec} (have {frame.columns})")
        frame = frame.select(keep)
    return frame


def _put_date_first(frame: pl.DataFrame, date_col: str | None, target: str) -> tuple[pl.DataFrame, str | None]:
    """Move the date column to position 0 and rename it to *target*.

    Args:
        frame: Source frame.
        date_col: Name of the date column, or None to use the first column.
        target: Name the date column must end up with.

    Returns:
        The reshaped frame and the original column name if it was renamed,
        otherwise None.
    """
    source = date_col or frame.columns[0]
    if source not in frame.columns:
        raise ContextError(f"date column {source!r} not in {frame.columns}")
    ordered = frame.select([source, *[c for c in frame.columns if c != source]])
    if source == target:
        return ordered, None
    return ordered.rename({source: target}), source


# ── building ──────────────────────────────────────────────────────────────────


def _cost_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    """Translate recipe cost arguments into constructor keywords.

    ``cost_model`` is recorded declaratively as ``{"kind": ..., "value": ...}``
    because the object itself is not a dataclass field on Portfolio.

    Args:
        args: The recipe's ``args`` block.

    Returns:
        Keyword arguments for a Portfolio constructor.
    """
    from jquantstats import CostModel

    out: dict[str, Any] = {}
    for key in ("cost_per_unit", "cost_bps", "annual_fee"):
        if key in args:
            out[key] = float(args[key])
    model = args.get("cost_model")
    if model:
        kind, value = model["kind"], float(model["value"])
        if kind == "per_unit":
            out["cost_model"] = CostModel.per_unit(value)
        elif kind == "turnover_bps":
            out["cost_model"] = CostModel.turnover_bps(value)
        else:
            raise ContextError(f"unknown cost_model kind {kind!r} (per_unit | turnover_bps)")
    return out


def _coerce_bound(value: Any) -> Any:
    """Turn a JSON truncate bound into something the library accepts.

    ``truncate`` advertises ``str`` in its signature but compares the bound
    against a temporal column without parsing it, so an ISO string raises
    ``InvalidOperationError``. JSON has no date type, so the conversion has to
    happen here.

    Args:
        value: A row index, an ISO date/datetime string, or None.

    Returns:
        An int, a ``date``, a ``datetime``, or None.

    Raises:
        ContextError: If a string is not ISO-8601.
    """
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    if isinstance(value, str):
        for parse in (date.fromisoformat, datetime.fromisoformat):
            try:
                return parse(value)
            except ValueError:
                continue
        raise ContextError(f"truncate bound {value!r} is not an ISO-8601 date, datetime, or row index")
    return value


def _apply_post(obj: Any, post: list[dict[str, Any]]) -> Any:
    """Replay the recorded post-construction transforms, in order.

    Without this a rebuild silently produces a *different* portfolio — a
    ``lag(1)`` study reconstructed unlagged looks entirely plausible.

    Args:
        obj: The freshly constructed Portfolio or Data.
        post: Ordered list of ``{"op": ..., ...}`` transforms.

    Returns:
        The transformed object.

    Raises:
        ContextError: If an op is unknown or unsupported for this type.
    """
    for step in post:
        op = step.get("op")
        if op == "lag":
            obj = _require(obj, "lag", op)(int(step["n"]))
        elif op == "truncate":
            method = _require(obj, "truncate", op)
            obj = method(start=_coerce_bound(step.get("start")), end=_coerce_bound(step.get("end")))
        elif op == "resample":
            obj = _require(obj, "resample", op)(step.get("every", "1mo"))
        else:
            raise ContextError(f"unknown post op {op!r} (lag | truncate | resample)")
    return obj


def _require(obj: Any, attr: str, op: str) -> Any:
    """Fetch a transform method, with a message naming the type that lacks it.

    Args:
        obj: Object to look on.
        attr: Method name.
        op: The recipe op requesting it.

    Returns:
        The bound method.

    Raises:
        ContextError: If the object has no such method.
    """
    method = getattr(obj, attr, None)
    if method is None:
        raise ContextError(f"post op {op!r} is not supported by {type(obj).__name__}")
    return method


def build_detailed(recipe: dict[str, Any], root: Path) -> tuple[Any, dict[str, pl.DataFrame], list[str]]:
    """Rebuild the object and keep the raw inputs for inspection.

    Args:
        recipe: A single recipe.
        root: Project root that relative paths resolve against.

    Returns:
        A ``(object, frames, notes)`` triple: the Portfolio or Data, the input
        frames keyed by role, and human-readable notes about any normalisation
        applied on the way in.

    Raises:
        ContextError: If the recipe is inconsistent with the library API.
    """
    entry = recipe.get("entry_point")
    ctor = recipe.get("constructor")
    inputs = recipe.get("inputs", {})
    args = dict(recipe.get("args", {}))
    notes: list[str] = []
    frames: dict[str, pl.DataFrame] = {}

    if entry == "Portfolio":
        from jquantstats import Portfolio

        if ctor not in PORTFOLIO_CONSTRUCTORS:
            raise ContextError(f"unknown Portfolio constructor {ctor!r} ({', '.join(PORTFOLIO_CONSTRUCTORS)})")
        pos_role = PORTFOLIO_CONSTRUCTORS[ctor]
        for role in ("prices", pos_role):
            if role not in inputs:
                raise ContextError(f"{ctor} needs an input named {role!r}")
            frame, renamed = _put_date_first(
                read_input(inputs[role], root), inputs[role].get("date_col"), PORTFOLIO_DATE_COL
            )
            if renamed:
                notes.append(
                    f"{role}: date column {renamed!r} renamed to 'date' — the Portfolio/Data bridge "
                    f"keeps the date axis only under that name"
                )
            frames[role] = frame
        kwargs: dict[str, Any] = {"aum": float(args["aum"])}
        if ctor == "from_risk_position":
            if "vola" in args:
                kwargs["vola"] = args["vola"]
            if args.get("vol_cap") is not None:
                kwargs["vol_cap"] = float(args["vol_cap"])
        kwargs.update(_cost_kwargs(args))
        obj = getattr(Portfolio, ctor)(prices=frames["prices"], **{pos_role: frames[pos_role]}, **kwargs)

    elif entry == "Data":
        from jquantstats import Data

        if ctor not in DATA_CONSTRUCTORS:
            raise ContextError(f"unknown Data constructor {ctor!r} ({', '.join(DATA_CONSTRUCTORS)})")
        role = DATA_CONSTRUCTORS[ctor]
        if role not in inputs:
            raise ContextError(f"{ctor} needs an input named {role!r}")
        frames[role] = read_input(inputs[role], root)
        date_col = inputs[role].get("date_col") or frames[role].columns[0]
        kwargs = {"date_col": date_col}
        if args.get("null_strategy"):
            kwargs["null_strategy"] = args["null_strategy"]
        if "benchmark" in inputs:
            frames["benchmark"] = read_input(inputs["benchmark"], root)
            kwargs["benchmark"] = frames["benchmark"]
        rf = args.get("rf")
        if isinstance(rf, dict):
            frames["rf"] = read_input(rf, root)
            kwargs["rf"] = frames["rf"]
        elif rf is not None:
            kwargs["rf"] = float(rf)
        obj = getattr(Data, ctor)(**{role: frames[role]}, **kwargs)

    else:
        raise ContextError(f"entry_point must be 'Portfolio' or 'Data', got {entry!r}")

    return _apply_post(obj, recipe.get("post", [])), frames, notes


def build(recipe: dict[str, Any], root: Path | None = None) -> Any:
    """Rebuild the object described by *recipe*.

    Args:
        recipe: A single recipe.
        root: Project root. Defaults to :func:`find_root`.

    Returns:
        The Portfolio or Data instance.
    """
    obj, _, _ = build_detailed(recipe, root or find_root())
    return obj


def load(name: str | None = None, root: Path | str | None = None) -> Any:
    """Rebuild the active (or named) context from disk.

    This is the entry point analysis snippets should use.

    Args:
        name: Context name. Defaults to the active one.
        root: Project root. Defaults to :func:`find_root`.

    Returns:
        The Portfolio or Data instance.
    """
    resolved = find_root(root)
    _, recipe = resolve(resolved, name)
    return build(recipe, resolved)


# ── fingerprints ──────────────────────────────────────────────────────────────


def fingerprint(path: Path) -> dict[str, Any]:
    """Hash a file so a digest can tell whether its inputs still match.

    Args:
        path: File to hash.

    Returns:
        A mapping with ``sha256`` and ``bytes``.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def fingerprints(recipe: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    """Fingerprint every file-backed input of *recipe*.

    Args:
        recipe: A single recipe.
        root: Project root.

    Returns:
        Mapping of relative path to fingerprint.
    """
    out: dict[str, dict[str, Any]] = {}
    specs = list(recipe.get("inputs", {}).values())
    rf = recipe.get("args", {}).get("rf")
    if isinstance(rf, dict):
        specs.append(rf)
    for spec in specs:
        for key in ("path", "script"):
            rel = spec.get(key)
            if rel:
                path = (root / rel).resolve()
                if path.exists():
                    out[rel] = fingerprint(path)
    return out


def stale_inputs(root: Path | str | None = None, name: str | None = None) -> list[str]:
    """List inputs whose content no longer matches a stored digest.

    A non-empty result means that digest's numbers must not be quoted;
    recompute instead.

    Args:
        root: Project root. Defaults to :func:`find_root`.
        name: Context to check. Defaults to the active one.

    Returns:
        Relative paths that changed or went missing.
    """
    resolved = find_root(root)
    digests = read_digests(resolved)
    if not digests:
        return []
    chosen = name or read_container(resolved).get("active")
    if not isinstance(chosen, str):
        return []
    stored = digests.get(chosen, {}).get("fingerprints", {})
    changed = []
    for rel, expected in stored.items():
        candidate = (resolved / rel).resolve()
        if not candidate.exists() or fingerprint(candidate) != expected:
            changed.append(rel)
    return changed


# ── digest ────────────────────────────────────────────────────────────────────


def _stats_of(obj: Any) -> Any:
    """Return the stats accessor for a Portfolio or Data."""
    return obj.stats


def _index_frame(obj: Any) -> pl.DataFrame:
    """Return the date/index frame behind a Portfolio or Data."""
    return obj.data.index if hasattr(obj, "cashposition") else obj.index


def _describe_rows(obj: Any) -> list[dict[str, Any]]:
    """Convert ``describe()`` into JSON-safe rows.

    Args:
        obj: Portfolio or Data.

    Returns:
        One dict per asset, with dates stringified.
    """
    rows = obj.describe().to_dicts()
    return [{k: (str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()} for row in rows]


def _null_counts(frames: dict[str, pl.DataFrame]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Count nulls and NaNs per column of every input frame.

    Polars keeps ``null`` (missing) and ``NaN`` (IEEE-754) distinct, and
    ``null_strategy`` acts on nulls only — so the two are reported separately.

    Args:
        frames: Input frames keyed by role.

    Returns:
        A ``(nulls, nans)`` pair, each mapping role to non-zero column counts.
    """
    nulls: dict[str, dict[str, int]] = {}
    nans: dict[str, dict[str, int]] = {}
    for role, frame in frames.items():
        counts = {c: int(n) for c, n in zip(frame.columns, frame.null_count().row(0), strict=True) if n}
        if counts:
            nulls[role] = counts
        nan_counts = {}
        for name, dtype in zip(frame.columns, frame.dtypes, strict=True):
            if dtype.is_float():
                total = int(frame[name].is_nan().sum() or 0)
                if total:
                    nan_counts[name] = total
        if nan_counts:
            nans[role] = nan_counts
    return nulls, nans


def _anchor_metrics(obj: Any) -> dict[str, Any]:
    """Compute a handful of metrics as a tripwire for later mistakes.

    These are not a cache: they are five numbers cheap to re-derive, so a
    rebuild that disagrees with them is visibly wrong.

    Args:
        obj: Portfolio or Data.

    Returns:
        Mapping of metric name to its per-column result, or to an error string.
    """
    stats = _stats_of(obj)
    out: dict[str, Any] = {}
    for name in _ANCHOR_METRICS:
        try:
            value = getattr(stats, name)()
        except Exception as exc:  # a null-poisoned frame must not abort the digest
            out[name] = f"error: {type(exc).__name__}: {exc}"
            continue
        out[name] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in value.items()}
    return out


def _empirical_periods_per_year(index: pl.DataFrame, rows: int) -> float | None:
    """Estimate observations per year from the calendar span.

    Uses rows-per-year rather than the median step, so business-daily (252),
    weekly (52) and monthly (12) data are all estimated correctly.

    Args:
        index: The date/index frame.
        rows: Number of observations.

    Returns:
        The estimate, or None if the axis is not temporal or spans too little.
    """
    column = index.columns[0]
    if not index[column].dtype.is_temporal() or rows < 2:
        return None
    first, last = index[column].min(), index[column].max()
    # is_temporal() also covers Time and Duration, which do not subtract into days
    if not isinstance(first, date) or not isinstance(last, date) or type(first) is not type(last):
        return None
    days = (last - first).days
    if days < 180:
        return None
    return rows / (days / 365.25)


def _warnings(obj: Any, frames: dict[str, pl.DataFrame], recipe: dict[str, Any], shape: dict[str, Any]) -> list[str]:
    """Derive the findings a diagnostics pass would otherwise have to rediscover.

    Args:
        obj: Portfolio or Data.
        frames: Input frames keyed by role.
        recipe: The recipe used.
        shape: The digest's ``shape`` block.

    Returns:
        Human-readable warnings, most consequential first.
    """
    out: list[str] = []
    nulls, nans = _null_counts(frames)
    if nulls and not recipe.get("args", {}).get("null_strategy"):
        detail = "; ".join(f"{role}.{col}={n}" for role, cols in nulls.items() for col, n in cols.items())
        out.append(
            f"nulls present ({detail}) and null_strategy is unset — handling then varies by metric: "
            f"aggregations skip nulls (shrinking the effective sample silently) while cumulative and "
            f"rolling paths can propagate them. Set null_strategy='drop' to make the sample explicit "
            f"and to match pandas."
        )
    if nans:
        detail = "; ".join(f"{role}.{col}={n}" for role, cols in nans.items() for col, n in cols.items())
        out.append(f"NaN values present ({detail}) — null_strategy does not touch NaN, only null")

    if not shape.get("date_axis_temporal", True):
        out.append(
            "the object's index is positional, not temporal — annualised metrics fall back to a "
            "default periods_per_year and date-axis plots lose their x-axis"
        )

    reported, empirical = shape.get("periods_per_year"), shape.get("periods_per_year_empirical")
    if reported and empirical and not 0.8 <= reported / empirical <= 1.25:
        out.append(
            f"periods_per_year is {reported:g} but the data spans {empirical:.1f} observations per year — "
            f"sharpe, sortino and volatility are annualised by the former and may be off by "
            f"~{(reported / empirical) ** 0.5:.2f}x"
        )

    benchmark = shape.get("benchmark")
    if benchmark and benchmark.get("overlap_rows") is not None:
        rows = shape.get("rows") or 0
        if benchmark["overlap_rows"] < rows:
            out.append(
                f"benchmark overlaps only {benchmark['overlap_rows']} of {rows} rows — "
                f"beta, alpha, r_squared and capture ratios use the overlap only"
            )
    return out


def digest(recipe: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Rebuild the context and describe what was built.

    Args:
        recipe: A single recipe.
        root: Project root. Defaults to :func:`find_root`.

    Returns:
        The digest mapping: fingerprints, shape, nulls, anchors, warnings.
    """
    resolved = root or find_root()
    obj, frames, notes = build_detailed(recipe, resolved)
    index = _index_frame(obj)
    rows = index.height
    stats = _stats_of(obj)
    try:
        reported = float(stats.periods_per_year)
    except Exception:  # inference can fail on a degenerate axis
        reported = None

    date_column = index.columns[0]
    temporal = index[date_column].dtype.is_temporal()
    shape: dict[str, Any] = {
        "assets": list(obj.assets),
        "rows": rows,
        "date_column": date_column,
        "date_axis_temporal": temporal,
        "start": str(index[date_column].min()) if rows else None,
        "end": str(index[date_column].max()) if rows else None,
        "periods_per_year": reported,
        "periods_per_year_empirical": _empirical_periods_per_year(index, rows),
        "per_asset": _describe_rows(obj),
    }
    if getattr(obj, "benchmark", None) is not None:
        bench = obj.benchmark
        shape["benchmark"] = {"columns": list(bench.columns), "overlap_rows": bench.height}

    nulls, nans = _null_counts(frames)
    anchors = _anchor_metrics(obj)
    if hasattr(obj, "turnover_summary"):
        anchors["turnover"] = {
            row["metric"]: round(float(row["value"]), 6)
            for row in obj.turnover_summary().to_dicts()
            if row["value"] is not None
        }

    return {
        "schema": SCHEMA,
        "context": recipe.get("name"),
        "built": f"{recipe.get('entry_point')}.{recipe.get('constructor')}",
        "post": recipe.get("post", []),
        "fingerprints": fingerprints(recipe, resolved),
        "shape": shape,
        "nulls": nulls,
        "nans": nans,
        "anchors": anchors,
        "notes": notes,
        "warnings": _warnings(obj, frames, recipe, shape),
    }
