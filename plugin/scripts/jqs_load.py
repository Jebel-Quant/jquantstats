#!/usr/bin/env python3
"""Build a jquantstats portfolio context from the command line.

Records a *recipe* in ``.jquantstats/context.json`` and prints a digest of what
was built, so later analysis snippets rebuild the same object with
``from jqs_context import load``.

Examples::

    # Portfolio from prices + cash positions, 5bp turnover cost, one-day delay
    jqs_load.py portfolio --prices data/prices.csv --cash-position data/pos.csv \
        --aum 1e6 --cost-bps 5 --lag 1

    # A return series with a benchmark, matching pandas null handling
    jqs_load.py data --returns data/returns.csv --benchmark data/spy.csv \
        --null-strategy drop

    jqs_load.py show          # rebuild the active context and re-print the digest
    jqs_load.py list          # every recorded context
    jqs_load.py check         # do the stored digest's numbers still hold?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jqs_context import (
    ContextError,
    digest,
    find_root,
    read_container,
    read_digests,
    resolve,
    stale_inputs,
    upsert,
    write_digest,
)


def _fmt(value: Any) -> str:
    """Format a scalar for the text digest.

    Args:
        value: Any digest value.

    Returns:
        A compact string.
    """
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4g}"
    return str(value)


def _render_anchors(anchors: dict[str, Any]) -> list[str]:
    """Render the anchor metrics, inline for one series and as a table for many.

    Args:
        anchors: The digest's ``anchors`` block.

    Returns:
        Lines of output.
    """
    lines: list[str] = []
    columns: list[str] = []
    for value in anchors.values():
        if isinstance(value, dict):
            columns = [c for c in value if c not in columns] or columns
            break
    for name, value in anchors.items():
        if isinstance(value, str):
            lines.append(f"  {name:<24} {value}")
        elif len(value) == 1:
            lines.append(f"  {name:<24} {_fmt(next(iter(value.values())))}")
        else:
            cells = "  ".join(f"{k}={_fmt(v)}" for k, v in value.items())
            lines.append(f"  {name:<24} {cells}")
    return lines


def render(name: str, recipe: dict[str, Any], payload: dict[str, Any], root: Path) -> str:
    """Render the digest as the compact text block that lands in the transcript.

    Args:
        name: Context name.
        recipe: The recipe used.
        payload: The computed digest.
        root: Project root, shown so paths are unambiguous.

    Returns:
        The rendered block.
    """
    shape = payload["shape"]
    lines = [
        f"context     {name}",
        f"built       {payload['built']}",
        f"root        {root}",
    ]

    for role, spec in recipe.get("inputs", {}).items():
        source = spec.get("path") or f"{spec.get('script')}:{spec.get('callable', 'build')}"
        suffix = f"  (date_col={spec['date_col']})" if spec.get("date_col") else ""
        lines.append(f"  {role:<24} {source}{suffix}")

    args = recipe.get("args", {})
    if args:
        lines.append("args        " + "  ".join(f"{k}={_fmt(v)}" for k, v in args.items() if v is not None))
    if recipe.get("post"):
        steps = "  ".join(
            f"{s['op']}({', '.join(f'{k}={v}' for k, v in s.items() if k != 'op')})" for s in recipe["post"]
        )
        lines.append(f"post        {steps}")

    assets = shape["assets"]
    shown = ", ".join(assets[:8]) + (f", … (+{len(assets) - 8})" if len(assets) > 8 else "")
    lines.append(f"assets      {len(assets)}  [{shown}]")
    axis = shape["date_column"] if shape["date_axis_temporal"] else f"{shape['date_column']} (positional!)"
    lines.append(f"span        {shape['start']} → {shape['end']}   {shape['rows']} rows   on '{axis}'")
    ppy, empirical = shape["periods_per_year"], shape["periods_per_year_empirical"]
    ppy_line = f"periods/yr  {_fmt(ppy)}"
    if empirical:
        ppy_line += f"   (empirical {_fmt(empirical)})"
    lines.append(ppy_line)
    if shape.get("benchmark"):
        bench = shape["benchmark"]
        lines.append(f"benchmark   {', '.join(bench['columns'])}   {bench['overlap_rows']} rows")

    for label, block in (("nulls", payload["nulls"]), ("NaNs", payload["nans"])):
        if block:
            detail = "  ".join(f"{role}.{col}={n}" for role, cols in block.items() for col, n in cols.items())
            lines.append(f"{label:<11} {detail}")

    lines.append("anchors")
    lines.extend(_render_anchors(payload["anchors"]))

    for note in payload.get("notes", []):
        lines.append(f"note        {note}")
    for warning in payload.get("warnings", []):
        lines.append(f"WARNING     {warning}")

    return "\n".join(lines)


def _input_spec(path: str | None, date_col: str | None, columns: list[str] | None) -> dict[str, Any]:
    """Assemble one recipe input spec.

    Args:
        path: File path, relative to the project root when possible.
        date_col: Explicit date column name, or None to use the first column.
        columns: Subset of asset columns to keep, or None for all.

    Returns:
        The spec mapping.
    """
    spec: dict[str, Any] = {"path": path}
    if date_col:
        spec["date_col"] = date_col
    if columns:
        spec["columns"] = columns
    return spec


def _relative(path: str, root: Path) -> str:
    r"""Make *path* relative to *root* when it lives inside it.

    Always POSIX-separated. The recipe is meant to be committed and read on
    other machines, and a Windows-authored ``data\prices.csv`` would neither
    resolve elsewhere nor read cleanly in JSON, which escapes the backslash.
    Forward slashes are accepted by pathlib on every platform.

    Args:
        path: A user-supplied path.
        root: Project root.

    Returns:
        A root-relative POSIX path where possible, else the resolved absolute one.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _post_ops(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Collect post-construction transforms from the parsed arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Ordered transform steps.
    """
    post: list[dict[str, Any]] = []
    if getattr(args, "lag", None):
        post.append({"op": "lag", "n": args.lag})
    if getattr(args, "start", None) or getattr(args, "end", None):
        step: dict[str, Any] = {"op": "truncate"}
        if args.start:
            step["start"] = args.start
        if args.end:
            step["end"] = args.end
        post.append(step)
    if getattr(args, "resample", None):
        post.append({"op": "resample", "every": args.resample})
    return post


def _portfolio_recipe(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Build a Portfolio recipe from the parsed arguments.

    Args:
        args: Parsed CLI arguments.
        root: Project root.

    Returns:
        The recipe.

    Raises:
        SystemExit: If no position input was given.
    """
    pairs = [
        ("from_cash_position", "cash_position", args.cash_position),
        ("from_position", "position", args.position),
        ("from_risk_position", "risk_position", args.risk_position),
    ]
    chosen = [(c, role, path) for c, role, path in pairs if path]
    if len(chosen) != 1:
        raise SystemExit("give exactly one of --cash-position / --position / --risk-position")
    ctor, role, path = chosen[0]

    recipe_args: dict[str, Any] = {"aum": args.aum}
    if ctor == "from_risk_position":
        recipe_args["vola"] = args.vola
        if args.vol_cap is not None:
            recipe_args["vol_cap"] = args.vol_cap
    for key in ("cost_per_unit", "cost_bps", "annual_fee"):
        value = getattr(args, key)
        if value:
            recipe_args[key] = value

    return {
        "name": args.name,
        "entry_point": "Portfolio",
        "constructor": ctor,
        "inputs": {
            "prices": _input_spec(_relative(args.prices, root), args.date_col, args.columns),
            role: _input_spec(_relative(path, root), args.date_col, args.columns),
        },
        "args": recipe_args,
        "post": _post_ops(args),
    }


def _data_recipe(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Build a Data recipe from the parsed arguments.

    Args:
        args: Parsed CLI arguments.
        root: Project root.

    Returns:
        The recipe.

    Raises:
        SystemExit: If neither or both of --returns/--prices were given.
    """
    if bool(args.returns) == bool(args.prices):
        raise SystemExit("give exactly one of --returns / --prices")
    ctor = "from_returns" if args.returns else "from_prices"
    role = "returns" if args.returns else "prices"
    source = args.returns or args.prices

    inputs = {role: _input_spec(_relative(source, root), args.date_col, args.columns)}
    if args.benchmark:
        inputs["benchmark"] = _input_spec(_relative(args.benchmark, root), args.date_col, None)

    recipe_args: dict[str, Any] = {}
    if args.null_strategy:
        recipe_args["null_strategy"] = args.null_strategy
    if args.rf:
        try:
            recipe_args["rf"] = float(args.rf)
        except ValueError:
            recipe_args["rf"] = _input_spec(_relative(args.rf, root), args.date_col, None)

    return {
        "name": args.name,
        "entry_point": "Data",
        "constructor": ctor,
        "inputs": inputs,
        "args": recipe_args,
        "post": _post_ops(args),
    }


def _emit(name: str, recipe: dict[str, Any], root: Path, args: argparse.Namespace) -> int:
    """Compute the digest, print it, and persist both files.

    Args:
        name: Context name.
        recipe: The recipe.
        root: Project root.
        args: Parsed CLI arguments (for ``--json`` / ``--no-write``).

    Returns:
        A process exit code.
    """
    payload = digest(recipe, root)
    if args.json:
        print(json.dumps({"context": name, "recipe": recipe, "digest": payload}, indent=2))
    else:
        print(render(name, recipe, payload, root))
    if not args.no_write:
        written = [upsert(root, recipe), write_digest(root, payload)]
        if not args.json:
            print("wrote       " + ", ".join(str(p.relative_to(root)) for p in written))
    return 1 if payload["warnings"] else 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Attach arguments shared by the build subcommands.

    Args:
        parser: The subparser to extend.
    """
    parser.add_argument("--name", default="base", help="context name, for keeping variants side by side")
    parser.add_argument("--date-col", help="date column name (default: the first column)")
    parser.add_argument("--columns", nargs="+", help="restrict to these asset columns")
    parser.add_argument("--start", help="truncate from this date")
    parser.add_argument("--end", help="truncate to this date")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of the text digest")
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch .jquantstats/")
    parser.add_argument("--root", help="project root (default: nearest .jquantstats/, .git or pyproject.toml)")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(prog="jqs_load.py", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    portfolio = subparsers.add_parser("portfolio", help="prices + positions (turnover, costs, lag, attribution)")
    portfolio.add_argument("--prices", required=True, help="price frame")
    portfolio.add_argument("--cash-position", help="positions in currency")
    portfolio.add_argument("--position", help="positions in share counts")
    portfolio.add_argument("--risk-position", help="positions in risk units")
    portfolio.add_argument("--aum", type=float, required=True, help="assets under management")
    portfolio.add_argument("--vola", type=int, default=32, help="EWMA lookback in periods (--risk-position only)")
    portfolio.add_argument("--vol-cap", type=float, help="volatility cap (--risk-position only)")
    portfolio.add_argument("--cost-per-unit", type=float, default=0.0, help="currency cost per unit traded")
    portfolio.add_argument("--cost-bps", type=float, default=0.0, help="bps of AUM turnover")
    portfolio.add_argument("--annual-fee", type=float, default=0.0, help="annual management fee")
    portfolio.add_argument("--lag", type=int, help="shift positions by n periods (execution delay)")
    _add_common(portfolio)

    data = subparsers.add_parser("data", help="a return or price series (the QuantStats-equivalent route)")
    data.add_argument("--returns", help="return frame")
    data.add_argument("--prices", help="price frame")
    data.add_argument("--benchmark", help="benchmark frame, passed once at construction")
    data.add_argument("--rf", help="risk-free rate: a number, or a path to a frame")
    data.add_argument("--null-strategy", choices=["raise", "drop", "forward_fill"], help="use 'drop' to match pandas")
    data.add_argument("--resample", help="resample the series, e.g. 1mo")
    _add_common(data)

    show = subparsers.add_parser("show", help="rebuild a recorded context and re-print its digest")
    show.add_argument("--name", help="context name (default: the active one)")
    show.add_argument("--json", action="store_true", help="emit JSON")
    show.add_argument("--no-write", action="store_true", help="do not refresh digest.json")
    show.add_argument("--root", help="project root")

    listing = subparsers.add_parser("list", help="list recorded contexts")
    listing.add_argument("--root", help="project root")

    activate = subparsers.add_parser("activate", help="make a recorded context the active one")
    activate.add_argument("name", help="context to activate")
    activate.add_argument("--root", help="project root")

    check = subparsers.add_parser("check", help="report inputs that changed since the digest was written")
    check.add_argument("--root", help="project root")
    return parser


def _check(root: Path) -> int:
    """Report, per recorded context, whether its inputs still match its digest.

    Args:
        root: Project root.

    Returns:
        0 when everything is fresh, 1 when any context is stale.
    """
    digests = read_digests(root)
    if not digests:
        print("no digest recorded yet — run 'portfolio', 'data' or 'show' first")
        return 0
    stale_names = []
    for name in digests:
        stale = stale_inputs(root, name)
        if stale:
            stale_names.append(name)
            print(f"STALE   {name:<16} {', '.join(stale)}")
        else:
            print(f"fresh   {name:<16} inputs match the recorded fingerprints")
    if stale_names:
        print("recompute the stale contexts before quoting any number from their digests")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 clean, 1 built with warnings, 2 on error.
    """
    args = build_parser().parse_args(argv)
    root = find_root(args.root)

    try:
        if args.command == "portfolio":
            recipe = _portfolio_recipe(args, root)
            return _emit(recipe["name"], recipe, root, args)
        if args.command == "data":
            recipe = _data_recipe(args, root)
            return _emit(recipe["name"], recipe, root, args)
        if args.command == "show":
            name, recipe = resolve(root, args.name)
            return _emit(name, recipe, root, args)
        if args.command == "list":
            container = read_container(root)
            active = container.get("active")
            for name, recipe in container["contexts"].items():
                marker = "*" if name == active else " "
                post = ",".join(step["op"] for step in recipe.get("post", [])) or "-"
                print(f"{marker} {name:<16} {recipe['entry_point']}.{recipe['constructor']:<20} post={post}")
            return 0
        if args.command == "activate":
            name, recipe = resolve(root, args.name)
            upsert(root, recipe, activate=True)
            print(f"active context is now {name}")
            return 0
        return _check(root)
    except ContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, TypeError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
