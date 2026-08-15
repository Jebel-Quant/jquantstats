#!/usr/bin/env python3
"""Report the jquantstats API surface as it actually is in the installed version.

The library's metric names move, and hand-maintained mapping tables drift out of
date silently — an invented method that "looks right" is the most expensive kind
of mistake here. Ask this script instead of recalling.

Examples::

    jqs_api.py                       # section counts
    jqs_api.py stats                 # every Stats method
    jqs_api.py --grep drawdown       # find a name across the whole surface
    jqs_api.py --show sharpe         # signature and docstring
    jqs_api.py --show conditional_value_at_risk
    jqs_api.py constructors          # how to build a Portfolio or Data
"""

from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

SECTION_ORDER = ("stats", "data-plots", "portfolio-plots", "portfolio", "data", "constructors")


def _sections() -> dict[str, Any]:
    """Map section names to the classes they describe.

    Returns:
        Mapping of section name to class object.
    """
    from jquantstats import Data, Portfolio
    from jquantstats._plots import DataPlots, PortfolioPlots
    from jquantstats._stats._stats import Stats

    return {
        "stats": Stats,
        "data-plots": DataPlots,
        "portfolio-plots": PortfolioPlots,
        "portfolio": Portfolio,
        "data": Data,
    }


def _public(cls: type) -> list[str]:
    """List the public attribute names of *cls*.

    Args:
        cls: Class to inspect.

    Returns:
        Sorted public names.
    """
    return sorted(name for name in dir(cls) if not name.startswith("_"))


def _signature(cls: type, name: str) -> str:
    """Render a member's call signature, or mark it as a property.

    The stats methods are declared taking a ``series`` argument but the accessor
    applies them per column and returns a dict, so that first parameter is
    dropped here to show the shape a caller actually uses.

    Args:
        cls: Owning class.
        name: Member name.

    Returns:
        A display string.
    """
    member = inspect.getattr_static(cls, name, None)
    if isinstance(member, property):
        return "  (property)"
    try:
        signature = inspect.signature(getattr(cls, name))
    except (TypeError, ValueError):
        return ""
    params = [p for p in signature.parameters.values() if p.name not in {"self", "series"}]
    rendered = ", ".join(str(p) for p in params)
    return f"({rendered})"


def _first_paragraph(doc: str | None) -> str:
    """Extract the summary line of a docstring.

    Args:
        doc: Raw docstring.

    Returns:
        The first non-empty line, or an empty string.
    """
    for line in (doc or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _constructors() -> list[tuple[str, str]]:
    """Collect the public constructor signatures.

    Returns:
        ``(qualified name, signature)`` pairs.
    """
    from jquantstats import Data, Portfolio

    out = []
    for cls, names in (
        (Portfolio, ("from_cash_position", "from_position", "from_risk_position")),
        (Data, ("from_returns", "from_prices")),
    ):
        for name in names:
            func = getattr(cls, name, None)
            if func is not None:
                out.append((f"{cls.__name__}.{name}", str(inspect.signature(func))))
    return out


def _matches(name: str, pattern: str | None) -> bool:
    """Test a name against a case-insensitive substring filter.

    Args:
        name: Candidate name.
        pattern: Substring, or None to match everything.

    Returns:
        Whether the name matches.
    """
    return pattern is None or pattern.lower() in name.lower()


def _show(name: str) -> int:
    """Print the signature and docstring of every member with this name.

    Args:
        name: Member name to look up.

    Returns:
        A process exit code.
    """
    found = False
    for section, cls in _sections().items():
        if name in _public(cls):
            found = True
            print(f"{section}:{cls.__name__}.{name}{_signature(cls, name)}")
            doc = inspect.getdoc(getattr(cls, name, None))
            if doc:
                print("\n".join(f"    {line}" for line in doc.splitlines()))
            print()
    if not found:
        print(f"{name!r} is not on any public surface.")
        print("It was either renamed to a canonical form or never ported — do not wrap it.")
        print("Try: jqs_api.py --grep <part of the name>")
        return 1
    return 0


def _dump_json(sections: dict[str, Any], pattern: str | None) -> None:
    """Emit the whole surface as JSON.

    Args:
        sections: Section-to-class mapping.
        pattern: Optional substring filter.
    """
    payload: dict[str, Any] = {
        section: {name: _signature(cls, name) for name in _public(cls) if _matches(name, pattern)}
        for section, cls in sections.items()
    }
    payload["constructors"] = dict(_constructors())
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(prog="jqs_api.py", description=__doc__.splitlines()[0])
    parser.add_argument("section", nargs="?", choices=[*SECTION_ORDER, "all"], help="surface to list")
    parser.add_argument("--grep", help="case-insensitive substring filter on member names")
    parser.add_argument("--show", help="print the signature and docstring of one member")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    sections = _sections()

    if args.show:
        return _show(args.show)
    if args.json:
        _dump_json(sections, args.grep)
        return 0

    wanted = SECTION_ORDER if args.section in (None, "all") else (args.section,)
    listing = args.section is not None or args.grep is not None

    for section in wanted:
        if section == "constructors":
            pairs = [(n, s) for n, s in _constructors() if _matches(n, args.grep)]
            if not pairs:
                continue
            print(f"constructors ({len(pairs)})")
            for name, signature in pairs:
                print(f"  {name}{signature}")
            print()
            continue

        cls = sections[section]
        names = [n for n in _public(cls) if _matches(n, args.grep)]
        if not names:
            continue
        print(f"{section} — {cls.__name__} ({len(names)}{'' if args.grep is None else ' matching'})")
        if listing:
            for name in names:
                summary = _first_paragraph(inspect.getdoc(getattr(cls, name, None)))
                print(f"  {name}{_signature(cls, name)}")
                if summary:
                    print(f"      {summary}")
        print()

    if not listing:
        print("pass a section name, --grep, or --show to see members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
