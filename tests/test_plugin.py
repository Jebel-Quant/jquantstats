"""Tests for the bundled Claude Code plugin.

The plugin's scripts are not part of the shipped package (they live under
``plugin/`` and are excluded from coverage), but a broken loader breaks every
skill that reads the shared portfolio context — so the recipe round-trip, the
digest, and the manifest wiring are all covered here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent / "plugin"
SCRIPTS = PLUGIN_ROOT / "scripts"


def _import(name: str):
    """Import a plugin script by file path, since ``plugin/`` is not a package."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jqs_context = _import("jqs_context")
jqs_load = _import("jqs_load")
jqs_api = _import("jqs_api")


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def prices_frame():
    """Sixty business days of two rising assets."""
    days = pl.date_range(date(2020, 1, 1), date(2020, 3, 20), interval="1d", eager=True).cast(pl.Date)
    return pl.DataFrame(
        {
            "date": days,
            "A": [100.0 + i for i in range(len(days))],
            "B": [200.0 - 0.5 * i for i in range(len(days))],
        }
    )


@pytest.fixture
def project(tmp_path, prices_frame):
    """A throwaway project directory holding prices, positions and returns."""
    data = tmp_path / "data"
    data.mkdir()
    prices_frame.write_csv(data / "prices.csv")
    prices_frame.select(
        pl.col("date"),
        pl.lit(500_000.0).alias("A"),
        pl.lit(300_000.0).alias("B"),
    ).write_csv(data / "pos.csv")
    prices_frame.select(
        pl.col("date").alias("Date"),
        pl.col("A").pct_change().alias("A"),
    ).write_csv(data / "returns.csv")
    return tmp_path


def _portfolio_recipe(name="base", post=None, prices="data/prices.csv", position="data/pos.csv"):
    """Build a Portfolio recipe dict."""
    return {
        "name": name,
        "entry_point": "Portfolio",
        "constructor": "from_cash_position",
        "inputs": {"prices": {"path": prices}, "cash_position": {"path": position}},
        "args": {"aum": 1e6},
        "post": post or [],
    }


def _data_recipe(name="ret", **args):
    """Build a Data recipe dict."""
    return {
        "name": name,
        "entry_point": "Data",
        "constructor": "from_returns",
        "inputs": {"returns": {"path": "data/returns.csv", "date_col": "Date"}},
        "args": args,
        "post": [],
    }


# ── manifest and skill wiring ─────────────────────────────────────────────────


def test_plugin_manifest_matches_marketplace():
    """The marketplace entry and the plugin manifest agree on name and description."""
    manifest = json.loads((PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text())
    marketplace = json.loads((PLUGIN_ROOT.parent / ".claude-plugin" / "marketplace.json").read_text())
    entry = next(p for p in marketplace["plugins"] if p["name"] == manifest["name"])
    assert entry["source"] == "./plugin"
    assert entry["description"] == manifest["description"]


@pytest.mark.parametrize(
    "skill",
    ["quantstats-migration", "portfolio-analysis", "portfolio-diagnostics", "portfolio-robustness"],
)
def test_skill_frontmatter(skill):
    """Every skill declares a name matching its directory, and a description."""
    text = (PLUGIN_ROOT / "skills" / skill / "SKILL.md").read_text()
    assert text.startswith("---\n")
    front = text.split("---", 2)[1]
    assert f"name: {skill}\n" in front
    description = next(line for line in front.splitlines() if line.startswith("description:"))
    assert len(description) > len("description: ") + 40


@pytest.mark.parametrize("command", ["load", "review"])
def test_command_frontmatter(command):
    """Every slash command declares a description."""
    front = (PLUGIN_ROOT / "commands" / f"{command}.md").read_text().split("---", 2)[1]
    assert "description:" in front


def test_referenced_plugin_paths_exist():
    """Every ${CLAUDE_PLUGIN_ROOT} path named in the docs resolves to a real file."""
    referenced = set()
    for path in PLUGIN_ROOT.rglob("*.md"):
        for token in path.read_text().split():
            if "CLAUDE_PLUGIN_ROOT}/" in token:
                referenced.add(token.split("CLAUDE_PLUGIN_ROOT}/", 1)[1].strip('"`,.:;)'))
    assert referenced, "no plugin-root references found — the docs lost their wiring"
    missing = [rel for rel in referenced if not (PLUGIN_ROOT / rel).exists()]
    assert not missing, f"dangling plugin references: {missing}"


# ── recipe round-trip ─────────────────────────────────────────────────────────


def test_build_and_load_round_trip(project):
    """A recipe written to disk rebuilds an object identical to the one measured."""
    recipe = _portfolio_recipe()
    payload = jqs_context.digest(recipe, project)
    jqs_context.upsert(project, recipe)

    rebuilt = jqs_context.load(root=project)
    assert type(rebuilt).__name__ == "Portfolio"
    # abs=1e-6 because the digest rounds its anchors to six decimals
    assert rebuilt.stats.sharpe()["returns"] == pytest.approx(payload["anchors"]["sharpe"]["returns"], abs=1e-6)


def test_post_ops_are_replayed(project):
    """A recorded lag survives the round-trip — an unlagged rebuild would differ."""
    plain = jqs_context.build(_portfolio_recipe(), project)
    lagged_recipe = _portfolio_recipe(name="lag1", post=[{"op": "lag", "n": 1}])
    jqs_context.upsert(project, lagged_recipe)

    lagged = jqs_context.load("lag1", root=project)
    assert lagged.returns.height == plain.returns.height
    assert lagged.stats.sharpe()["returns"] != pytest.approx(plain.stats.sharpe()["returns"])


def test_truncate_post_op(project):
    """A truncate transform narrows the rebuilt span."""
    recipe = _portfolio_recipe(post=[{"op": "truncate", "start": "2020-02-01"}])
    obj = jqs_context.build(recipe, project)
    assert obj.prices["date"].min() >= date(2020, 2, 1)


def test_truncate_bounds_are_coerced(project):
    """An ISO string bound reaches the library as a date object.

    JSON has no date type, and `truncate` rejects strings despite advertising
    `str` in its signature — so the recipe layer converts.
    """
    from_recipe = jqs_context.build(_portfolio_recipe(post=[{"op": "truncate", "start": "2020-01-11"}]), project)
    direct = jqs_context.build(_portfolio_recipe(), project).truncate(start=date(2020, 1, 11))
    assert from_recipe.prices.height == direct.prices.height < 80

    with pytest.raises(jqs_context.ContextError, match="not an ISO-8601 date"):
        jqs_context.build(_portfolio_recipe(post=[{"op": "truncate", "start": "last tuesday"}]), project)


def test_integer_truncate_bound_is_a_library_noop(project):
    """Pins current library behaviour: an int bound on a temporal axis truncates nothing.

    `truncate` accepts `int` as a row index, but routes on the index dtype, so a
    temporal axis silently ignores it. Kept as a test so a future fix is visible
    here rather than surprising a recipe that used a row index.
    """
    obj = jqs_context.build(_portfolio_recipe(post=[{"op": "truncate", "start": 10}]), project)
    assert obj.prices.height == 80


def test_resample_post_op(project):
    """Data supports resample; Portfolio does not, and says so."""
    resampled = jqs_context.build(_data_recipe(post=[]) | {"post": [{"op": "resample", "every": "1mo"}]}, project)
    assert resampled.returns.height < 60

    with pytest.raises(jqs_context.ContextError, match="not supported by Portfolio"):
        jqs_context.build(_portfolio_recipe(post=[{"op": "resample", "every": "1mo"}]), project)


def test_named_contexts_are_independent(project):
    """Activating and resolving named contexts keeps each recipe intact."""
    jqs_context.upsert(project, _portfolio_recipe("base"))
    jqs_context.upsert(project, _portfolio_recipe("lag1", post=[{"op": "lag", "n": 1}]))

    container = jqs_context.read_container(project)
    assert container["active"] == "lag1"
    assert set(container["contexts"]) == {"base", "lag1"}

    name, recipe = jqs_context.resolve(project, "base")
    assert (name, recipe["post"]) == ("base", [])


# ── digest ────────────────────────────────────────────────────────────────────


def test_digest_reports_shape(project):
    """The digest carries the facts an analysis needs before choosing a metric."""
    payload = jqs_context.digest(_portfolio_recipe(), project)
    shape = payload["shape"]
    assert shape["assets"] == ["A", "B"]
    assert shape["date_axis_temporal"] is True
    assert shape["date_column"] == "date"
    assert shape["periods_per_year"] > 0
    assert payload["built"] == "Portfolio.from_cash_position"
    assert {row["asset"] for row in shape["per_asset"]} == {"A", "B"}
    assert "turnover" in payload["anchors"]


def test_digest_warns_about_nulls(project):
    """Unset null_strategy on a frame with nulls is surfaced, not silently accepted."""
    payload = jqs_context.digest(_data_recipe(), project)
    assert payload["nulls"]["returns"]["A"] == 1
    assert any("null_strategy" in w for w in payload["warnings"])


def test_null_strategy_silences_the_warning(project):
    """Declaring the strategy resolves the finding."""
    payload = jqs_context.digest(_data_recipe(null_strategy="drop"), project)
    assert not any("null_strategy" in w for w in payload["warnings"])


def test_digest_warns_about_positional_index(project, prices_frame):
    """A non-temporal axis is reported, since it silently changes annualisation."""
    frame = prices_frame.with_row_index("i").drop("date")
    frame.write_csv(project / "data" / "int_returns.csv")
    recipe = {
        "name": "positional",
        "entry_point": "Data",
        "constructor": "from_returns",
        "inputs": {"returns": {"path": "data/int_returns.csv", "date_col": "i"}},
        "args": {},
        "post": [],
    }
    payload = jqs_context.digest(recipe, project)
    assert payload["shape"]["date_axis_temporal"] is False
    assert any("positional" in w for w in payload["warnings"])


def test_date_column_is_normalised_for_portfolio(project, prices_frame):
    """A 'Date' price column is renamed, because the bridge only keeps 'date'."""
    prices_frame.rename({"date": "Date"}).write_csv(project / "data" / "prices_upper.csv")
    prices_frame.select(pl.col("date").alias("Date"), pl.lit(1e5).alias("A"), pl.lit(1e5).alias("B")).write_csv(
        project / "data" / "pos_upper.csv"
    )
    recipe = _portfolio_recipe(prices="data/prices_upper.csv", position="data/pos_upper.csv")
    obj, _, notes = jqs_context.build_detailed(recipe, project)

    assert obj.data.index.columns == ["date"]
    assert any("renamed to 'date'" in note for note in notes)


# ── fingerprints ──────────────────────────────────────────────────────────────


def test_staleness_is_detected(project):
    """Changing an input after the digest is written invalidates it."""
    recipe = _portfolio_recipe()
    jqs_context.upsert(project, recipe)
    jqs_context.write_digest(project, jqs_context.digest(recipe, project))
    assert jqs_context.stale_inputs(project) == []

    target = project / "data" / "prices.csv"
    target.write_text(target.read_text() + "2020-04-01,999.0,999.0\n")
    assert jqs_context.stale_inputs(project) == ["data/prices.csv"]


def test_digests_are_keyed_by_context(project):
    """Writing one context's digest must not invalidate a sibling's fingerprints."""
    base, other = _portfolio_recipe("base"), _data_recipe("ret")
    for recipe in (base, other):
        jqs_context.upsert(project, recipe)
        jqs_context.write_digest(project, jqs_context.digest(recipe, project))

    assert set(jqs_context.read_digests(project)) == {"base", "ret"}

    target = project / "data" / "returns.csv"
    target.write_text(target.read_text() + "2020-04-01,0.01\n")
    assert jqs_context.stale_inputs(project, "base") == []
    assert jqs_context.stale_inputs(project, "ret") == ["data/returns.csv"]


def test_context_dir_ignores_the_derived_digest(project):
    """The context directory ignores its own derived file, host .gitignore untouched."""
    jqs_context.upsert(project, _portfolio_recipe())
    ignore = project / jqs_context.CONTEXT_DIRNAME / ".gitignore"
    assert ignore.exists()
    patterns = [line for line in ignore.read_text().splitlines() if line and not line.startswith("#")]
    assert patterns == [jqs_context.DIGEST_FILENAME]  # the recipe stays committable


def test_stale_inputs_without_a_digest(project):
    """No digest yet means nothing to invalidate."""
    jqs_context.upsert(project, _portfolio_recipe())
    assert jqs_context.stale_inputs(project) == []


# ── script-backed inputs ──────────────────────────────────────────────────────


def test_script_backed_input(project):
    """Positions can come from a repo-owned function instead of a file."""
    (project / "build_pos.py").write_text(
        "import pathlib\n"
        "import polars as pl\n"
        "def positions():\n"
        "    here = pathlib.Path(__file__).parent\n"
        "    frame = pl.read_csv(here / 'data' / 'prices.csv', try_parse_dates=True)\n"
        "    return frame.select(pl.col('date'), pl.lit(4e5).alias('A'), pl.lit(4e5).alias('B'))\n"
    )
    recipe = _portfolio_recipe()
    recipe["inputs"]["cash_position"] = {"script": "build_pos.py", "callable": "positions"}
    obj = jqs_context.build(recipe, project)
    assert obj.assets == ["A", "B"]
    assert "build_pos.py" in jqs_context.fingerprints(recipe, project)


def test_script_backed_input_errors(project):
    """A missing script, callable, or wrong return type each fail loudly."""
    recipe = _portfolio_recipe()
    recipe["inputs"]["cash_position"] = {"script": "nope.py"}
    with pytest.raises(jqs_context.ContextError, match="builder script not found"):
        jqs_context.build(recipe, project)

    (project / "bad.py").write_text("def other():\n    return 1\n")
    recipe["inputs"]["cash_position"] = {"script": "bad.py", "callable": "missing"}
    with pytest.raises(jqs_context.ContextError, match="no callable named"):
        jqs_context.build(recipe, project)

    recipe["inputs"]["cash_position"] = {"script": "bad.py", "callable": "other"}
    with pytest.raises(jqs_context.ContextError, match=r"expected pl\.DataFrame"):
        jqs_context.build(recipe, project)


# ── error paths ───────────────────────────────────────────────────────────────


def test_missing_context_file(tmp_path):
    """Reading a context that was never created explains how to make one."""
    with pytest.raises(jqs_context.ContextError, match=r"run jqs_load\.py"):
        jqs_context.read_container(tmp_path)


def test_malformed_context_file(tmp_path):
    """A corrupt recipe file is reported as such, not as a KeyError later on."""
    (tmp_path / jqs_context.CONTEXT_DIRNAME).mkdir()
    jqs_context.context_path(tmp_path).write_text("{not json")
    with pytest.raises(jqs_context.ContextError, match="not valid JSON"):
        jqs_context.read_container(tmp_path)


def test_future_schema_is_refused(tmp_path):
    """A newer on-disk schema is refused rather than half-understood."""
    (tmp_path / jqs_context.CONTEXT_DIRNAME).mkdir()
    jqs_context.context_path(tmp_path).write_text(json.dumps({"schema": 99, "contexts": {}}))
    with pytest.raises(jqs_context.ContextError, match="schema 99"):
        jqs_context.read_container(tmp_path)


def test_unknown_context_name(project):
    """Resolving an unknown name lists the ones that exist."""
    jqs_context.upsert(project, _portfolio_recipe("base"))
    with pytest.raises(jqs_context.ContextError, match="known: base"):
        jqs_context.resolve(project, "nope")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"entry_point": "Nope"}, "entry_point must be"),
        ({"constructor": "from_magic"}, "unknown Portfolio constructor"),
        ({"inputs": {"prices": {"path": "data/prices.csv"}}}, "needs an input named"),
        ({"post": [{"op": "warp"}]}, "unknown post op"),
    ],
)
def test_invalid_recipes(project, mutation, match):
    """Each way a recipe can be wrong raises ContextError with a usable message."""
    recipe = _portfolio_recipe() | mutation
    with pytest.raises(jqs_context.ContextError, match=match):
        jqs_context.build(recipe, project)


def test_missing_input_file(project):
    """A recipe pointing at a deleted file names the path."""
    recipe = _portfolio_recipe(prices="data/gone.csv")
    with pytest.raises(jqs_context.ContextError, match="input file not found"):
        jqs_context.build(recipe, project)


def test_unsupported_format(project):
    """An unreadable suffix is refused before polars guesses at it."""
    (project / "data" / "prices.xlsx").write_text("nope")
    with pytest.raises(jqs_context.ContextError, match="unsupported table format"):
        jqs_context.build(_portfolio_recipe(prices="data/prices.xlsx"), project)


def test_input_spec_without_source(project):
    """An input naming neither a path nor a script is rejected."""
    recipe = _portfolio_recipe()
    recipe["inputs"]["prices"] = {"date_col": "date"}
    with pytest.raises(jqs_context.ContextError, match="needs a 'path' or a 'script'"):
        jqs_context.build(recipe, project)


def test_unknown_columns(project):
    """Restricting to absent columns fails with the available ones listed."""
    recipe = _portfolio_recipe()
    recipe["inputs"]["prices"] = {"path": "data/prices.csv", "columns": ["ZZZ"]}
    with pytest.raises(jqs_context.ContextError, match="columns"):
        jqs_context.build(recipe, project)


def test_unknown_cost_model(project):
    """A cost model kind outside the two supported forms is refused."""
    recipe = _portfolio_recipe()
    recipe["args"]["cost_model"] = {"kind": "gut_feel", "value": 1}
    with pytest.raises(jqs_context.ContextError, match="unknown cost_model kind"):
        jqs_context.build(recipe, project)


@pytest.mark.parametrize("kind", ["per_unit", "turnover_bps"])
def test_cost_models(project, kind):
    """Both declarative cost models reach the constructor."""
    recipe = _portfolio_recipe()
    recipe["args"]["cost_model"] = {"kind": kind, "value": 2.0}
    assert jqs_context.build(recipe, project).cost_model is not None


# ── the loader CLI ────────────────────────────────────────────────────────────


def test_cli_portfolio_writes_context(project, capsys):
    """The portfolio subcommand prints a digest and records both files."""
    code = jqs_load.main(
        [
            "portfolio",
            "--prices",
            str(project / "data" / "prices.csv"),
            "--cash-position",
            str(project / "data" / "pos.csv"),
            "--aum",
            "1e6",
            "--cost-bps",
            "5",
            "--lag",
            "1",
            "--root",
            str(project),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Portfolio.from_cash_position" in out
    assert "lag(n=1)" in out

    container = jqs_context.read_container(project)
    recipe = container["contexts"]["base"]
    assert recipe["post"] == [{"op": "lag", "n": 1}]
    assert recipe["args"]["cost_bps"] == 5.0
    assert recipe["inputs"]["prices"]["path"] == "data/prices.csv"
    assert jqs_context.digest_path(project).exists()


def test_recipe_paths_are_posix(project, prices_frame):
    r"""Recipe paths use forward slashes on every platform, so a recipe travels.

    A Windows-authored `data\prices.csv` would not resolve on another machine,
    and the recipe is meant to be committed and shared.
    """
    nested = project / "data" / "nested"
    nested.mkdir()
    prices_frame.write_csv(nested / "prices.csv")
    prices_frame.select(pl.col("date"), pl.lit(1e5).alias("A"), pl.lit(1e5).alias("B")).write_csv(nested / "pos.csv")

    jqs_load.main(
        [
            "portfolio",
            "--prices",
            str(nested / "prices.csv"),
            "--cash-position",
            str(nested / "pos.csv"),
            "--aum",
            "1e6",
            "--root",
            str(project),
        ]
    )
    recipe = jqs_context.read_container(project)["contexts"]["base"]
    assert recipe["inputs"]["prices"]["path"] == "data/nested/prices.csv"
    assert "\\" not in json.dumps(recipe)
    assert jqs_context.load(root=project).assets == ["A", "B"]  # and it still rebuilds


def test_cli_data_warns_and_exits_nonzero(project, capsys):
    """Building over nulls without a strategy is reported and flagged in the code."""
    code = jqs_load.main(["data", "--returns", str(project / "data" / "returns.csv"), "--root", str(project)])
    assert code == 1
    assert "WARNING" in capsys.readouterr().out


def test_cli_requires_one_position_flag(project):
    """Zero or several position flags is a usage error, not a guess."""
    args = ["portfolio", "--prices", str(project / "data" / "prices.csv"), "--aum", "1e6", "--root", str(project)]
    with pytest.raises(SystemExit, match="exactly one of"):
        jqs_load.main(args)
    with pytest.raises(SystemExit, match="exactly one of"):
        jqs_load.main([*args, "--cash-position", "a.csv", "--position", "b.csv"])


def test_cli_data_requires_one_source(project):
    """Data needs exactly one of returns or prices."""
    with pytest.raises(SystemExit, match="exactly one of"):
        jqs_load.main(["data", "--root", str(project)])


def test_cli_show_list_activate_and_check(project, capsys):
    """The read-only subcommands report the recorded contexts."""
    common = ["--prices", str(project / "data" / "prices.csv"), "--root", str(project)]
    jqs_load.main(["portfolio", *common, "--cash-position", str(project / "data" / "pos.csv"), "--aum", "1e6"])
    jqs_load.main(
        [
            "portfolio",
            *common,
            "--cash-position",
            str(project / "data" / "pos.csv"),
            "--aum",
            "1e6",
            "--lag",
            "1",
            "--name",
            "lag1",
        ]
    )
    capsys.readouterr()

    assert jqs_load.main(["list", "--root", str(project)]) == 0
    listing = capsys.readouterr().out
    assert "* lag1" in listing
    assert "  base" in listing

    assert jqs_load.main(["activate", "base", "--root", str(project)]) == 0
    assert "base" in capsys.readouterr().out

    assert jqs_load.main(["show", "--root", str(project)]) == 0
    assert "Portfolio.from_cash_position" in capsys.readouterr().out

    assert jqs_load.main(["check", "--root", str(project)]) == 0
    assert "fresh" in capsys.readouterr().out


def test_cli_check_reports_stale(project, capsys):
    """A mutated input makes check fail loudly."""
    jqs_load.main(
        [
            "portfolio",
            "--prices",
            str(project / "data" / "prices.csv"),
            "--cash-position",
            str(project / "data" / "pos.csv"),
            "--aum",
            "1e6",
            "--root",
            str(project),
        ]
    )
    target = project / "data" / "pos.csv"
    target.write_text(target.read_text() + "2020-04-01,1.0,1.0\n")
    capsys.readouterr()

    assert jqs_load.main(["check", "--root", str(project)]) == 1
    assert "STALE" in capsys.readouterr().out


def test_cli_check_without_digest(project, capsys):
    """Checking before anything was built explains what to do."""
    assert jqs_load.main(["check", "--root", str(project)]) == 0
    assert "no digest recorded yet" in capsys.readouterr().out


def test_cli_json_output(project, capsys):
    """--json emits the recipe and digest for programmatic use."""
    code = jqs_load.main(
        [
            "data",
            "--returns",
            str(project / "data" / "returns.csv"),
            "--null-strategy",
            "drop",
            "--json",
            "--no-write",
            "--root",
            str(project),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["recipe"]["args"]["null_strategy"] == "drop"
    assert payload["digest"]["shape"]["assets"] == ["A"]
    assert not jqs_context.context_path(project).exists()


def test_cli_rf_as_scalar_and_path(project):
    """--rf takes a number or a frame path, and records which it was."""
    common = ["data", "--returns", str(project / "data" / "returns.csv"), "--no-write", "--root", str(project)]
    jqs_load.main([*common, "--rf", "0.01"])
    recipe = jqs_load._data_recipe(
        jqs_load.build_parser().parse_args([*common, "--rf", "0.01"]),
        project,
    )
    assert recipe["args"]["rf"] == 0.01

    path_recipe = jqs_load._data_recipe(
        jqs_load.build_parser().parse_args([*common, "--rf", str(project / "data" / "returns.csv")]),
        project,
    )
    assert path_recipe["args"]["rf"]["path"] == "data/returns.csv"


def test_cli_reports_context_errors(project, capsys):
    """A broken recipe surfaces as a message and exit code 2, not a traceback."""
    jqs_context.upsert(project, _portfolio_recipe(prices="data/gone.csv"))
    assert jqs_load.main(["show", "--root", str(project)]) == 2
    assert "input file not found" in capsys.readouterr().err


# ── the API introspection script ──────────────────────────────────────────────


def test_api_sections(capsys):
    """The bare listing reports every section it knows about."""
    assert jqs_api.main([]) == 0
    out = capsys.readouterr().out
    for section in ("stats", "data-plots", "portfolio-plots", "constructors"):
        assert section in out


def test_api_show_existing(capsys):
    """--show prints the signature and docstring of a real member."""
    assert jqs_api.main(["--show", "conditional_value_at_risk"]) == 0
    out = capsys.readouterr().out
    assert "confidence" in out
    assert "alpha" in out


def test_api_show_missing(capsys):
    """--show on a QuantStats alias reports absence instead of implying a wrapper."""
    assert jqs_api.main(["--show", "cvar"]) == 1
    assert "not on any public surface" in capsys.readouterr().out


def test_api_grep_and_json(capsys):
    """--grep filters names; --json emits the whole surface."""
    assert jqs_api.main(["--grep", "drawdown"]) == 0
    assert "max_drawdown" in capsys.readouterr().out

    assert jqs_api.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "sharpe" in payload["stats"]
    assert "Portfolio.from_cash_position" in payload["constructors"]


def test_api_section_listing(capsys):
    """Naming a section lists its members with summaries."""
    assert jqs_api.main(["portfolio-plots"]) == 0
    out = capsys.readouterr().out
    assert "lead_lag_ir_plot" in out
    assert "snapshot" in out
