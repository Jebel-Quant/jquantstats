# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.9.7] - 2026-07-10

### New Features
- Add ClusterFuzzLite fuzzing scaffold for jquantstats (#845)

### Bug Fixes
- *(make)* Skip worktree/current branches in clean branch-prune (#838)
- *(stats)* Kill scipy VaR RuntimeWarning (#839) and split large _stats modules (#840) (#841)
- *(reports)* Narrow broad except blocks swallowing benchmark-metric errors (#857)
- *(deps)* Migrate pl.concat to how="horizontal_extend" (#866) (#867)

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 5 updates (#831)
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update) (#829)
- Skip kaleido image-export tests when no browser is available (#833)
- Centralize cross-mixin TYPE_CHECKING stubs (closes #834) (#835)
- Fix Sharpe scale-invariance property test (closes #836) (#837)
- Chore(deps)(deps): bump the python-dependencies group with 6 updates (#844)
- *(plots)* Split 1404-line _plots/_data.py into a subpackage (#846)
- Chore(deps)(deps): bump the github-actions group with 3 updates (#849)
- Chore(deps)(deps): bump the python-dependencies group with 6 updates (#850)
- Silence incidental BenchmarkAlignmentWarning noise in the suite (#848)
- Split the three largest src modules into focused submodules (#851)
- *(exceptions)* Reduce noqa[TRY003] density via reusable exception subclasses (#858)
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#860)
- Decompose C-grade blocks to reduce cyclomatic complexity (#863) (#864)
- Update rhiza to v1.0.2 (#865)
- Chore(deps)(deps): bump docker/login-action from 4.2.0 to 4.4.0 in the github-actions group (#868)
- Chore(deps)(deps): bump the python-dependencies group with 7 updates (#869)
- Update rhiza to v1.1.1 (#870)
- Remove orphaned git_repo fixture and test_utils from .rhiza/tests (#877)
- Update rhiza to v1.1.2 (#878)

### Other Changes
- Add jquantstats vs QuantStats comparison documents
- Move QuantStats comparison docs into book
- Sync Rhiza template v0.18.8 → v0.19.3 (#832)
- Sync Rhiza template v0.19.3 → v0.19.4 (#843)
- Sync Rhiza template v0.19.4 → v0.19.6 + add github-paper (#847)
- Sync Rhiza template v0.19.6 → v0.19.9 (#854)
- Delete .rhiza/tests/test_git_repo_fixture.py (#876)
- Delete .claude/commands directory (#875)

## [0.9.6] - 2026-06-13

### Other Changes
- Round 3 quality: input validation, mutation-gated baseline, PR benchmarks, architecture docs (#825)
- Bump version 0.9.5 → 0.9.6

## [0.9.5] - 2026-06-12

### Bug Fixes
- Warn when benchmark alignment drops rows, loosen kaleido pin
- Raise timeout for kaleido tests to absorb Chrome cold-start

### Other Changes
- Bump version 0.9.4 → 0.9.5

## [0.9.4] - 2026-06-12

### New Features
- Typed stat decorators, snapshot tests, docs FAQ/diagrams, protocol dedup

### Bug Fixes
- CAGR sign flip in reports, accept integer rf, harden API endpoint (#791)
- Harden public API, messaged exceptions, numerical edge-case docs, 100% coverage gate
- Move coverage gate to custom-env.mk so rhiza validation passes

### Maintenance
- Increase coverage to 100%
- Chore(deps)(deps): bump the github-actions group with 9 updates (#789)
- Chore(deps)(deps): bump the python-dependencies group with 6 updates (#790)

### Other Changes
- Round 2 quality: API observability & auth, branch coverage + mutation pass, docs (#812)
- Bump version 0.9.3 → 0.9.4

## [0.9.3] - 2026-06-05

### Bug Fixes
- Update test_release_workflow to match inline release workflow

### Maintenance
- Update rhiza to v0.15.2 (#778)
- Apply rhiza sync v0.17.0
- Update rhiza to v0.15.3 (#779)
- Upgrade paper workflow actions to Node.js 24-compatible versions
- Update rhiza to v0.18.4 (#781)
- Chore(deps)(deps): bump the python-dependencies group with 5 updates (#783)
- Chore(deps)(deps): bump the github-actions group with 8 updates (#782)
- Chore(deps)(deps): bump the python-dependencies group with 10 updates (#786)
- Chore(deps)(deps): bump the github-actions group with 9 updates (#785)

### Other Changes
- Merge branch 'main' into rhiza_v0.17.0
- Update branch
- Delete .github/workflows/rhiza_quality.yml
- Bump version 0.9.2 → 0.9.3

## [0.9.2] - 2026-05-24

### Maintenance
- Chore(deps)(deps): bump github/codeql-action from 4.35.4 to 4.35.5 in the github-actions group (#772)
- Chore(deps)(deps): bump the python-dependencies group with 4 updates (#773)
- Sync rhiza template to v0.10.9 (#774)

### Other Changes
- Bump version 0.9.1 → 0.9.2

## [0.9.1] - 2026-05-17

### Documentation
- Fix README — remove duplicate lines, correct version and parameter name

### Other Changes
- Remove deprecated stats aliases (`ghpr`, `r2`, `win_loss_ratio`) from API surface (#762)
- Bump version 0.9.0 → 0.9.1

## [0.9.0] - 2026-05-17

### New Features
- Add winsorise and exponential_cov to PortfolioUtils (#726)
- *(plots)* Add DataPlots.compare and figsize parity for log_returns/rolling_beta (#750)
- *(plots)* Add Monte Carlo fan chart and simulation distribution plots to `DataPlots` (#749)

### Documentation
- Clarify hhi_positive / hhi_negative as intentionally public optional metrics (#730)
- Close edge-case metric coverage gaps vs quantstats (9 → 10) (#766)

### Performance
- Align `rolling_sortino` to explicit native Polars expressions and add 10Y benchmark coverage (#740)

### Maintenance
- Hide back-references on all facade classes (#709)
- Add `_positive`/`_negative` filter helpers and eliminate inline filter duplication in `_basic.py` (#725)
- Extract _is_finite/_fmt to _reports/_formatting.py (#727)
- Document and normalise null-return convention in stats mixins (#724)
- *(stats)* Rename `_PerformanceStatsMixin` to `_RiskStatsMixin` for clearer domain boundaries (#732)
- Centralize `DataLike` protocol at package root and remove subpackage redefinitions (#736)
- Cancel redundant runs with concurrency group
- *(tests)* Extract inline Data setup into sub-fixtures in `test_stats.py` (#747)
- Document `_ReportingStatsMixin.rar()` cross-mixin dependency via isolation test (#748)

### Other Changes
- Refactor `test_stats.py` to reuse integer-indexed Data setup via sub-fixture (#711)
- Initial plan
- Merge branch 'copilot/address-issue-1a-1b-2a-2b-2c' into main
- Refactor `rolling_sortino` to align with rolling stats implementation pattern (#723)
- Update rhiza_ci.yml
- Document cross-mixin method dependencies in reporting and risk stats mixins (#752)
- Refactor stats decorators to support explicit data attribute contracts (#735)
- Add Monte Carlo stats suite to close QuantStats parity gap (#751)
- Refactor stats mixins to enforce Data null invariants and remove redundant null guards (#739)
- Refine benchmark-aware null handling in `information_ratio` (#754)
- Deprecate `ghpr`, `r2`, and `win_loss_ratio` aliases (preserve compatibility) (#756)
- Refactor `_reports` `StatsLike` to the actual Report-facing surface (#755)
- Document legitimate empty-after-filter guards in stats mixins (#761)
- Consolidate residual duplication in reports, stats, and plot helpers (#760)
- Add snapshot tests for all DataPlots methods; update plot coverage docs to 10/10 (#765)
- Add end-to-end benchmarks and close map_elements / Python-loop performance gaps (#768)
- Add migration tests for new jquantstats functionality (#770)
- Bump version 0.8.4 → 0.9.0

## [0.8.4] - 2026-05-16

### Bug Fixes
- *(docs)* Add src path to mkdocstrings so jquantstats is importable

### Maintenance
- *(reports)* Merge save() into to_html(path=None)

### Other Changes
- Bump version 0.8.3 → 0.8.4

## [0.8.3] - 2026-05-15

### New Features
- *(portfolio)* Accept pl.Expr in all three factory methods (#708)

### Other Changes
- Bump version 0.8.2 → 0.8.3

## [0.8.2] - 2026-05-14

### New Features
- *(utils)* Add DataUtils.exponential_cov (#706)

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 4 updates (#705)
- Chore(deps)(deps): bump github/codeql-action from 4.35.3 to 4.35.4 in the github-actions group (#704)

### Other Changes
- Bump version 0.8.1 → 0.8.2

## [0.8.1] - 2026-05-07

### Dependencies
- *(deps)* Lock file maintenance (#687)
- *(deps)* Lock file maintenance (#689)
- *(deps)* Lock file maintenance (#691)
- *(deps)* Lock file maintenance (#694)
- *(deps)* Lock file maintenance (#695)
- *(deps)* Lock file maintenance (#696)
- *(deps)* Lock file maintenance (#697)
- *(deps)* Lock file maintenance (#698)
- *(deps)* Lock file maintenance (#699)
- *(deps)* Lock file maintenance (#702)

### Maintenance
- Chore(deps-dev)(deps-dev): bump ipython from 9.10.1 to 9.13.0 in the python-dependencies group (#690)
- Chore(deps)(deps): bump kaleido from 1.2.0 to 1.3.0 in the python-dependencies group (#701)
- Chore(deps)(deps): bump github/codeql-action from 4.35.2 to 4.35.3 in the github-actions group (#700)

### Other Changes
- Revise citation format in README
- Delete .github/CITATIONS.bib
- Delete renovate.json
- Update template.yml
- Code coverage badge
- Bump version 0.8.0 → 0.8.1

## [0.8.0] - 2026-04-25

### Dependencies
- *(deps)* Lock file maintenance (#678)
- *(deps)* Lock file maintenance (#679)
- *(deps)* Lock file maintenance (#680)
- *(deps)* Lock file maintenance (#681)
- *(deps)* Lock file maintenance (#682)
- *(deps)* Lock file maintenance (#683)
- *(deps)* Lock file maintenance (#684)

### Other Changes
- Result is coming
- Result is coming
- Add `interpolate`: interior forward-fill for numeric Polars columns (#686)
- Bump version 0.7.0 → 0.8.0

## [0.7.0] - 2026-04-23

### Bug Fixes
- Set dev as permanent default for GitHub Pages
- Add missing mkdocs deps to mike set-default invocation
- Use uv tool install to avoid duplicate uvx invocations for mike
- Remove duplicate test fixtures in test_utils.py

### Documentation
- List notebooks and reports individually in nav
- Replace Sphinx :attr:/:meth: refs with plain backticks
- Enforce strict Google docstring style throughout

### Dependencies
- *(deps)* Lock file maintenance (#653)
- *(deps)* Lock file maintenance (#659)
- *(deps)* Lock file maintenance (#660)
- *(deps)* Lock file maintenance (#667)
- *(deps)* Lock file maintenance (#670)
- *(deps)* Lock file maintenance (#675)
- *(deps)* Lock file maintenance (#677)

### Maintenance
- Refactor security and license targets using extensible hooks
- Chore(deps-dev)(deps-dev): bump yfinance from 1.2.2 to 1.3.0 in the python-dependencies group (#669)
- Chore(deps)(deps): bump github/codeql-action from 4.35.1 to 4.35.2 in the github-actions group (#668)

### Other Changes
- Weekly (#654)
- Fmt
- Delete docs/notebooks.md (#655)
- Scope interrogate pre-commit hook to src/ only (#657)
- Delete docs/rhiza directory (#658)
- Reduce make complexity (#661)
- Delete book/marimo/notebooks/yfinance_demo.py
- Release with mike?
- Update mkdocs-base.yml
- Slim down mkdocs.yml
- Update coverage badge link in README.md
- Update test coverage link in README.md
- Slim down mkdocs.yml
- Extract shared config into docs/mkdocs-base.yml, inherit in mkdocs.yml
- Use material.extensions.emoji in mkdocs-base.yml for CI compatibility
- Moving to zensical
- Moving to zensical
- Replace mike deployment with peaceiris/actions-gh-pages
- Moving to zensical
- Remove tests that require mkdocs-build/MKDOCS_EXTRA_PACKAGES in book.mk
- Run tests before make book to populate coverage HTML report
- Update coverage badge link in README.md
- Generate coverage badge in book workflow, deploy via GitHub Pages
- Fix coverage badge cp: write to /tmp, copy conditionally
- Fix SC2015: use if/fi instead of && || for badge copy
- Explicitly copy coverage HTML into _book/ after build
- Fix coverage badge link: point to docs root instead of 404 path
- Fix MkDocs build: add missing theme name to base config
- Fix book build: replace lucide icons with material, add coverage badge to _book
- Fix coverage badge: use coverage.xml instead of coverage.json for genbadge
- Fix MkDocs build: replace lucide icons with material equivalents
- Fix coverage badge: use genbadge[coverage] extra for coverage subcommand
- Fix coverage badge link to point to HTML report
- Update mkdocs.yml
- Update mkdocs.yml
- Change Test Report path in mkdocs.yml
- Remove mkdocstrings plugin from mkdocs configuration
- Update mkdocs-base.yml
- Fix docs build: install package into uvx env and correct nav paths
- Improve docs site format inspired by TinyCTA style
- Switch book build from mkdocs to zensical (TinyCTA style)
- Remove coverage badge job from CI workflow
- Remove broken coverage badge links from README
- Rhiza2 (#672)
- Feature/add returns winsorising (#676)
- Add to_volatility_adjusted_returns with pluggable vol estimator (#674)
- Bump version 0.6.5 → 0.7.0

## [0.6.5] - 2026-04-14

### Bug Fixes
- Move semgrep.yml from .rhiza to .github (#647)
- Add mkdocstrings[python] to MKDOCS_EXTRA_PACKAGES

### Dependencies
- *(deps)* Lock file maintenance (#643)
- *(deps)* Lock file maintenance (#644)
- *(deps)* Lock file maintenance (#648)
- *(deps)* Lock file maintenance (#651)

### Maintenance
- Sync rhiza template to v0.9.5 (#649)
- Chore(deps-dev)(deps-dev): bump yfinance from 1.2.0 to 1.2.2 in the python-dependencies group (#650)
- Simplify mkdocs.yml via INHERIT from docs/mkdocs-base.yml

### Other Changes
- Delete docs/marimo/rhiza.py (#652)
- Bump version 0.6.4 → 0.6.5

## [0.6.4] - 2026-04-12

### New Features
- Add yfinance portfolio demo notebook

### Bug Fixes
- *(yfinance_demo)* Avoid pyarrow dependency in pandas→polars conversion

### Dependencies
- *(deps)* Lock file maintenance (#641)

### Maintenance
- Remove docs/marimo/ from .gitignore
- Chore(deps)(deps): bump the python-dependencies group with 6 updates (#639)
- Chore(deps)(deps): bump docker/login-action from 4.0.0 to 4.1.0 in the github-actions group (#638)

### Other Changes
- Update repository reference to version 0.9.1 (#640)
- Bump version 0.6.3 → 0.6.4

## [0.6.3] - 2026-03-31

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 2 updates (#636)
- Chore(deps)(deps): bump github/codeql-action from 4.35.0 to 4.35.1 in the github-actions group (#635)
- Improve test layout, module naming consistency, and fixture magic numbers (#633)

### Other Changes
- Fix code block formatting in README.md (#627)
- Release on RhizaSkip branch!? (#629)
- Replace test CSV dependency in README with inline random price data (#631)
- Improve code quality: precise type hints, caching docs, implicit-rename warning (#637)
- Getting started
- Bump version 0.6.2 → 0.6.3

## [0.6.1] - 2026-03-30

### New Features
- Add comprehensive plots & reports gallery notebook (#596)
- Add `null_strategy` parameter to `Data.from_returns` / `from_prices` (#609)
- *(tests)* Property-based tests for financial metric invariants via hypothesis (#620)

### Bug Fixes
- Resolve 404 for Marimo Notebooks and Reports pages in deployed book (#600)
- Pin pygments<2.19 in mkdocs build to avoid NoneType crash

### Documentation
- Sync root migration.md with recent library changes
- Surface CostModel as public API and add dedicated usage guide (#622)
- *(migration)* Expand API mapping table and add jquantstats-only stats section
- *(mkdocs)* Add Docs nav section with all docs/ markdown files
- *(migration)* Acknowledge QuantStats and clarify jquantstats' purpose
- *(migration)* Rewrite introduction to clarify conceptual differences between jquantstats and QuantStats
- Remove comprehensive markdown docs in favor of simpler guidance

### Performance
- Cache expensive Portfolio NAV/returns/tilt/turnover properties (#624)

### Maintenance
- Rename test_quantstats.py to test_autocorrelation.py
- Simplify `conditional_value_at_risk` API and remove deprecation shim
- Add pytest integration tests for FastAPI endpoints in `app.py`
- Drop plot_ prefix from DataPlots methods (#616)
- *(workflows)* Disable scheduled runs, workflow dispatch, and unused steps in rhiza_validate.yml

### Other Changes
- Clean up .env by removing book variables (#594)
- Port quantstats test suite to jquantstats (#598)
- Add `annualise` parameter to `information_ratio` for QuantStats parity (#607)
- Fix double-multiplied and raw-decimal percentage values in performance tables (#602)
- Fix avg_drawdown sign convention to match QuantStats (negative fraction) (#608)
- Add `confidence` deprecation shim to `conditional_value_at_risk` (#610)
- Add minimal FastAPI app and Railway deployment config (#611)
- Add railway.toml to fix uvicorn not found on Railway deployment (#612)
- Delete Procfile (#613)
- Delete architecture section from README
- Move `app.py` from repo root to `api/app.py` (#614)
- Copilot/move app py to api directory (#615)
- Remove annualisation details for information_ratio
- Extract shared computation layer into `_stats/_internals.py` (#618)
- Delete docs/migration.md (#626)
- Migration
- Bump version 0.6.0 → 0.6.1

## [0.5.1] - 2026-03-29

### New Features
- License workflow produces LICENSES.md artifact (#590)
- Add _utils subpackage mirroring qs.utils API

### Bug Fixes
- Add security exception docs to test_portfolio conftest
- Mark illustrative README snippets +RHIZA_SKIP to pass validate

### Documentation
- Update CHANGELOG with 0.5.0 release details
- Rewrite README to foreground the Portfolio route and execution-delay analysis

### Maintenance
- Consolidate root-level files into .github/ and docs/ (#591)
- Split test_portfolio.py into focused modules
- Split portfolio.py into focused mixin modules (#593)

### Other Changes
- Delete .rhiza/templates/minibook directory
- Delete .rhiza/make.d/gh-aw.mk (#588)
- Marimushka go (#592)
- Bump version 0.5.0 → 0.5.1

## [0.5.0] - 2026-03-28

### New Features
- *(stats)* Add Omega ratio (#554)
- *(stats)* Add outliers, remove_outliers, outlier_win_ratio, outlier_loss_ratio
- *(stats)* Add comp, compsum, ghpr (#582)
- *(stats)* Add drawdown_details, expected_return, rolling_greeks, t… (#583)

### Bug Fixes
- Update Rhiza badge URL in test to reflect GitHub org rename
- Update all remaining tschm → jebel-quant org references

### Documentation
- Update README and paper for v0.4.0
- Update links and badges in README for GitHub organization rename
- Update links and badges in README for GitHub organization rename
- Update links and badges in README for GitHub organization rename
- Add REPOSITORY_ANALYSIS.md via make analyse-repo

### Maintenance
- Chore(deps)(deps): bump github/codeql-action from 4.34.1 to 4.35.0 in the github-actions group (#543)
- Chore(deps)(deps): bump the python-dependencies group with 2 updates (#542)
- Merge rhiza_deptry and rhiza_pre-commit into rhiza_quality
- Merge rhiza_docs into rhiza_quality
- Merge rhiza_typecheck into rhiza_validate
- Merge rhiza_pip_audit into rhiza_validate
- Merge rhiza_security into rhiza_validate
- Merge rhiza_semgrep into rhiza_validate
- Consolidate rhiza_validate push/PR jobs into one
- Integrate link_check into rhiza_quality

### Other Changes
- Implement `autocorrelation()` and `acf()` in Stats (#544)
- Quantstats bench (#579)
- Risk (#581)
- Simplify (#584)
- Update README.md
- Merge remote-tracking branch 'origin/main'
- Delete REPOSITORY_ANALYSIS.md
- Sync out (#587)
- Decompose complex `Reports.metrics()` into focused helpers (#586)
- Bump version 0.4.0 → 0.5.0

## [0.4.0] - 2026-03-26

### New Features
- Autopilot loop — analyse → create issues → solve with Claude (#523)
- Add companion paper and LaTeX CI workflow
- Add portfolio initialization from positions, enhance plotting features, and update snapshots

### Bug Fixes
- Raise ValueError when CostModel is constructed with both cost fields non-zero (#528)
- Fix generate_svgs path scope, update coverage badge URL, add from_position tests

### Documentation
- Add paper badge linking to compiled PDF on paper branch
- Reorder Figure 1 facades column to report > stats > plots
- Reorder Figure 1 facades column to stats > report > plots
- Update CHANGELOG for v0.2.0 through v0.3.4 and unreleased
- Add 2026-03-26 v0.3.4 repository analysis entry

### Maintenance
- Remove deploy-versioned-docs job from rhiza_docs workflow
- Remove generate_svgs.py from repository

### Other Changes
- Add 1/n equal-weight portfolio SVG charts from real AAPL/META data (#519)
- Delete .github/workflows/autopilot.yml
- Move mkdocs.yml to root level (#526)
- Update README.md
- Remove exclusion of mail links from link check
- Revisit pyproject
- [WIP] Add kaleido static-export tests to CI matrix (#538)
- Add .github/CODEOWNERS (#540)
- Bump version 0.3.4 → 0.4.0

## [0.3.4] - 2026-03-26

### New Features
- Replace minibook with MkDocs-based book pipeline (#502)
- Add per-subcategory 1–10 scores to analyser agent
- Add vol-normalisation cap and input validation to `from_risk_position` (#521)

### Bug Fixes
- Move mkdocs.yml to repo root to resolve docs_dir config error (#501)
- Limit API Reference TOC depth to 3 to reduce clutter

### Documentation
- Append 2026-03-25 repository analysis entry (v0.3.3)

### Performance
- Vectorise trading_cost_impact — O(1) allocations for the cost sweep (#516)
- Add `slots=True` to `Portfolio` and `Data` frozen dataclasses (#515)

### Maintenance
- Add kaleido static image export tests (#504)
- Move mkdocs.yml to repo root (#514)
- Add kaleido marker and dedicated CI job for static image export (#520)

### Other Changes
- Potential fix for code scanning alert no. 1: Workflow does not contain permissions (#503)
- Revert "chore: move mkdocs.yml to repo root" (#517)
- [WIP] Fix PortfolioLike protocol to align with CostModel abstraction (#522)
- Analysis
- Bump version 0.3.3 → 0.3.4

## [0.3.3] - 2026-03-24

### Other Changes
- Fix mkdocs site_dir nested inside docs_dir
- Remove obsolete notebook
- Add Data.from_prices classmethod (#499)
- Reorganize tests/ to mirror src/jquantstats/ structure (#500)
- Bump version 0.3.2 → 0.3.3

## [0.3.2] - 2026-03-24

### Other Changes
- Update README.md
- [WIP] Add GitHub Actions workflow for link checking in README (#497)
- Fix mike deploy failing due to mkdocs.yml not found at repo root
- Bump version 0.3.1 → 0.3.2

## [0.3.1] - 2026-03-24

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#493)

### Other Changes
- Remove GitHub Codespaces badge
- Remove Renovate badge from README
- Fix README link to jQuantStats
- Replace broken shields.io pyversions badge with static Python version badge (#495)
- [WIP] Update README.md with absolute URLs for references (#496)
- Bump version 0.3.0 → 0.3.1

## [0.3.0] - 2026-03-24

### New Features
- Add `__repr__` with date range to `Data` and `Portfolio` (#489)

### Documentation
- Add versioned documentation with mike (#480)
- Add dashboard screenshot, quick-start output, and Marimo badge to README (#481)

### Maintenance
- Add macOS and Windows to CI test matrix (#482)
- Increase coverage to 100%

### Other Changes
- Fix docs build: update plots reference and broken links in CUSTOMIZATION.md (#470)
- Collapse PortfolioData into Portfolio (#473)
- Add PyPI classifiers and explicit `__all__` (#491)
- Bump version 0.2.0 → 0.3.0

## [0.2.0] - 2026-03-23

### New Features
- Cache stats/plots/report properties on Portfolio and add pytest-cov
- *(packaging)* Add py.typed marker and configure release notes
- *(lint)* Enable ANN2 return-type annotation enforcement in ruff
- Add interrogate to enforce 100% docstring coverage
- Automate release notes with git-cliff in release workflow

### Bug Fixes
- Fix ruff formatting in test_edge_cases.py
- Correct README code examples to pass make validate
- Exclude book/ from interrogate docstring coverage
- Exclude book/ from interrogate pre-commit hook
- Fix import path in test_report: _report -> _reports

### Documentation
- Add Data and Stats classes to mkdocs nav
- Rewrite SECURITY.md for jquantstats
- Expand README with comparison table, Mermaid diagram, more examples, badge row
- Add GitHub Discussions templates and update contributing guide
- Add docs/stability.md and update mkdocs nav
- Replace build_data with Data.from_returns in README
- Document uv.lock update requirement in CONTRIBUTING.md
- Add QuantStats migration guide (docs/migration.md)

### Maintenance
- Extract figure_structure to shared plot_test_utils module, fix ruff formatting
- Add ISSUE_TEMPLATE config.yml and PULL_REQUEST_TEMPLATE.md
- Enable blank issues in ISSUE_TEMPLATE config
- Update issue templates to be jquantstats-specific
- Exclude generated report.html from git tracking
- Add CITATIONS.bib and Citing section to README
- Group _stats*.py files into _stats/ subpackage
- Add test_api_contract.py to guard public API surface

### Other Changes
- Initial plan
- Add property-based tests with hypothesis and fix sortino ZeroDivisionError
- Initial plan
- Add CHANGELOG.md, cliff.toml, and make changelog target
- Merge pull request #384 from tschm/copilot/add-changelog-file
- Merge branch 'main' into copilot/add-property-based-tests
- Remove hypothesis from pyproject.toml; use importorskip for graceful skip
- Initial plan
- Merge branch 'main' into copilot/add-snapshot-tests-for-plots
- Merge pull request #377 from tschm/copilot/add-snapshot-tests-for-plots
- Merge branch 'main' into copilot/add-property-based-tests
- Merge remote-tracking branch 'origin/copilot/add-property-based-tests' into copilot/add-property-based-tests
- Initial plan
- Merge branch 'main' into copilot/add-truncate-in-api
- Add truncate method to Data class with tests
- Merge pull request #386 from tschm/copilot/add-truncate-in-api
- Initial plan
- Merge branch 'main' into copilot/add-conda-forge-recipe
- Add conda-forge recipe and update README installation docs
- Merge branch 'main' into copilot/add-conda-forge-recipe
- Merge pull request #391 from tschm/copilot/add-conda-forge-recipe
- Merge branch 'main' into copilot/add-property-based-tests
- Initial plan
- Merge branch 'main' into copilot/expand-test-edge-cases
- Merge branch 'main' into copilot/expand-test-edge-cases
- Merge pull request #378 from tschm/copilot/expand-test-edge-cases
- Merge branch 'main' into copilot/add-property-based-tests
- Merge pull request #379 from tschm/copilot/add-property-based-tests
- Initial plan
- Initial plan
- Merge branch 'main' into plan10
- Merge branch 'main' into plan10
- Initial plan
- Merge branch 'main' into copilot/add-return-type-annotations
- Merge branch 'main' into copilot/add-return-type-annotations
- Merge pull request #385 from tschm/copilot/add-return-type-annotations
- Merge branch 'main' into plan10
- Merge pull request #396 from tschm/plan10
- Merge branch 'main' into copilot/add-data-and-stats-docs
- Merge pull request #401 from tschm/copilot/add-data-and-stats-docs
- Merge branch 'main' into copilot/add-issue-templates
- Initial plan
- Merge branch 'main' into copilot/stabilize-date-col-convention
- Merge branch 'main' into copilot/stabilize-date-col-convention
- Merge pull request #400 from tschm/copilot/stabilize-date-col-convention
- Merge branch 'main' into copilot/add-issue-templates
- Remove Mypy configuration from pyproject.toml
- Merge pull request #404 from tschm/noMyPy
- Merge branch 'main' into copilot/add-issue-templates
- Merge pull request #403 from tschm/copilot/add-issue-templates
- Add narwhals support to accept pandas/polars/modin inputs in build_data
- Initial plan
- Initial plan
- Add Ideas discussion template and update issue config to redirect to discussions
- Merge branch 'main' into copilot/add-discussions-tab
- Merge pull request #411 from tschm/copilot/add-discussions-tab
- Merge branch 'main' into copilot/add-issue-templates-bug-report-feature-request
- Initial plan
- Add full API reference: new reference pages + updated mkdocs.yml nav + docs dep group
- Merge branch 'main' into copilot/add-mkdocs-mkdocstrings
- Merge pull request #412 from tschm/copilot/add-mkdocs-mkdocstrings
- Merge branch 'main' into copilot/add-issue-templates-bug-report-feature-request
- Initial plan
- Merge branch 'main' into copilot/expand-readme-add-content
- Merge branch 'main' into copilot/expand-readme-add-content
- Merge branch 'main' into copilot/expand-readme-add-content
- Merge branch 'main' into copilot/expand-readme-add-content
- Merge branch 'main' into copilot/expand-readme-add-content
- Merge branch 'main' into copilot/expand-readme-add-content
- Merge pull request #402 from tschm/copilot/expand-readme-add-content
- Merge branch 'main' into copilot/add-issue-templates-bug-report-feature-request
- Initial plan
- Merge branch 'main' into copilot/add-discussions-tab-again
- Merge branch 'main' into copilot/add-discussions-tab-again
- Merge pull request #414 from tschm/copilot/add-discussions-tab-again
- Merge branch 'main' into copilot/add-issue-templates-bug-report-feature-request
- Merge pull request #413 from tschm/copilot/add-issue-templates-bug-report-feature-request
- Initial plan
- Add __repr__ to Data, Stats, Plots and Portfolio
- Initial plan
- Add ROADMAP.md to repository root
- Initial plan
- Merge pull request #428 from tschm/copilot/add-citations-bib-file
- Merge branch 'main' into copilot/add-roadmap-md
- Update ROADMAP.md
- Merge pull request #429 from tschm/copilot/add-roadmap-md
- Merge branch 'main' into copilot/add-repr-to-data-stats-plots-portfolio
- Initial plan
- Replace Any with NativeFrame type alias for narwhals inputs
- Merge branch 'main' into copilot/replace-any-with-nativeframe-type-alias
- Merge branch 'main' into copilot/replace-any-with-nativeframe-type-alias
- Merge pull request #426 from tschm/copilot/replace-any-with-nativeframe-type-alias
- Merge branch 'main' into copilot/add-repr-to-data-stats-plots-portfolio
- Merge pull request #427 from tschm/copilot/add-repr-to-data-stats-plots-portfolio
- Initial plan
- Initial plan
- Merge pull request #435 from tschm/copilot/write-docs-stability-md
- Merge branch 'main' into copilot/fix-tech-debt-in-repository-analysis
- Fix linting issues in test_cost_model.py (match params, FrozenInstanceError, fixture naming)
- Initial plan
- Merge branch 'main' into copilot/add-interrogate-docstring-coverage
- Analysis
- Merge branch 'main' into copilot/add-interrogate-docstring-coverage
- Merge pull request #434 from tschm/copilot/add-interrogate-docstring-coverage
- Merge branch 'main' into copilot/fix-tech-debt-in-repository-analysis
- Merge remote-tracking branch 'origin/copilot/fix-tech-debt-in-repository-analysis' into copilot/fix-tech-debt-in-repository-analysis
- Add docstrings to _stats_core.py for interrogate compliance
- Apply ruff auto-fixes after _stats split
- Fix ty type errors in stats mixin classes
- Merge pull request #437 from tschm/copilot/fix-tech-debt-in-repository-analysis
- Initial plan
- Apply ruff format
- Merge pull request #438 from tschm/copilot/add-data-from-returns-classmethod
- Initial plan
- Initial plan
- Apply ruff auto-fix
- Update import statement for Portfolio from jquantstats
- Update notebooks for flattened analytics API and marimo 0.20.4
- Merge pull request #439 from tschm/copilot/flatten-analytics-subpackage
- Merge branch 'main' into copilot/move-stats-files-to-subpackage
- Initial plan
- Introduce DataLike protocol; remove upward Data imports from _stats mixins
- Fix stale analytics subpackage references in docs
- Merge branch 'main' into copilot/move-stats-files-to-subpackage
- Merge pull request #440 from tschm/copilot/move-stats-files-to-subpackage
- Merge branch 'main' into copilot/remove-upward-imports-from-mixins
- Merge pull request #443 from tschm/copilot/remove-upward-imports-from-mixins
- Initial plan
- Fix ruff D105: add docstring to __post_init__ in data.py
- Merge pull request #445 from tschm/copilot/remove-api-methods
- Initial plan
- Initial plan
- Changes before error encountered
- Initial plan
- Initial plan
- Fmt
- Merge pull request #455 from tschm/copilot/refactor-module-structure
- Merge branch 'main' into copilot/remove-portfolio-plot
- Remove _portfolio_plots.py backward-compat shim
- Use PortfolioPlots directly without alias, like _stats trick
- Initial plan
- Remove _report.py shim, update imports and docstring
- Apply same trick as _stats: re-export _fmt and _stats_table_html from _reports/__init__.py
- Merge pull request #459 from tschm/copilot/remove-jquantstats-report-file
- Merge branch 'main' into copilot/remove-portfolio-plot
- Fmt
- Merge pull request #457 from tschm/copilot/remove-portfolio-plot
- Initial plan
- Reduce __init__.py exports in _stats, _reports, _plots subpackages
- Merge pull request #460 from tschm/copilot/refactor-internal-symbols
- Initial plan
- Rename Plots to DataPlots for symmetry with PortfolioPlots
- Merge pull request #462 from tschm/copilot/rename-plots-to-dataplots
- Initial plan
- Initial plan
- Merge pull request #467 from tschm/copilot/add-test-api-contract-file
- Merge branch 'main' into copilot/add-uv-lock-freeze-ci-installs
- Initial plan
- Merge branch 'main' into copilot/configure-git-cliff-automate-release-notes
- Initial plan
- Merge branch 'main' into copilot/write-docs-migration-guide
- Merge pull request #466 from tschm/copilot/write-docs-migration-guide
- Merge branch 'main' into copilot/configure-git-cliff-automate-release-notes
- Merge pull request #464 from tschm/copilot/configure-git-cliff-automate-release-notes
- Merge branch 'main' into copilot/add-uv-lock-freeze-ci-installs
- Initial plan
- Add Data.describe() method and tests
- Merge branch 'main' into copilot/add-data-describe-method
- Merge branch 'main' into copilot/add-data-describe-method
- Merge branch 'main' into copilot/add-data-describe-method
- Merge pull request #468 from tschm/copilot/add-data-describe-method
- Merge branch 'main' into copilot/add-uv-lock-freeze-ci-installs
- Merge pull request #465 from tschm/copilot/add-uv-lock-freeze-ci-installs
- Update REPOSITORY_ANALYSIS.md with post-PR-462 entry
- Bump version 0.1.1 → 0.2.0

## [0.1.1] - 2026-03-23

### New Features
- Integer-index first-class support for annual_breakdown
- Add cost_bps construction parameter to unify cost model interface
- Allow per-asset vola dict in from_risk_position
- Forward cost_per_unit and cost_bps through from_risk_position

### Bug Fixes
- Correct README pandas claims
- Forward cost_per_unit through all portfolio transforms
- Integer-indexed portfolios use 252 periods/year in _periods_per_year
- Resolve ty type errors in _stats.py and portfolio.py
- Correct README code examples for make validate

### Documentation
- Document analytics facades, cost models, and integer-index limits
- Clarify build_data / Portfolio relationship in README and docstring
- Add 2026-03-23 repository analysis entry for post-merge main branch

### Performance
- Eliminate 21x Data+Stats construction in trading_cost_impact
- Cache Data bridge object in Portfolio to eliminate repeated validation

### Maintenance
- Remove unreachable CleaningInvariantError guards from profits
- Add section dividers to _stats.py
- Update REPOSITORY_ANALYSIS.md to 7.5/10 post-refactor analysis
- Remove CleaningInvariantError (dead code since profits guards removed)
- Update REPOSITORY_ANALYSIS.md to 8.5/10
- Annotate pandas dev dependency as quantstats-only
- Extract duplicated NAV layout code into _apply_nav_layout helper

### Other Changes
- Initial plan
- Fix ruff E501 line-length in exceptions.py docstrings
- Document S101 security exception in analytics test conftest
- Merge pull request #368 from tschm/copilot/copy-analytics-subpackage
- Increase coverage to 100%
- Add REPOSITORY_ANALYSIS.md with initial journal entry
- Refactor Phase 1: structural fixes and API clarity
- Refactor Phase 2a: add Portfolio.data bridge property
- Phase 2b+2c+2d: delegate portfolio.stats to legacy Stats; migrate analytics stats
- Phase 3: add Architecture section to README
- Update Python version and requirements in README
- Update README to reflect polars DataFrame support only
- Update README to remove polars DataFrames support
- Merge branch 'main' into refactor
- Clean portfolio API
- Initial plan
- Merge pull request #372 from tschm/copilot/fix-duplicated-code-in-analytics-plots
- Bump version 0.1.0 → 0.1.1

## [0.0.36] - 2026-03-22

### Other Changes
- Initial plan
- Remove pandas and pyarrow as direct dependencies
- Add pandas back as a dev dependency
- Merge pull request #367 from tschm/copilot/discuss-pandas-and-pyarrow
- Bump version 0.0.35 → 0.0.36

## [0.0.35] - 2026-03-22

### Other Changes
- Update template
- Sync
- Initial plan
- Fix broken coverage badge in README
- Use gh-pages branch ref for coverage badge instead of pinned commit SHA
- Merge pull request #363 from tschm/copilot/fix-broken-badge
- Increase test coverage to 100%
- Initial plan
- Add license and repository details to pyproject.toml
- Merge branch 'main' into copilot/fix-license-badge
- Fix LICENSE badge and description from Apache to MIT
- Merge pull request #365 from tschm/copilot/fix-license-badge
- Bump version 0.0.34 → 0.0.35

## [0.0.34] - 2026-03-17

### Bug Fixes
- Fixing README

### Maintenance
- Chore(deps)(deps): bump plotly in the python-dependencies group
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Update via rhiza
- Chore(deps)(deps): bump the github-actions group with 4 updates

### Other Changes
- Replace Coveralls badge with rhiza-generated coverage badge
- Use rhiza-tools generate-coverage-badge for shields.io endpoint
- Ensure _book/tests/ exists before generating coverage badge
- Update coverage-badge test assertions to match rhiza-tools implementation
- Sync
- Restore coverage-badge.json generation for GitHub Pages badge
- Handle pytest exit code 5 in hypothesis-test target
- Use hardcoded rhiza-tools version instead of RHIZA_VERSION
- Update reference version from v0.8.2 to v0.8.3
- Sync
- Remove redundant cast() calls flagged by ty type checker
- Merge pull request #351 from tschm/tschm-patch-1
- Update reference version to v0.8.5
- Sync
- Merge pull request #352 from tschm/tschm-patch-100
- Merge pull request #353 from tschm/dependabot/github_actions/github-actions-aa99a42152
- Merge branch 'main' into dependabot/uv/python-dependencies-698941b3fe
- Merge pull request #354 from tschm/dependabot/uv/python-dependencies-698941b3fe
- Merge pull request #357 from tschm/dependabot/uv/python-dependencies-41f8a01e75
- Merge branch 'main' into dependabot/github_actions/github-actions-b974f349ee
- Merge pull request #356 from tschm/dependabot/github_actions/github-actions-b974f349ee
- Merge pull request #358 from tschm/rhiza/23122555872
- Merge pull request #359 from tschm/dependabot/github_actions/github-actions-75f14e9cca
- Update reference version and templates in template.yml
- Remove obsolete .gitkeep
- Merge pull request #361 from tschm/tschm-patch-150
- Bump version 0.0.33 → 0.0.34

## [0.0.33] - 2026-02-24

### Bug Fixes
- *(deps)* Update dependency pytest to v9.0.1
- *(deps)* Update dependency ipython to v9.8.0
- *(deps)* Update dependency pytest to v9.0.2
- *(deps)* Update dependency yfinance to v1
- Resolve mypy strict mode type errors
- Resolve conftest import conflict in test_rhiza_workflows

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.6
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.11
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#248)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.14 (#251)
- *(deps)* Update softprops/action-gh-release action to v2.5.0
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.15 (#253)
- *(deps)* Lock file maintenance (#255)
- *(deps)* Lock file maintenance (#256)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.16 (#259)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.8
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.17
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.47.0
- *(deps)* Lock file maintenance
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.9 (#265)
- *(deps)* Lock file maintenance (#267)
- *(deps)* Update github artifact actions
- *(deps)* Lock file maintenance
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.0 (#272)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.10 (#275)
- *(deps)* Lock file maintenance (#276)
- *(deps)* Lock file maintenance (#277)
- *(deps)* Lock file maintenance (#284)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.20 (#286)
- *(deps)* Update dependency astral-sh/uv to v0.9.20 (#287)
- *(deps)* Lock file maintenance
- *(deps)* Update dependency ipython to v9.9.0 (#290)
- *(deps)* Update dependency astral-sh/uv to v0.9.22 (#291)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.22 (#292)
- *(deps)* Lock file maintenance (#293)
- *(deps)* Lock file maintenance (#294)
- *(deps)* Lock file maintenance (#299)
- *(deps)* Update dependency quantstats to v0.0.81 (#301)
- *(deps)* Update pre-commit hook pycqa/bandit to v1.9.3
- *(deps)* Lock file maintenance (#303)
- *(deps)* Lock file maintenance
- *(deps)* Update dependency yfinance to v1.1.0
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.1 (#307)
- *(deps)* Update dependency astral-sh/uv to v0.9.27
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.27
- *(deps)* Lock file maintenance (#310)
- *(deps)* Update pre-commit hook abravalheri/validate-pyproject to v0.25
- *(deps)* Update github/codeql-action action to v4.32.1
- *(deps)* Update dependency ipython to v9.10.0
- *(deps)* Lock file maintenance (#315)
- *(deps)* Update dependency astral-sh/uv to v0.10.2
- *(deps)* Lock file maintenance
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.2
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.11
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#332)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.0
- *(deps)* Update actions/download-artifact action to v7
- *(deps)* Lock file maintenance
- *(deps)* Update dependency astral-sh/uv to v0.10.3 (#337)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.3
- *(deps)* Update dependency astral-sh/uv to v0.10.4 (#340)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.4 (#341)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.2 (#342)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.2 (#343)
- *(deps)* Update github/codeql-action action to v4.32.4 (#344)
- *(deps)* Lock file maintenance (#345)
- *(deps)* Lock file maintenance (#350)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.2

### Maintenance
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Tests
- Sync template files
- Remove deprecated files
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Chore(deps)(deps): bump the python-dependencies group with 2 updates

### Other Changes
- Delete tests/test_docs.py
- Merge pull request #239 from tschm/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #238 from tschm/renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #240 from tschm/renovate/pytest-9.x
- Merge pull request #241 from tschm/renovate/lock-file-maintenance
- Merge pull request #242 from tschm/template-updates
- Revisit README file
- Remove tests/test_taskfile.py Taskfile.yml taskfiles
- Merge pull request #247 from tschm/template-updates
- Merge branch 'main' into remove-file-6
- Merge pull request #246 from tschm/remove-file-6
- Delete .github/workflows/devcontainer.yml
- Update template.yml to modify included and excluded files
- Merge pull request #250 from tschm/tschm-patch-1
- Merge branch 'main' into template-updates
- Merge pull request #249 from tschm/template-updates
- Merge pull request #252 from tschm/renovate/softprops-action-gh-release-2.x
- Merge pull request #254 from tschm/renovate/ipython-9.x
- Update template.yml
- Delete tests/test_makefile.py
- Delete tests/test_readme.py
- Merge pull request #258 from tschm/tschm-patch-1
- Merge pull request #261 from tschm/renovate/pytest-9.x
- Merge pull request #260 from tschm/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #263 from tschm/renovate/igorshubovych-markdownlint-cli-0.x
- Merge branch 'main' into renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #262 from tschm/renovate/ghcr.io-astral-sh-uv-0.x
- Merge branch 'main' into renovate/lock-file-maintenance
- Merge pull request #264 from tschm/renovate/lock-file-maintenance
- Merge pull request #266 from tschm/renovate/major-github-artifact-actions
- Change template repository from tschm to jebel-quant
- Merge pull request #269 from tschm/template-updates
- Script header in notebook
- Merge pull request #270 from tschm/template-updates
- Merge pull request #271 from tschm/renovate/lock-file-maintenance
- Fix formatting of include and exclude lists in template
- Delete .github/workflows/_devcontainer.yml
- Delete .github/workflows/devcontainer.yml
- Delete .github/workflows/docker.yml
- Delete .github/scripts/build-extras.sh
- Rhiza
- Delete .github/README.md
- Delete .github/scripts/sync.sh
- Merge pull request #274 from tschm/cleanup/delete-files
- New sync
- New sync
- New template
- Reduced Makefile
- Makefile.tests
- Rename workflow to 'RHIZA VALIDATE'
- Add 'presentation' to template.yml include list
- Presentation in
- Rhiza subfolder in .github
- Remove RHIZA workflow file
- Merge pull request #281 from tschm/renovate/yfinance-1.x
- Rhiza
- Rhiza
- Migrate
- Sync
- Add renovate.json
- Merge pull request #283 from tschm/renovate/configure
- Rhiza
- Renovate
- Delete .github/workflows/structure.yml
- Merge pull request #285 from tschm/rhiza/20561446968
- Rhiza
- Rhiza
- Rhiza
- Add pytest-html dependency to dev requirements
- Missing pytest-html
- Update README.md
- Sync
- Sync
- Dependencies
- Rename benchmark fixture
- Merge pull request #288 from tschm/renovate/lock-file-maintenance
- Merge branch 'main' into rhiza/20701211891
- Rename benchmark fixture
- Update conftest.py
- Rename benchmark fixture
- Benchmark fixture pain
- Merge pull request #289 from tschm/rhiza/20701211891
- Sync
- Merge pull request #295 from tschm/rhiza/20904134037
- Rhiza sync
- Delete .rhiza.env
- Merge pull request #296 from tschm/tschm-patch-1
- Sync
- Fix ruff linting errors for TRY003, RUF043, and PT012
- Initial plan
- Add deptry package_module_name_map configuration to suppress warnings
- Merge pull request #298 from tschm/copilot/fix-2046079-977845118-594f1aac-28d9-459f-b704-94d91d617dd0
- Sync
- Sync
- Fix all mypy type errors across the codebase
- Merge branch 'main' into rhiza/21120963614
- Merge pull request #300 from tschm/rhiza/21120963614
- Merge pull request #302 from tschm/renovate/pycqa-bandit-1.x
- Merge branch 'main' into rhiza/21341949879
- Merge pull request #304 from tschm/rhiza/21341949879
- Merge pull request #305 from tschm/renovate/yfinance-1.x
- Merge branch 'main' into renovate/lock-file-maintenance
- Merge pull request #306 from tschm/renovate/lock-file-maintenance
- Merge pull request #308 from tschm/renovate/astral-sh-uv-0.x
- Merge pull request #309 from tschm/renovate/ghcr.io-astral-sh-uv-0.x
- Sync
- Missing __init__ in test_rhiza
- Merge branch 'main' into rhiza/21572859042
- Merge pull request #311 from tschm/rhiza/21572859042
- Sync
- Merge pull request #314 from tschm/renovate/abravalheri-validate-pyproject-0.x
- Merge pull request #312 from tschm/renovate/github-codeql-action-4.x
- Merge pull request #313 from tschm/renovate/ipython-9.x
- Update template.yml
- Merge pull request #317 from tschm/tschm-patch-1
- Delete tests/test_rhiza directory
- Merge pull request #318 from tschm/tschm-patch-2
- Delete .github/workflows/rhiza_benchmarks.yml
- Merge pull request #319 from tschm/tschm-patch-3
- Sync
- Merge pull request #320 from tschm/sync20
- Add Rhiza Sync badge to README
- Update README.md
- Initial plan
- Replace static Rhiza badge with workflow status badge
- Merge pull request #322 from tschm/copilot/update-readme-badge
- Merge pull request #323 from tschm/renovate/astral-sh-uv-0.x
- Merge branch 'main' into renovate/astral-sh-uv-pre-commit-0.x
- Initial plan
- Merge branch 'main' into copilot/show-latest-passing-date
- Merge pull request #327 from tschm/copilot/show-latest-passing-date
- Merge branch 'main' into renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #324 from tschm/renovate/astral-sh-uv-pre-commit-0.x
- Merge branch 'main' into renovate/lock-file-maintenance
- Merge pull request #325 from tschm/renovate/lock-file-maintenance
- Update template repository and reference version
- Sync
- Merge pull request #330 from tschm/renovate/rhysd-actionlint-1.x
- Merge pull request #331 from tschm/renovate/lock-file-maintenance
- Sync
- Merge pull request #336 from tschm/renovate/lock-file-maintenance
- Merge branch 'main' into renovate/major-github-artifact-actions
- Merge pull request #335 from tschm/renovate/major-github-artifact-actions
- Merge branch 'main' into renovate/jebel-quant-rhiza-0.x
- Merge pull request #334 from tschm/renovate/jebel-quant-rhiza-0.x
- Merge pull request #338 from tschm/renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #339 from tschm/dependabot/uv/python-dependencies-b29b48e19c
- Remove Rhiza Sync badge from README
- Initial plan
- Add dynamic Rhiza badge to README.md and test
- Merge branch 'main' into copilot/add-rhiza-version-badge
- Merge pull request #348 from tschm/copilot/add-rhiza-version-badge
- Merge pull request #349 from tschm/renovate/jebel-quant-rhiza-0.x
- Sync
- Create gh-aw.mk
- Delete .rhiza/tests/security/__init__.py
- Document security exceptions in conftest.py
- Handle pytest exit code 5 in hypothesis-test target
- Towards release
- Ignore the missing .rhiza/template-bundles.yml file
- Template-bundles
- Bump version 0.0.32 → 0.0.33

## [0.0.32] - 2025-11-16

### Bug Fixes
- *(deps)* Update dependency ipython to v9 (#221)
- *(deps)* Update dependency ipython to v9.7.0 (#227)
- *(deps)* Update dependency kaleido to v1.2.0 (#228)
- *(deps)* Update dependency pytest to v9

### Dependencies
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.6 (#215)
- *(deps)* Lock file maintenance (#222)
- *(deps)* Lock file maintenance (#223)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.4 (#226)
- *(deps)* Lock file maintenance (#229)
- *(deps)* Lock file maintenance (#230)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.35.0 (#234)
- *(deps)* Lock file maintenance (#235)

### Maintenance
- Sync template files (#224)
- Sync template files (#225)
- Sync template files
- Sync template files

### Other Changes
- Remove old test doctest
- Merge pull request #1 from tschm/main
- Adds sharpe variance & prob sharpe ratio
- Merge pull request #231 from tschm/template-updates
- Merge branch 'main' into prob_sharp_ratio
- Merge pull request #233 from tschm/renovate/pytest-9.x
- Merge branch 'main' into prob_sharp_ratio
- Clarified doctrs of sharpe_var and prob_sharpe
- Drops unused noqa
- Merge pull request #232 from mjvakili/prob_sharp_ratio
- Merge branch 'main' into doctest
- Merge pull request #236 from tschm/doctest
- Merge pull request #237 from tschm/template-updates

## [0.0.31] - 2025-10-30

### Dependencies
- *(deps)* Lock file maintenance (#219)

### Other Changes
- Adds return concentrations (#220)

## [0.0.30] - 2025-10-26

### Other Changes
- Update pyproject

## [0.0.29] - 2025-10-26

### Bug Fixes
- *(deps)* Update dependency quantstats to v0.0.73 (#165)
- *(deps)* Update dependency quantstats to v0.0.75 (#169)
- *(deps)* Update dependency quantstats to v0.0.76 (#172)
- *(deps)* Update dependency pytest-cov to v6.3.0 (#180)
- *(deps)* Update dependency quantstats to v0.0.77 (#179)
- *(deps)* Update dependency pytest to v8.4.2 (#178)
- *(deps)* Update dependency kaleido to v1.1.0 (#184)
- *(deps)* Update dependency pytest-cov to v7 (#185)
- *(deps)* Update dependency yfinance to v0.2.66 (#190)

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.9 (#164)
- *(deps)* Lock file maintenance (#166)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.3 (#168)
- *(deps)* Lock file maintenance (#170)
- *(deps)* Update actions/upload-pages-artifact action to v4 (#173)
- *(deps)* Lock file maintenance (#174)
- *(deps)* Update softprops/action-gh-release action to v2.3.3 (#177)
- *(deps)* Lock file maintenance (#181)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.12 (#176)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.0 (#183)
- *(deps)* Lock file maintenance (#186)
- *(deps)* Lock file maintenance (#187)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.1 (#189)
- *(deps)* Lock file maintenance (#192)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.34.0 (#191)
- *(deps)* Lock file maintenance (#193)
- *(deps)* Lock file maintenance (#197)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.2 (#196)
- *(deps)* Lock file maintenance (#200)
- *(deps)* Lock file maintenance (#202)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.3 (#204)
- *(deps)* Update softprops/action-gh-release action to v2.4.0 (#205)
- *(deps)* Lock file maintenance (#206)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.34.1 (#208)
- *(deps)* Update softprops/action-gh-release action to v2.4.1 (#210)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.8 (#209)
- *(deps)* Lock file maintenance (#211)
- *(deps)* Update dependency python to 3.14 (#213)
- *(deps)* Lock file maintenance (#218)

### Maintenance
- Sync config files from .config-templates (#167)
- Sync config files from .config-templates (#171)
- Sync config files from .config-templates (#175)
- Sync config files from .config-templates (#182)
- Trigger renovate
- Trigger renovate
- Sync config files from .config-templates (#188)
- Sync config files from .config-templates (#194)
- Sync template files (#195)
- Sync template files (#198)
- Sync template files (#199)
- Sync template files (#201)
- Sync template files (#203)
- Sync template files (#207)
- Sync template files (#212)
- Sync template from tschm/.config-templates@main (#214)

### Other Changes
- Change Renovate schedule time to 11am on Tuesday
- Delete .github/taskfiles directory
- Delete .github/CODE_OF_CONDUCT.md
- Delete .github/CONTRIBUTING.md
- Refactor sync workflow to use sync_template action
- Delete .devcontainer directory
- Refactor sync.yml for permissions and action version
- Create template.yml
- Change template branch from '77-hot' to 'main'

## [0.0.28] - 2025-08-11

### Bug Fixes
- *(deps)* Update dependency yfinance to v0.2.65 (#127)
- *(deps)* Update dependency quantstats to v0.0.69 (#137)
- *(deps)* Update dependency quantstats to v0.0.70 (#153)

### Dependencies
- *(deps)* Update jebel-quant/marimushka action to v0.1.4 (#126)
- *(deps)* Update tschm/cradle action to v0.2.1 (#128)
- *(deps)* Lock file maintenance (#131)
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.12.3 (#132)
- *(deps)* Update tschm/cradle action to v0.3.01 (#133)
- *(deps)* Lock file maintenance (#134)
- *(deps)* Update tschm/.config-templates action to v0.1.6 (#136)
- *(deps)* Lock file maintenance (#139)
- *(deps)* Update tschm/.config-templates action to v0.1.7 (#142)
- *(deps)* Lock file maintenance (#144)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.7 (#145)
- *(deps)* Update tschm/.config-templates action to v0.1.8 (#146)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.35.1 (#147)
- *(deps)* Update tschm/.config-templates action to v0.2.0 (#148)
- *(deps)* Update actions/download-artifact action to v5 (#149)
- *(deps)* Lock file maintenance (#150)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.8 (#152)
- *(deps)* Update tschm/.config-templates action to v0.3.4 (#154)
- *(deps)* Lock file maintenance (#155)
- *(deps)* Lock file maintenance (#158)
- *(deps)* Update tschm/.config-templates action to v0.4.6 (#159)
- *(deps)* Update actions/checkout action to v5 (#160)
- *(deps)* Lock file maintenance (#161)
- *(deps)* Update actions/checkout action to v5 (#163)

### Maintenance
- Sync config files from .config-templates (#135)
- Sync config files from .config-templates (#138)
- Sync config files from .config-templates (#141)
- Sync config files from .config-templates (#151)
- Sync config files from .config-templates (#157)
- Sync config files from .config-templates (#162)

### Other Changes
- Update deptry.yml (#130)
- Update .pre-commit-config.yaml
- Fmt
- Adding update
- Adding update
- Env
- Jobs from template
- Templated Makefile
- Editorconfig
- Templates
- Workflows
- Update script
- Update workflows
- Update via action
- Update via action
- Update sync.yml
- Delete .devcontainer/startup.sh
- Update sync.yml
- Update sync.yml

## [0.0.27] - 2025-07-05

### Dependencies
- *(deps)* Update jebel-quant/marimushka action to v0.1.3 (#121)
- *(deps)* Lock file maintenance (#125)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.2 (#123)
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.12.2 (#122)
- *(deps)* Update tschm/cradle action to v0.1.80 (#124)

## [0.0.26] - 2025-07-01

### Bug Fixes
- *(deps)* Update dependency yfinance to v0.2.63 (#101)
- *(deps)* Update dependency pytest-cov to v6.2.1 (#102)
- *(deps)* Update dependency pytest to v8.4.1 (#104)
- *(deps)* Update dependency kaleido to v1 (#109)
- *(deps)* Update dependency yfinance to v0.2.64 (#116)

### Dependencies
- *(deps)* Lock file maintenance (#99)
- *(deps)* Lock file maintenance (#106)
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.12.0 (#105)
- *(deps)* Lock file maintenance (#107)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.1 (#108)
- *(deps)* Lock file maintenance (#110)
- *(deps)* Update jebel-quant/marimushka action to v0.1.2 (#113)
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.12.1 (#114)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.34.0 (#118)
- *(deps)* Update tschm/cradle action to v0.1.73 (#120)

### Other Changes
- 90 revisit notebook (#100)
- Update book.yml
- Update marimo.yml (#103)
- Fmt
- Tschm patch 1 (#111)
- Update deptry.yml (#112)
- Book simplifications (#117)
- Book simplifications
- Book simplifications
- Book simplifications
- Update book.yml
- Update deptry.yml (#119)
- Update book.yml
- Update book.yml

## [0.0.25] - 2025-06-10

### Bug Fixes
- *(deps)* Update dependency yfinance to v0.2.62 (#92)

### Dependencies
- *(deps)* Lock file maintenance (#93)
- *(deps)* Lock file maintenance (#94)
- *(deps)* Update tschm/cradle action to v0.1.72 (#97)
- *(deps)* Lock file maintenance (#98)

### Other Changes
- App.setup (#91)

## [0.0.24] - 2025-06-07

### Other Changes
- 88 ruff (#89)

## [0.0.23] - 2025-06-06

### Other Changes
- Ty intro (#87)
- Demo with plot

## [0.0.22] - 2025-06-06

### Dependencies
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.11.13 (#86)

### Maintenance
- Testing code in README

### Other Changes
- Revisiting README
- Revisiting README
- Revisiting README
- Flag D

## [0.0.21] - 2025-06-06

### Other Changes
- Update README.md
- Reports with periods (#85)

## [0.0.20] - 2025-06-05

### Other Changes
- Public folder (#82)
- Update README.md
- 81 move from data to public (#83)
- Update book.yml
- Update book.yml

## [0.0.19] - 2025-06-05

### Other Changes
- Update _data.py (#76)
- 77 remove the path construction from notebooks (#78)
- 79 link to notebook from book (#80)

## [0.0.18] - 2025-06-03

### Bug Fixes
- *(deps)* Update dependency ipython to v8.37.0 (#70)
- *(deps)* Update dependency pytest to v8.4.0 (#74)

### Dependencies
- *(deps)* Update pre-commit hook crate-ci/typos to v1.33.1 (#73)
- *(deps)* Lock file maintenance (#75)
- *(deps)* Update tschm/cradle action to v0.1.71 (#72)

### Other Changes
- Update book.yml
- Update book.yml
- Update book.yml (#71)

## [0.0.17] - 2025-05-31

### Dependencies
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.11.12 (#66)

### Other Changes
- Update pyproject.toml (#69)
- Update book.yml

## [0.0.16] - 2025-05-30

### Other Changes
- 63 prepare for pyodide (#65)

## [0.0.15] - 2025-05-29

### Dependencies
- *(deps)* Update pre-commit hook asottile/pyupgrade to v3.20.0 (#58)
- *(deps)* Lock file maintenance (#59)
- *(deps)* Update tschm/cradle action to v0.1.69 (#60)

### Other Changes
- Update README.md
- Delete opinion.md
- 63 prepare for pyodide (#64)

## [0.0.14] - 2025-05-23

### Other Changes
- Better pdoc
- Symbols
- Only one ticker (#56)

## [0.0.13] - 2025-05-23

### Bug Fixes
- *(deps)* Update dependency polars to v1.30.0 (#52)
- Fix link in book
- Fixing readme links
- Fixing readme links

### Dependencies
- *(deps)* Lock file maintenance (#45)
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.45.0 (#47)
- *(deps)* Update tschm/cradle action to v0.1.66 (#46)
- *(deps)* Lock file maintenance (#48)
- *(deps)* Lock file maintenance (#50)
- *(deps)* Update tschm/cradle action to v0.1.68 (#49)
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.11.11 (#51)

### Maintenance
- Testing build data with Pandas (#55)
- Testing build data with Pandas

### Other Changes
- 53 remove monthly table (#54)
- Better pdoc
- Better pdoc

## [0.0.12] - 2025-05-18

### Maintenance
- Test coverage back to 100%

### Other Changes
- Buttons for interval
- Less light

## [0.0.11] - 2025-05-18

### Other Changes
- Remove some plotting functionality
- Remove the compounding business
- Remove the compounding business

## [0.0.10] - 2025-05-16

### Other Changes
- 43 drawdown in plot factor 100 problem (#44)

## [0.0.9] - 2025-05-16

### Other Changes
- Periods per year as float (#42)

## [0.0.8] - 2025-05-16

### Bug Fixes
- Fix pyarrow

### Dependencies
- *(deps)* Lock file maintenance (#38)
- *(deps)* Update tschm/cradle action to v0.1.64 (#39)
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.11.10 (#40)

### Other Changes
- Workflows with detailed comments generated by Junie (#37)
- Update release.yml

## [0.0.7] - 2025-05-14

### Other Changes
- Problem at build stage

## [0.0.6] - 2025-05-13

### Other Changes
- Opinion generated by Junie
- Opinion generated by Junie

## [0.0.5] - 2025-05-13

### Bug Fixes
- *(deps)* Update dependency ipython to v8.36.0 (#33)

### Other Changes
- 31 testing with quantstats (#32)
- 35 more benchmark with quantstats (#36)
- Opinion generated by Junie

## [0.0.4] - 2025-05-12

### Bug Fixes
- *(deps)* Update dependency yfinance to v0.2.61 (#27)

### Dependencies
- *(deps)* Lock file maintenance (#28)

### Maintenance
- Testing benchmark_pd with missing benchmark

### Other Changes
- 23 return pandas frame returns and benchmark (#26)
- Update ci.yml
- 22 make demo marimo notebook (#29)

## [0.0.3] - 2025-05-12

### Bug Fixes
- *(deps)* Update dependency yfinance to v0.2.59 (#11)

### Dependencies
- *(deps)* Update pre-commit hook crate-ci/typos to v1.32.0 (#10)
- *(deps)* Lock file maintenance (#17)
- *(deps)* Lock file maintenance (#21)

### Other Changes
- Coverage (#9)
- Addressing coverage (#12)
- 16  datapy (#18)
- Periods (#25)

## [0.0.2] - 2025-05-11

### Dependencies
- *(deps)* Update pre-commit hook charliermarsh/ruff-pre-commit to v0.11.9 (#7)
- *(deps)* Lock file maintenance (#8)

### Maintenance
- Test build data
- Test_data
- Testing stats
- Testing two edge cases
- Test an edge case

### Other Changes
- Polars frame for fixtures
- Make polars a first class dependency
- Ruff ignore F821
- Ruff ignore F821
- Ruff ignore F821
- All tests pass
- Remove prices from api
- Remove pandas
- Merge pull request #6 from tschm/5-more-polars

## [0.0.1] - 2025-05-07

### Bug Fixes
- Fixing conftest
- Fix github jobs pointing to quantstats
- Fixing README

### Maintenance
- Tests and api

### Other Changes
- Ignore file
- Pyproject file
- Towards book
- Pre-commit hooks
- Makefile
- Missing init file
- Rename jquantstats
- Update README.md
- Devcontainer
- Install polars as dev dependency
- Bring in polars and pyarrow
- Fmt
- Merge pull request #4 from tschm/3-update-readme

<!-- generated by git-cliff -->
