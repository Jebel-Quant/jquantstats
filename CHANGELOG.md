## [0.9.2] - 2026-05-24

### 💼 Other

- Bump version 0.9.1 → 0.9.2

### ⚙️ Miscellaneous Tasks

- Sync rhiza template to v0.10.9 (#774)
- Bump rhiza to v0.14.1 with github-project profile and marimo template (#775)
## [0.9.1] - 2026-05-17

### 💼 Other

- Bump version 0.9.0 → 0.9.1

### 📚 Documentation

- Fix README — remove duplicate lines, correct version and parameter name
## [0.9.0] - 2026-05-17

### 🚀 Features

- Add winsorise and exponential_cov to PortfolioUtils (#726)
- *(plots)* Add DataPlots.compare and figsize parity for log_returns/rolling_beta (#750)
- *(plots)* Add Monte Carlo fan chart and simulation distribution plots to `DataPlots` (#749)

### 💼 Other

- Bump version 0.8.4 → 0.9.0

### 🚜 Refactor

- Hide back-references on all facade classes (#709)
- Add `_positive`/`_negative` filter helpers and eliminate inline filter duplication in `_basic.py` (#725)
- Extract _is_finite/_fmt to _reports/_formatting.py (#727)
- Document and normalise null-return convention in stats mixins (#724)
- *(stats)* Rename `_PerformanceStatsMixin` to `_RiskStatsMixin` for clearer domain boundaries (#732)
- Centralize `DataLike` protocol at package root and remove subpackage redefinitions (#736)
- *(tests)* Extract inline Data setup into sub-fixtures in `test_stats.py` (#747)

### 📚 Documentation

- Clarify hhi_positive / hhi_negative as intentionally public optional metrics (#730)
- Close edge-case metric coverage gaps vs quantstats (9 → 10) (#766)

### ⚡ Performance

- Align `rolling_sortino` to explicit native Polars expressions and add 10Y benchmark coverage (#740)

### 🧪 Testing

- Document `_ReportingStatsMixin.rar()` cross-mixin dependency via isolation test (#748)

### ⚙️ Miscellaneous Tasks

- Cancel redundant runs with concurrency group
## [0.8.4] - 2026-05-16

### 🐛 Bug Fixes

- *(docs)* Add src path to mkdocstrings so jquantstats is importable

### 💼 Other

- Bump version 0.8.3 → 0.8.4

### 🚜 Refactor

- *(reports)* Merge save() into to_html(path=None)
## [0.8.3] - 2026-05-15

### 🚀 Features

- *(portfolio)* Accept pl.Expr in all three factory methods (#708)

### 💼 Other

- Bump version 0.8.2 → 0.8.3
## [0.8.2] - 2026-05-14

### 🚀 Features

- *(utils)* Add DataUtils.exponential_cov (#706)

### 💼 Other

- Bump version 0.8.1 → 0.8.2
## [0.8.1] - 2026-05-07

### 💼 Other

- Bump version 0.8.0 → 0.8.1
## [0.8.0] - 2026-04-25

### 💼 Other

- Bump version 0.7.0 → 0.8.0
## [0.7.0] - 2026-04-23

### 🐛 Bug Fixes

- Set dev as permanent default for GitHub Pages
- Add missing mkdocs deps to mike set-default invocation
- Use uv tool install to avoid duplicate uvx invocations for mike
- Remove duplicate test fixtures in test_utils.py

### 💼 Other

- Bump version 0.6.5 → 0.7.0

### 📚 Documentation

- List notebooks and reports individually in nav
- Replace Sphinx :attr:/:meth: refs with plain backticks
- Enforce strict Google docstring style throughout

### ⚙️ Miscellaneous Tasks

- Refactor security and license targets using extensible hooks
- Bump rhiza template to v0.10.1 (#666)
## [0.6.5] - 2026-04-14

### 🐛 Bug Fixes

- Move semgrep.yml from .rhiza to .github (#647)
- Add mkdocstrings[python] to MKDOCS_EXTRA_PACKAGES

### 💼 Other

- Bump version 0.6.4 → 0.6.5

### ⚙️ Miscellaneous Tasks

- Sync rhiza template to v0.9.5 (#649)
- Simplify mkdocs.yml via INHERIT from docs/mkdocs-base.yml
## [0.6.4] - 2026-04-12

### 🚀 Features

- Add yfinance portfolio demo notebook

### 🐛 Bug Fixes

- *(yfinance_demo)* Avoid pyarrow dependency in pandas→polars conversion

### 💼 Other

- Bump version 0.6.3 → 0.6.4

### ⚙️ Miscellaneous Tasks

- Remove docs/marimo/ from .gitignore
## [0.6.3] - 2026-03-31

### 💼 Other

- Bump version 0.6.2 → 0.6.3

### 🚜 Refactor

- Improve test layout, module naming consistency, and fixture magic numbers (#633)
## [0.6.1] - 2026-03-30

### 🚀 Features

- Add comprehensive plots & reports gallery notebook (#596)
- Add `null_strategy` parameter to `Data.from_returns` / `from_prices` (#609)
- *(tests)* Property-based tests for financial metric invariants via hypothesis (#620)

### 🐛 Bug Fixes

- Resolve 404 for Marimo Notebooks and Reports pages in deployed book (#600)
- Pin pygments<2.19 in mkdocs build to avoid NoneType crash

### 💼 Other

- Extract shared computation layer into `_stats/_internals.py` (#618)
- Bump version 0.6.0 → 0.6.1

### 🚜 Refactor

- Rename test_quantstats.py to test_autocorrelation.py
- Simplify `conditional_value_at_risk` API and remove deprecation shim
- Drop plot_ prefix from DataPlots methods (#616)

### 📚 Documentation

- Sync root migration.md with recent library changes
- Surface CostModel as public API and add dedicated usage guide (#622)
- *(migration)* Expand API mapping table and add jquantstats-only stats section
- *(mkdocs)* Add Docs nav section with all docs/ markdown files
- *(migration)* Acknowledge QuantStats and clarify jquantstats' purpose
- *(migration)* Rewrite introduction to clarify conceptual differences between jquantstats and QuantStats
- Remove comprehensive markdown docs in favor of simpler guidance

### ⚡ Performance

- Cache expensive Portfolio NAV/returns/tilt/turnover properties (#624)

### 🧪 Testing

- Add pytest integration tests for FastAPI endpoints in `app.py`

### ⚙️ Miscellaneous Tasks

- *(workflows)* Disable scheduled runs, workflow dispatch, and unused steps in rhiza_validate.yml
## [0.5.1] - 2026-03-29

### 🚀 Features

- License workflow produces LICENSES.md artifact (#590)
- Add _utils subpackage mirroring qs.utils API

### 🐛 Bug Fixes

- Add security exception docs to test_portfolio conftest
- Mark illustrative README snippets +RHIZA_SKIP to pass validate

### 💼 Other

- Bump version 0.5.0 → 0.5.1

### 🚜 Refactor

- Split test_portfolio.py into focused modules
- Split portfolio.py into focused mixin modules (#593)

### 📚 Documentation

- Update CHANGELOG with 0.5.0 release details
- Rewrite README to foreground the Portfolio route and execution-delay analysis

### ⚙️ Miscellaneous Tasks

- Consolidate root-level files into .github/ and docs/ (#591)
## [0.5.0] - 2026-03-28

### 🚀 Features

- *(stats)* Add Omega ratio (#554)
- *(stats)* Add outliers, remove_outliers, outlier_win_ratio, outlier_loss_ratio
- *(stats)* Add comp, compsum, ghpr (#582)
- *(stats)* Add drawdown_details, expected_return, rolling_greeks, t… (#583)

### 🐛 Bug Fixes

- Update Rhiza badge URL in test to reflect GitHub org rename
- Update all remaining tschm → jebel-quant org references

### 💼 Other

- Decompose complex `Reports.metrics()` into focused helpers (#586)
- Bump version 0.4.0 → 0.5.0

### 📚 Documentation

- Update README and paper for v0.4.0
- Update links and badges in README for GitHub organization rename
- Update links and badges in README for GitHub organization rename
- Update links and badges in README for GitHub organization rename
- Add REPOSITORY_ANALYSIS.md via make analyse-repo

### ⚙️ Miscellaneous Tasks

- Merge rhiza_deptry and rhiza_pre-commit into rhiza_quality
- Merge rhiza_docs into rhiza_quality
- Merge rhiza_typecheck into rhiza_validate
- Merge rhiza_pip_audit into rhiza_validate
- Merge rhiza_security into rhiza_validate
- Merge rhiza_semgrep into rhiza_validate
- Consolidate rhiza_validate push/PR jobs into one
- Integrate link_check into rhiza_quality
## [0.4.0] - 2026-03-26

### 🚀 Features

- Autopilot loop — analyse → create issues → solve with Claude (#523)
- Add companion paper and LaTeX CI workflow
- Add portfolio initialization from positions, enhance plotting features, and update snapshots

### 🐛 Bug Fixes

- Raise ValueError when CostModel is constructed with both cost fields non-zero (#528)
- Fix generate_svgs path scope, update coverage badge URL, add from_position tests

### 💼 Other

- Bump version 0.3.4 → 0.4.0

### 📚 Documentation

- Add paper badge linking to compiled PDF on paper branch
- Reorder Figure 1 facades column to report > stats > plots
- Reorder Figure 1 facades column to stats > report > plots
- Update CHANGELOG for v0.2.0 through v0.3.4 and unreleased
- Add 2026-03-26 v0.3.4 repository analysis entry

### ⚙️ Miscellaneous Tasks

- Remove deploy-versioned-docs job from rhiza_docs workflow
- Remove generate_svgs.py from repository
## [0.3.4] - 2026-03-26

### 🚀 Features

- Replace minibook with MkDocs-based book pipeline (#502)
- Add per-subcategory 1–10 scores to analyser agent
- Add vol-normalisation cap and input validation to `from_risk_position` (#521)

### 🐛 Bug Fixes

- Move mkdocs.yml to repo root to resolve docs_dir config error (#501)
- Limit API Reference TOC depth to 3 to reduce clutter

### 💼 Other

- Bump version 0.3.3 → 0.3.4

### 📚 Documentation

- Append 2026-03-25 repository analysis entry (v0.3.3)

### ⚡ Performance

- Vectorise trading_cost_impact — O(1) allocations for the cost sweep (#516)
- Add `slots=True` to `Portfolio` and `Data` frozen dataclasses (#515)

### 🧪 Testing

- Add kaleido static image export tests (#504)
- Add kaleido marker and dedicated CI job for static image export (#520)

### ⚙️ Miscellaneous Tasks

- Move mkdocs.yml to repo root (#514)
## [0.3.3] - 2026-03-24

### 💼 Other

- Bump version 0.3.2 → 0.3.3
## [0.3.2] - 2026-03-24

### 💼 Other

- Bump version 0.3.1 → 0.3.2
## [0.3.1] - 2026-03-24

### 💼 Other

- Bump version 0.3.0 → 0.3.1
## [0.3.0] - 2026-03-24

### 🚀 Features

- Add `__repr__` with date range to `Data` and `Portfolio` (#489)

### 💼 Other

- Add PyPI classifiers and explicit `__all__` (#491)
- Bump version 0.2.0 → 0.3.0

### 📚 Documentation

- Add versioned documentation with mike (#480)
- Add dashboard screenshot, quick-start output, and Marimo badge to README (#481)

### 🧪 Testing

- Increase coverage to 100%

### ⚙️ Miscellaneous Tasks

- Add macOS and Windows to CI test matrix (#482)
## [0.2.0] - 2026-03-23

### 🚀 Features

- Cache stats/plots/report properties on Portfolio and add pytest-cov
- *(packaging)* Add py.typed marker and configure release notes
- *(lint)* Enable ANN2 return-type annotation enforcement in ruff
- Add interrogate to enforce 100% docstring coverage
- Automate release notes with git-cliff in release workflow

### 🐛 Bug Fixes

- Correct README code examples to pass make validate
- Exclude book/ from interrogate docstring coverage
- Exclude book/ from interrogate pre-commit hook

### 💼 Other

- Bump version 0.1.1 → 0.2.0

### 🚜 Refactor

- Extract figure_structure to shared plot_test_utils module, fix ruff formatting
- Group _stats*.py files into _stats/ subpackage

### 📚 Documentation

- Add Data and Stats classes to mkdocs nav
- Rewrite SECURITY.md for jquantstats
- Expand README with comparison table, Mermaid diagram, more examples, badge row
- Add GitHub Discussions templates and update contributing guide
- Add docs/stability.md and update mkdocs nav
- Replace build_data with Data.from_returns in README
- Document uv.lock update requirement in CONTRIBUTING.md
- Add QuantStats migration guide (docs/migration.md)

### 🧪 Testing

- Add test_api_contract.py to guard public API surface

### ⚙️ Miscellaneous Tasks

- Add ISSUE_TEMPLATE config.yml and PULL_REQUEST_TEMPLATE.md
- Enable blank issues in ISSUE_TEMPLATE config
- Update issue templates to be jquantstats-specific
- Exclude generated report.html from git tracking
- Add CITATIONS.bib and Citing section to README
## [0.1.1] - 2026-03-23

### 🚀 Features

- Integer-index first-class support for annual_breakdown
- Add cost_bps construction parameter to unify cost model interface
- Allow per-asset vola dict in from_risk_position
- Forward cost_per_unit and cost_bps through from_risk_position

### 🐛 Bug Fixes

- Correct README pandas claims
- Forward cost_per_unit through all portfolio transforms
- Integer-indexed portfolios use 252 periods/year in _periods_per_year
- Resolve ty type errors in _stats.py and portfolio.py
- Correct README code examples for make validate

### 💼 Other

- Increase coverage to 100%
- Bump version 0.1.0 → 0.1.1

### 🚜 Refactor

- Remove unreachable CleaningInvariantError guards from profits
- Add section dividers to _stats.py
- Remove CleaningInvariantError (dead code since profits guards removed)
- Extract duplicated NAV layout code into _apply_nav_layout helper

### 📚 Documentation

- Document analytics facades, cost models, and integer-index limits
- Clarify build_data / Portfolio relationship in README and docstring
- Add 2026-03-23 repository analysis entry for post-merge main branch

### ⚡ Performance

- Eliminate 21x Data+Stats construction in trading_cost_impact
- Cache Data bridge object in Portfolio to eliminate repeated validation

### ⚙️ Miscellaneous Tasks

- Update REPOSITORY_ANALYSIS.md to 7.5/10 post-refactor analysis
- Update REPOSITORY_ANALYSIS.md to 8.5/10
- Annotate pandas dev dependency as quantstats-only
## [0.0.36] - 2026-03-22

### 💼 Other

- Bump version 0.0.35 → 0.0.36
## [0.0.35] - 2026-03-22

### 💼 Other

- Bump version 0.0.34 → 0.0.35
## [0.0.34] - 2026-03-17

### 💼 Other

- Replace Coveralls badge with rhiza-generated coverage badge
- Use rhiza-tools generate-coverage-badge for shields.io endpoint
- Ensure _book/tests/ exists before generating coverage badge
- Update coverage-badge test assertions to match rhiza-tools implementation
- Restore coverage-badge.json generation for GitHub Pages badge
- Handle pytest exit code 5 in hypothesis-test target
- Use hardcoded rhiza-tools version instead of RHIZA_VERSION
- Remove redundant cast() calls flagged by ty type checker
- Bump version 0.0.33 → 0.0.34

### ⚙️ Miscellaneous Tasks

- Update via rhiza
## [0.0.33] - 2026-02-24

### 🐛 Bug Fixes

- *(deps)* Update dependency pytest to v9.0.1
- *(deps)* Update dependency ipython to v9.8.0
- *(deps)* Update dependency pytest to v9.0.2
- *(deps)* Update dependency yfinance to v1
- Resolve mypy strict mode type errors
- Resolve conftest import conflict in test_rhiza_workflows

### 💼 Other

- Bump version 0.0.32 → 0.0.33

### ⚙️ Miscellaneous Tasks

- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Remove deprecated files
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
## [0.0.32] - 2025-11-16

### 🐛 Bug Fixes

- *(deps)* Update dependency ipython to v9 (#221)
- *(deps)* Update dependency ipython to v9.7.0 (#227)
- *(deps)* Update dependency kaleido to v1.2.0 (#228)
- *(deps)* Update dependency pytest to v9

### ⚙️ Miscellaneous Tasks

- Sync template files (#224)
- Sync template files (#225)
- Sync template files
- Sync template files
## [0.0.29] - 2025-10-26

### 🐛 Bug Fixes

- *(deps)* Update dependency quantstats to v0.0.73 (#165)
- *(deps)* Update dependency quantstats to v0.0.75 (#169)
- *(deps)* Update dependency quantstats to v0.0.76 (#172)
- *(deps)* Update dependency pytest-cov to v6.3.0 (#180)
- *(deps)* Update dependency quantstats to v0.0.77 (#179)
- *(deps)* Update dependency pytest to v8.4.2 (#178)
- *(deps)* Update dependency kaleido to v1.1.0 (#184)
- *(deps)* Update dependency pytest-cov to v7 (#185)
- *(deps)* Update dependency yfinance to v0.2.66 (#190)

### ⚙️ Miscellaneous Tasks

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
## [0.0.28] - 2025-08-11

### 🐛 Bug Fixes

- *(deps)* Update dependency yfinance to v0.2.65 (#127)
- *(deps)* Update dependency quantstats to v0.0.69 (#137)
- *(deps)* Update dependency quantstats to v0.0.70 (#153)

### ⚙️ Miscellaneous Tasks

- Sync config files from .config-templates (#135)
- Sync config files from .config-templates (#138)
- Sync config files from .config-templates (#141)
- Sync config files from .config-templates (#151)
- Sync config files from .config-templates (#157)
- Sync config files from .config-templates (#162)
## [0.0.26] - 2025-07-01

### 🐛 Bug Fixes

- *(deps)* Update dependency yfinance to v0.2.63 (#101)
- *(deps)* Update dependency pytest-cov to v6.2.1 (#102)
- *(deps)* Update dependency pytest to v8.4.1 (#104)
- *(deps)* Update dependency kaleido to v1 (#109)
- *(deps)* Update dependency yfinance to v0.2.64 (#116)
## [0.0.25] - 2025-06-10

### 🐛 Bug Fixes

- *(deps)* Update dependency yfinance to v0.2.62 (#92)
## [0.0.18] - 2025-06-03

### 🐛 Bug Fixes

- *(deps)* Update dependency ipython to v8.37.0 (#70)
- *(deps)* Update dependency pytest to v8.4.0 (#74)
## [0.0.13] - 2025-05-23

### 🐛 Bug Fixes

- *(deps)* Update dependency polars to v1.30.0 (#52)
## [0.0.5] - 2025-05-13

### 🐛 Bug Fixes

- *(deps)* Update dependency ipython to v8.36.0 (#33)
## [0.0.4] - 2025-05-12

### 🐛 Bug Fixes

- *(deps)* Update dependency yfinance to v0.2.61 (#27)
## [0.0.3] - 2025-05-12

### 🐛 Bug Fixes

- *(deps)* Update dependency yfinance to v0.2.59 (#11)
## [0.0.1] - 2025-05-07
