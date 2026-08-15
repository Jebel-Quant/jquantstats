#!/usr/bin/env bash
# Run a jquantstats plugin script under the first Python that can import the library.
#
# Resolution order: the active virtualenv, a repo-local .venv, whatever `python3`
# resolves to, then `uv run --with jquantstats` as a last resort. Without this the
# plugin only works from inside a repo that already has jquantstats installed.
#
# Usage: jqs.sh jqs_load.py portfolio --prices ... | jqs.sh jqs_api.py --show sharpe
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

name="${1:-}"
if [[ -z "${name}" ]]; then
    echo "usage: jqs.sh <jqs_load.py|jqs_api.py> [args...]" >&2
    exit 64
fi
shift

target="${here}/${name}"
if [[ ! -f "${target}" ]]; then
    echo "no such plugin script: ${name}" >&2
    exit 66
fi

can_import() {
    "$1" -c 'import jquantstats, polars' >/dev/null 2>&1
}

candidates=()
[[ -n "${VIRTUAL_ENV:-}" ]] && candidates+=("${VIRTUAL_ENV}/bin/python")
candidates+=("./.venv/bin/python" "python3" "python")

for candidate in "${candidates[@]}"; do
    if command -v "${candidate}" >/dev/null 2>&1 && can_import "${candidate}"; then
        exec "${candidate}" "${target}" "$@"
    fi
done

if command -v uv >/dev/null 2>&1; then
    exec uv run --quiet --with jquantstats --with polars python "${target}" "$@"
fi

echo "no Python with jquantstats available — install jquantstats, or install uv" >&2
exit 69
