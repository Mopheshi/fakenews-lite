#!/usr/bin/env bash
# Runs the CI pipeline locally, mirroring .github/workflows/ci.yml.
#
# Executes the same steps GitHub Actions would: install, fetch NLTK resources,
# lint, test. Useful when Actions is unavailable (billing, outage, fork
# restrictions) or to check a change before pushing.
#
# Unlike CI, this exercises only one interpreter, not the 3.11/3.12/3.13 matrix.
#
# Usage:
#   ./scripts/ci-local.sh                 # use .venv, install first
#   ./scripts/ci-local.sh --skip-install  # environment already provisioned
#   PYTHON=python3.12 ./scripts/ci-local.sh

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Prefer an explicit PYTHON, then the project venv (Windows or POSIX layout).
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
    python_bin=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
else
    python_bin="python3"
fi

skip_install=0
[[ "${1:-}" == "--skip-install" ]] && skip_install=1

if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
    printf '\033[31mInterpreter not found: %s\033[0m\n' "$python_bin"
    exit 1
fi

version="$("$python_bin" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "Repository : $repo_root"
echo "Interpreter: $python_bin ($version)"

failed=()

step() {
    local name="$1"; shift
    printf '\n\033[36m=== %s ===\033[0m\n' "$name"
    if "$@"; then
        printf '\033[32mPASS  %s\033[0m\n' "$name"
    else
        printf '\033[31mFAIL  %s\033[0m\n' "$name"
        failed+=("$name")
    fi
}

if [[ $skip_install -eq 0 ]]; then
    step "Install" bash -c "'$python_bin' -m pip install --upgrade pip --quiet && '$python_bin' -m pip install -e '.[app,bench,dev]' --quiet"
fi

step "NLTK resources" "$python_bin" -c \
    "from fakenews.nltk_resources import ensure_nltk_resources; ensure_nltk_resources()"

step "Lint" "$python_bin" -m ruff check .

step "Test" "$python_bin" -m pytest -q

echo
if [[ ${#failed[@]} -gt 0 ]]; then
    printf '\033[31mCI FAILED: %s\033[0m\n' "${failed[*]}"
    exit 1
fi

printf '\033[32mCI PASSED (Python %s)\033[0m\n' "$version"
printf '\033[33mNote: CI covers 3.11, 3.12 and 3.13; this run exercised %s only.\033[0m\n' "$version"
