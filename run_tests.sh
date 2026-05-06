#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory where this script lives (works when called as a submodule).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MARKER_EXPRESSION] [-- PYTEST_ARGS...]

Run autox-ci tests. Handles virtualenv creation, dependency installation,
optional .env sourcing, and pytest execution with marker-based test selection.

Arguments:
  MARKER_EXPRESSION    Pytest marker expression (e.g. "autorag and functional and smoke").
                       Passed as pytest -m <expr>. Optional — runs all tests if omitted.

Options:
  --env-file FILE      Source this .env file before running tests. Variables already
                       exported in the shell take precedence (dotenv override=False).
  --extras NAME        pip extras to install (default: test_autorag). Comma-separated
                       for multiple, e.g. "test_autorag,test_rhoai".
  --no-setup           Skip virtualenv creation and pip install (assume deps are ready).
  --dry-run            Print the pytest command without executing it.
  --python CMD         Python interpreter to use (default: python3).
  -h, --help           Show this help message.

Everything after "--" is forwarded to pytest verbatim.

Examples:
  # Run AutoRAG functional smoke tests
  $(basename "$0") --env-file .env "autorag and functional and smoke"

  # Run all integration tests, skip venv setup
  $(basename "$0") --no-setup "integration"

  # Run from a parent repo using autox-ci as a submodule
  ./submodules/autox-ci/$(basename "$0") --env-file my.env "autorag and functional"

  # Pass extra pytest flags (verbose, stop on first failure)
  $(basename "$0") --env-file .env "autorag" -- -v -x
EOF
}

# --- Parse arguments ---
ENV_FILE=""
EXTRAS="test_autorag"
SKIP_SETUP=false
DRY_RUN=false
MARKER_EXPR=""
PYTEST_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --extras)
            EXTRAS="$2"
            shift 2
            ;;
        --no-setup)
            SKIP_SETUP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --)
            shift
            PYTEST_EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "Error: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "$MARKER_EXPR" ]]; then
                MARKER_EXPR="$1"
            else
                echo "Error: unexpected argument '$1' (marker expression already set to '$MARKER_EXPR')" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# --- Source .env file ---
if [[ -n "$ENV_FILE" ]]; then
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "Error: env file not found: $ENV_FILE" >&2
        exit 1
    fi
    echo "Sourcing env file: $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# --- Virtualenv + install ---
if [[ "$SKIP_SETUP" == false ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtualenv at $VENV_DIR ..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    IFS=',' read -ra EXTRAS_ARRAY <<< "$EXTRAS"
    EXTRAS_SPEC=""
    for e in "${EXTRAS_ARRAY[@]}"; do
        e="$(echo "$e" | xargs)"
        if [[ -n "$e" ]]; then
            EXTRAS_SPEC="${EXTRAS_SPEC:+$EXTRAS_SPEC,}$e"
        fi
    done

    echo "Installing autox-ci [${EXTRAS_SPEC}] ..."
    pip install -q -e "${SCRIPT_DIR}[${EXTRAS_SPEC}]"
else
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
fi

# --- Build pytest command ---
PYTEST_CMD=(pytest --rootdir "$SCRIPT_DIR")

if [[ -n "$MARKER_EXPR" ]]; then
    PYTEST_CMD+=(-m "$MARKER_EXPR")
fi

if [[ ${#PYTEST_EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${PYTEST_EXTRA_ARGS[@]}")
fi

# --- Execute ---
# Change into autox-ci so pytest does not pick up conftest.py from a parent repo.
cd "$SCRIPT_DIR"

if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] ${PYTEST_CMD[*]}"
    exit 0
fi

echo "Running: ${PYTEST_CMD[*]}"
exec "${PYTEST_CMD[@]}"
