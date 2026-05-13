#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MARKER_EXPR] [-- PYTEST_ARGS...]

Run tests via uv + pytest with automatic dependency installation, optional
.env sourcing, and marker-based test selection.

Arguments:
  MARKER_EXPR          Pytest marker expression (e.g. "autorag and positive").
                       Omit to run all registered tests.

Options:
  -t, --tags TAGS      Comma-separated tags for scenario filtering (matched
                       against the 'tags' field in test config JSON files).
  --env-file FILE      Source a .env file before running. Shell-exported vars
                       take precedence (dotenv override=False semantics).
  --extras NAME        uv extras to install (default: test_autorag).
                       Comma-separated for multiple, e.g. "test_autorag,test_rhoai".
  --dry-run            Print the pytest command without executing it.
  -h, --help           Show this message and exit.

Everything after "--" is forwarded to pytest verbatim.

Examples:
  $(basename "$0") --env-file autox_tests/.env.rag "autorag and positive"
  $(basename "$0") --env-file autox_tests/.env.rag -t smoke "autorag"
  $(basename "$0") "autorag" -- -v -x --no-header
  $(basename "$0") --dry-run "autorag and negative"
EOF
}

# ── Parse arguments ──────────────────────────────────────────────────────────

ENV_FILE=""
EXTRAS="test_autorag"
DRY_RUN=false
MARKER_EXPR=""
TESTS_TAGS=""
PYTEST_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--tags)
            TESTS_TAGS="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --extras)
            EXTRAS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --)
            shift
            PYTEST_EXTRA_ARGS=("$@")
            break
            ;;
        -*)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "$MARKER_EXPR" ]]; then
                MARKER_EXPR="$1"
            else
                echo "error: unexpected argument '$1' (marker already set to '$MARKER_EXPR')" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# ── Source .env file ─────────────────────────────────────────────────────────

if [[ -n "$ENV_FILE" ]]; then
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "error: env file not found: $ENV_FILE" >&2
        exit 1
    fi
    echo "env: sourcing $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# ── Build command ────────────────────────────────────────────────────────────

PYTEST_CMD=(uv run --project "$SCRIPT_DIR")

IFS=',' read -ra EXTRAS_ARRAY <<< "$EXTRAS"
for e in "${EXTRAS_ARRAY[@]}"; do
    e="$(echo "$e" | xargs)"
    [[ -n "$e" ]] && PYTEST_CMD+=(--extra "$e")
done

PYTEST_CMD+=(pytest --rootdir "$SCRIPT_DIR")

[[ -n "$MARKER_EXPR" ]] && PYTEST_CMD+=(-m "$MARKER_EXPR")

if [[ -n "$TESTS_TAGS" ]]; then
    export TESTS_TAGS
fi

if [[ ${#PYTEST_EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${PYTEST_EXTRA_ARGS[@]}")
fi

# ── Execute ──────────────────────────────────────────────────────────────────

cd "$SCRIPT_DIR"

DISPLAY_PREFIX=""
[[ -n "$TESTS_TAGS" ]] && DISPLAY_PREFIX="TESTS_TAGS=\"$TESTS_TAGS\" "

if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] ${DISPLAY_PREFIX}${PYTEST_CMD[*]}"
    exit 0
fi

echo "exec: ${DISPLAY_PREFIX}${PYTEST_CMD[*]}"
exec "${PYTEST_CMD[@]}"
