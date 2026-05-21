#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [MARKER_EXPR] [-- PYTEST_ARGS...]

Run tests via uv + pytest with automatic dependency installation, optional
.env sourcing, and marker-based test selection.

Arguments:
  MARKER_EXPR              Pytest marker expression (e.g. "positive", "negative").
                           Omit to run all tests in the selected suite.

Options:
  -s, --suite SUITE        Test suite to run: automl | autorag | all.
                           Sets the default extras, tags env var, and test path.
  -t, --tags TAGS          Comma-separated tags for scenario filtering (matched
                           against the 'tags' field in test config JSON files).
                           Exported as AUTOML_FUNCTIONAL_TESTS_TAGS for automl,
                           FUNCTIONAL_TESTS_TAGS for autorag (both when --suite all).
  --env-file FILE          Source a .env file before running. May be repeated to
                           source multiple files in order. Shell-exported vars
                           take precedence (dotenv override=False semantics).
  --extras NAME            uv extras to install (overrides suite default).
                           Comma-separated for multiple, e.g. "test_automl,other".
  --tabular-pipeline PATH  Path to compiled tabular pipeline YAML
                           (sets AUTOML_TABULAR_PIPELINE_PATH; AutoML only).
  --timeseries-pipeline PATH
                           Path to compiled timeseries pipeline YAML
                           (sets AUTOML_TIMESERIES_PIPELINE_PATH; AutoML only).
                           AutoRAG pipeline path is set via AUTORAG_PIPELINE_PATH
                           in the env file.
  --rag-configs PATH       Custom test_configs.json for AutoRAG scenarios
                           (sets AUTORAG_TEST_CONFIGS_PATH).
  --tabular-configs PATH   Custom tabular_test_configs.json for AutoML tabular
                           (sets AUTOML_TABULAR_TEST_CONFIGS_PATH).
  --timeseries-configs PATH
                           Custom timeseries_test_configs.json for AutoML
                           time series (sets AUTOML_TIMESERIES_TEST_CONFIGS_PATH).
  --dry-run                Print the pytest command without executing it.
  -h, --help               Show this message and exit.

Everything after "--" is forwarded to pytest verbatim.

Examples:
  # AutoML — all tests
  $(basename "$0") --suite automl --env-file autox_tests/.env.ml

  # AutoML — smoke scenarios only, with local pipeline YAMLs
  $(basename "$0") --suite automl --env-file autox_tests/.env.ml \\
      --tabular-pipeline pipeline.yaml --timeseries-pipeline pipeline_ts.yaml \\
      -t smoke

  # AutoML — positive tabular tests only
  $(basename "$0") --suite automl --env-file autox_tests/.env.ml positive -- \\
      autox_tests/automl/test_tabular_functional.py -v

  # AutoRAG — all tests
  $(basename "$0") --suite autorag --env-file autox_tests/.env.rag

  # AutoRAG — smoke scenarios only
  $(basename "$0") --suite autorag --env-file autox_tests/.env.rag -t smoke

  # Both suites in one run (--tabular/timeseries-pipeline for AutoML only;
  # AutoRAG pipeline path is read from AUTORAG_PIPELINE_PATH in .env.rag)
  $(basename "$0") --suite all \\
      --env-file autox_tests/.env.ml --env-file autox_tests/.env.rag \\
      --tabular-pipeline pipeline.yaml --timeseries-pipeline pipeline_ts.yaml

  # Pass extra pytest flags
  $(basename "$0") --suite automl --env-file autox_tests/.env.ml -- -v -x --no-header

  # Use custom test configs (submodule / downstream repo)
  $(basename "$0") --suite autorag --env-file my.env \\
      --rag-configs my_configs/autorag_scenarios.json

  $(basename "$0") --suite automl --env-file my.env \\
      --tabular-configs my_configs/tabular.json \\
      --timeseries-configs my_configs/timeseries.json
EOF
}

# ── Parse arguments ──────────────────────────────────────────────────────────

SUITE=""
ENV_FILES=()
EXTRAS=""
DRY_RUN=false
MARKER_EXPR=""
TESTS_TAGS=""
TABULAR_PIPELINE=""
TIMESERIES_PIPELINE=""
RAG_CONFIGS=""
TABULAR_CONFIGS=""
TIMESERIES_CONFIGS=""
PYTEST_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -s|--suite)
            SUITE="$2"
            shift 2
            ;;
        -t|--tags)
            TESTS_TAGS="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILES+=("$2")
            shift 2
            ;;
        --extras)
            EXTRAS="$2"
            shift 2
            ;;
        --tabular-pipeline)
            TABULAR_PIPELINE="$2"
            shift 2
            ;;
        --timeseries-pipeline)
            TIMESERIES_PIPELINE="$2"
            shift 2
            ;;
        --rag-configs)
            RAG_CONFIGS="$2"
            shift 2
            ;;
        --tabular-configs)
            TABULAR_CONFIGS="$2"
            shift 2
            ;;
        --timeseries-configs)
            TIMESERIES_CONFIGS="$2"
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

# ── Validate suite ───────────────────────────────────────────────────────────

if [[ -n "$SUITE" && "$SUITE" != "automl" && "$SUITE" != "autorag" && "$SUITE" != "all" ]]; then
    echo "error: --suite must be 'automl', 'autorag', or 'all', got '$SUITE'" >&2
    exit 1
fi

# ── Source .env files ────────────────────────────────────────────────────────

if [[ ${#ENV_FILES[@]} -gt 0 ]]; then
    for ENV_FILE in "${ENV_FILES[@]}"; do
        if [[ ! -f "$ENV_FILE" ]]; then
            echo "error: env file not found: $ENV_FILE" >&2
            exit 1
        fi
        echo "env: sourcing $ENV_FILE"
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
    done
fi

# ── Apply pipeline path overrides (automl) ───────────────────────────────────

if [[ -n "$TABULAR_PIPELINE" ]]; then
    export AUTOML_TABULAR_PIPELINE_PATH="$TABULAR_PIPELINE"
fi
if [[ -n "$TIMESERIES_PIPELINE" ]]; then
    export AUTOML_TIMESERIES_PIPELINE_PATH="$TIMESERIES_PIPELINE"
fi

# ── Apply custom test config overrides ───────────────────────────────────────

for _pair in \
    "RAG_CONFIGS:AUTORAG_TEST_CONFIGS_PATH:--rag-configs" \
    "TABULAR_CONFIGS:AUTOML_TABULAR_TEST_CONFIGS_PATH:--tabular-configs" \
    "TIMESERIES_CONFIGS:AUTOML_TIMESERIES_TEST_CONFIGS_PATH:--timeseries-configs"; do
    IFS=: read -r _var _env _flag <<< "$_pair"
    _val="${!_var}"
    if [[ -n "$_val" ]]; then
        if [[ ! -f "$_val" ]]; then
            echo "error: ${_flag} file not found: $_val" >&2
            exit 1
        fi
        export "$_env"="$_val"
        echo "config: using custom $_flag from $_val"
    fi
done

# ── Build command ────────────────────────────────────────────────────────────

# Set suite-specific defaults
case "$SUITE" in
    automl)
        EXTRAS="${EXTRAS:-test_automl}"
        if [[ -n "$TESTS_TAGS" ]]; then
            export AUTOML_FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
        fi
        ;;
    autorag)
        EXTRAS="${EXTRAS:-test_autorag}"
        if [[ -n "$TESTS_TAGS" ]]; then
            export FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
        fi
        ;;
    all)
        EXTRAS="${EXTRAS:-test_automl,test_autorag}"
        if [[ -n "$TESTS_TAGS" ]]; then
            export AUTOML_FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
            export FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
        fi
        ;;
    *)
        EXTRAS="${EXTRAS:-test_autorag}"
        ;;
esac

PYTEST_CMD=(uv run --project "$SCRIPT_DIR")

IFS=',' read -ra EXTRAS_ARRAY <<< "$EXTRAS"
for e in "${EXTRAS_ARRAY[@]}"; do
    e="$(echo "$e" | xargs)"
    [[ -n "$e" ]] && PYTEST_CMD+=(--extra "$e")
done

PYTEST_CMD+=(pytest --rootdir "$SCRIPT_DIR")

[[ -n "$MARKER_EXPR" ]] && PYTEST_CMD+=(-m "$MARKER_EXPR")

# Append suite test path(s) before user-supplied pytest args
case "$SUITE" in
    automl)  PYTEST_CMD+=(autox_tests/automl/) ;;
    autorag) PYTEST_CMD+=(autox_tests/autorag/) ;;
    all)     PYTEST_CMD+=(autox_tests/automl/ autox_tests/autorag/) ;;
esac

if [[ ${#PYTEST_EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${PYTEST_EXTRA_ARGS[@]}")
fi

# ── Execute ──────────────────────────────────────────────────────────────────

cd "$SCRIPT_DIR"

DISPLAY_PREFIX=""
if [[ -n "$TESTS_TAGS" ]]; then
    case "$SUITE" in
        automl)  DISPLAY_PREFIX="AUTOML_FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" " ;;
        autorag) DISPLAY_PREFIX="FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" " ;;
        all)     DISPLAY_PREFIX="AUTOML_FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" " ;;
    esac
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] ${DISPLAY_PREFIX}${PYTEST_CMD[*]}"
    exit 0
fi

echo "exec: ${DISPLAY_PREFIX}${PYTEST_CMD[*]}"
exec "${PYTEST_CMD[@]}"
