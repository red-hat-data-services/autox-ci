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
  -s, --suite SUITE        Test suite to run: automl | autorag | tls | all.
                           tls selects certificate tests in each product suite,
                           using its own .env file in a separate process.
  -t, --tags TAGS          Comma-separated tags for scenario filtering (matched
                           against the 'tags' field in test config JSON files).
                           Exported as AUTOML_FUNCTIONAL_TESTS_TAGS for automl,
                           AUTORAG_FUNCTIONAL_TESTS_TAGS for autorag.
                           tls/all use the corresponding product's variable.
  --env-file FILE          Source a .env file before running. May be repeated to
                           source multiple files in order. Shell-exported vars
                           take precedence (dotenv override=False semantics).
  --legacy-pipeline-yaml   Upload pipeline.yaml per run (sets RHOAI_USE_MANAGED_PIPELINES=false).
                           Default: use pipelines from DSPA managedPipelines (no YAML paths).
  --tabular-pipeline PATH  With --legacy-pipeline-yaml: tabular package path
                           (sets AUTOML_TABULAR_PIPELINE_PATH).
  --timeseries-pipeline PATH
                           With --legacy-pipeline-yaml: timeseries package path
                           (sets AUTOML_TIMESERIES_PIPELINE_PATH).
                           AutoRAG legacy: AUTORAG_PIPELINE_PATH in the env file.
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
For --suite tls, pass marker expressions before "--"; extra pytest -m is rejected.

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
DRY_RUN=false
MARKER_EXPR=""
TESTS_TAGS=""
LEGACY_PIPELINE_YAML=false
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
        --legacy-pipeline-yaml)
            LEGACY_PIPELINE_YAML=true
            shift
            ;;
        --managed-pipelines)
            echo "warning: --managed-pipelines is deprecated (managed mode is the default)" >&2
            shift
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

if [[ -n "$SUITE" && "$SUITE" != "automl" && "$SUITE" != "autorag" && "$SUITE" != "tls" && "$SUITE" != "all" ]]; then
    echo "error: --suite must be 'automl', 'autorag', 'tls', or 'all', got '$SUITE'" >&2
    exit 1
fi

# Run the TLS subsets separately so .env.ml and .env.rag cannot overwrite
# each other's shared setting names. A product participates when it has tls/.
if [[ "$SUITE" == "tls" ]]; then
    for PYTEST_ARG in ${PYTEST_EXTRA_ARGS[@]+"${PYTEST_EXTRA_ARGS[@]}"}; do
        [[ "$PYTEST_ARG" == -- ]] && break
        case "$PYTEST_ARG" in
            -m*)
                echo 'error: --suite tls does not accept pytest -m after --; pass the marker expression before -- to combine it with tls' >&2
                exit 1
                ;;
        esac
    done
    TLS_ARGS=()
    for ENV_FILE in ${ENV_FILES[@]+"${ENV_FILES[@]}"}; do
        TLS_ARGS+=(--env-file "$ENV_FILE")
    done
    [[ "$DRY_RUN" == true ]] && TLS_ARGS+=(--dry-run)
    [[ "$LEGACY_PIPELINE_YAML" == true ]] && TLS_ARGS+=(--legacy-pipeline-yaml)
    for _pair in \
        "TESTS_TAGS:--tags" \
        "TABULAR_PIPELINE:--tabular-pipeline" \
        "TIMESERIES_PIPELINE:--timeseries-pipeline" \
        "RAG_CONFIGS:--rag-configs" \
        "TABULAR_CONFIGS:--tabular-configs" \
        "TIMESERIES_CONFIGS:--timeseries-configs"; do
        IFS=: read -r _var _flag <<< "$_pair"
        [[ -n "${!_var}" ]] && TLS_ARGS+=("$_flag" "${!_var}")
    done
    TLS_MARKER=tls
    [[ -n "$MARKER_EXPR" ]] && TLS_MARKER="tls and ($MARKER_EXPR)"
    TLS_PRODUCTS=()
    for PRODUCT_SUITE in automl autorag; do
        [[ -d "$SCRIPT_DIR/autox_tests/$PRODUCT_SUITE/tls" ]] && TLS_PRODUCTS+=("$PRODUCT_SUITE")
    done
    if [[ ${#TLS_PRODUCTS[@]} -eq 0 ]]; then
        echo "error: no product TLS directories found" >&2
        exit 5
    fi
    tls_report_path() {
        local report_path="$1" product="$2"
        if [[ "$report_path" == */* ]]; then
            printf '%s/%s-%s' "${report_path%/*}" "$product" "${report_path##*/}"
        else
            printf '%s-%s' "$product" "$report_path"
        fi
    }
    TLS_STATUS=5
    for PRODUCT_SUITE in "${TLS_PRODUCTS[@]}"; do
        PRODUCT_PYTEST_ARGS=()
        JUNIT_PATH_NEXT=false
        for PYTEST_ARG in ${PYTEST_EXTRA_ARGS[@]+"${PYTEST_EXTRA_ARGS[@]}"}; do
            if [[ ${#TLS_PRODUCTS[@]} -gt 1 && "$JUNIT_PATH_NEXT" == true ]]; then
                PYTEST_ARG="$(tls_report_path "$PYTEST_ARG" "$PRODUCT_SUITE")"
                JUNIT_PATH_NEXT=false
            elif [[ ${#TLS_PRODUCTS[@]} -gt 1 ]]; then
                case "$PYTEST_ARG" in
                    --junitxml|--junit-xml) JUNIT_PATH_NEXT=true ;;
                    --junitxml=*|--junit-xml=*)
                        PYTEST_ARG="${PYTEST_ARG%%=*}=$(tls_report_path "${PYTEST_ARG#*=}" "$PRODUCT_SUITE")"
                        ;;
                esac
            fi
            PRODUCT_PYTEST_ARGS+=("$PYTEST_ARG")
        done
        echo "TLS: running $PRODUCT_SUITE with its product environment"
        if bash "$SCRIPT_DIR/run_tests.sh" --suite "$PRODUCT_SUITE" \
            ${TLS_ARGS[@]+"${TLS_ARGS[@]}"} "$TLS_MARKER" -- \
            ${PRODUCT_PYTEST_ARGS[@]+"${PRODUCT_PYTEST_ARGS[@]}"}; then
            [[ "$TLS_STATUS" == 5 ]] && TLS_STATUS=0
        else
            PRODUCT_STATUS=$?
            if [[ "$PRODUCT_STATUS" != 5 ]]; then
                TLS_STATUS=$PRODUCT_STATUS
            fi
        fi
    done
    exit "$TLS_STATUS"
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

# ── DSPA auto-setup (default when RHOAI_KFP_URL is unset in .env) ───────────

if [[ "$MARKER_EXPR" == tls || "$MARKER_EXPR" == tls\ and\ * ]]; then
    export RHOAI_CREATE_DSPA=false
    echo "cluster: TLS uses existing namespace/DSPA; automatic setup disabled"
elif [[ -z "${RHOAI_KFP_URL:-}" && -z "${KFP_HOST:-}" ]]; then
    case "${RHOAI_CREATE_DSPA:-}" in
        false|0|no|off) ;;  # user explicitly disabled
        *) export RHOAI_CREATE_DSPA=true
           echo "cluster: will create DSPA (RHOAI_KFP_URL unset; set RHOAI_CREATE_DSPA=false to disable)" ;;
    esac
fi

# ── Pipeline source: managed KFP (default) vs legacy YAML paths ───────────────

if [[ "$LEGACY_PIPELINE_YAML" == true ]]; then
    export RHOAI_USE_MANAGED_PIPELINES=false
    echo "pipeline: legacy package upload (RHOAI_USE_MANAGED_PIPELINES=false)"
elif [[ -n "${AUTOML_TABULAR_PIPELINE_PATH:-}" || -n "${AUTOML_TIMESERIES_PIPELINE_PATH:-}" || -n "${AUTORAG_PIPELINE_PATH:-}" ]]; then
    export RHOAI_USE_MANAGED_PIPELINES=false
    echo "pipeline: legacy package upload (pipeline path set in environment)"
elif [[ -z "${RHOAI_USE_MANAGED_PIPELINES:-}" ]]; then
    if [[ "$MARKER_EXPR" == tls || "$MARKER_EXPR" == tls\ and\ * ]]; then
        # Let pytest load the product's env before resolving the pipeline source.
        echo "pipeline: TLS resolves managed/YAML settings from the product .env"
    else
        export RHOAI_USE_MANAGED_PIPELINES=true
        echo "pipeline: DSPA managed pipelines (default; no pipeline.yaml required)"
    fi
fi
if [[ -n "$TABULAR_PIPELINE" ]]; then
    export AUTOML_TABULAR_PIPELINE_PATH="$TABULAR_PIPELINE"
    export RHOAI_USE_MANAGED_PIPELINES=false
fi
if [[ -n "$TIMESERIES_PIPELINE" ]]; then
    export AUTOML_TIMESERIES_PIPELINE_PATH="$TIMESERIES_PIPELINE"
    export RHOAI_USE_MANAGED_PIPELINES=false
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
        if [[ -n "$TESTS_TAGS" ]]; then
            export AUTOML_FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
        fi
        ;;
    autorag)
        if [[ -n "$TESTS_TAGS" ]]; then
            export AUTORAG_FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
        fi
        ;;
    all)
        if [[ -n "$TESTS_TAGS" ]]; then
            export AUTOML_FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
            export AUTORAG_FUNCTIONAL_TESTS_TAGS="$TESTS_TAGS"
        fi
        ;;
    *) ;;
esac

# A TLS-only invocation stays within that product's TLS directory even if an
# additional pytest -m option is supplied after --.
if [[ "$MARKER_EXPR" == tls || "$MARKER_EXPR" == tls\ and\ * ]]; then
    PYTEST_TLS_PATH="autox_tests/$SUITE/tls/"
else
    PYTEST_TLS_PATH=""
fi

PYTEST_CMD=(uv run --project "$SCRIPT_DIR")

# Custom markers can select TLS tests, including overrides forwarded to pytest.
TLS_EXTRA_NEEDED=false
[[ -n "$MARKER_EXPR" ]] && TLS_EXTRA_NEEDED=true
for PYTEST_ARG in ${PYTEST_EXTRA_ARGS[@]+"${PYTEST_EXTRA_ARGS[@]}"}; do
    [[ "$PYTEST_ARG" == -- ]] && break
    case "$PYTEST_ARG" in
        -m*) TLS_EXTRA_NEEDED=true ;;
    esac
done
[[ "$TLS_EXTRA_NEEDED" == true ]] && PYTEST_CMD+=(--extra tls)

PYTEST_CMD+=(pytest --rootdir "$SCRIPT_DIR")

if [[ -n "$MARKER_EXPR" ]]; then
    PYTEST_CMD+=(-m "$MARKER_EXPR")
elif [[ "$SUITE" == "automl" ]]; then
    PYTEST_CMD+=(-m "not config and not tls")
else
    PYTEST_CMD+=(-m "not tls")
fi

# Append suite test path(s) before user-supplied pytest args
if [[ -n "$PYTEST_TLS_PATH" && ( "$SUITE" == automl || "$SUITE" == autorag ) ]]; then
    PYTEST_CMD+=("$PYTEST_TLS_PATH")
else
    case "$SUITE" in
        automl)  PYTEST_CMD+=(autox_tests/automl/) ;;
        autorag) PYTEST_CMD+=(autox_tests/autorag/) ;;
        all)     PYTEST_CMD+=(autox_tests/automl/ autox_tests/autorag/) ;;
    esac
fi

if [[ ${#PYTEST_EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${PYTEST_EXTRA_ARGS[@]}")
fi

# ── Execute ──────────────────────────────────────────────────────────────────

cd "$SCRIPT_DIR"

DISPLAY_PREFIX=""
if [[ -n "$TESTS_TAGS" ]]; then
    case "$SUITE" in
        automl)  DISPLAY_PREFIX="AUTOML_FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" " ;;
        autorag) DISPLAY_PREFIX="AUTORAG_FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" " ;;
        all)     DISPLAY_PREFIX="AUTOML_FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" AUTORAG_FUNCTIONAL_TESTS_TAGS=\"$TESTS_TAGS\" " ;;
    esac
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] ${DISPLAY_PREFIX}${PYTEST_CMD[*]}"
    exit 0
fi

echo "exec: ${DISPLAY_PREFIX}${PYTEST_CMD[*]}"
exec "${PYTEST_CMD[@]}"
