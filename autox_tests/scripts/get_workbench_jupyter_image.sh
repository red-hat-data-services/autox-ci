#!/usr/bin/env bash
# Print the RHOAI Py312 Data Science workbench image from the installed CSV.
set -euo pipefail

if ! oc whoami >/dev/null 2>&1; then
    echo "ERROR: log in to the target OpenShift cluster with oc before running this script" >&2
    exit 1
fi

WORKBENCH_IMAGE="$(oc get csv -n redhat-ods-operator -o json | \
    jq -r '[.items[].spec.install.spec.deployments[].spec.template.spec.containers[].env[]? | select(.name=="RELATED_IMAGE_ODH_WORKBENCH_JUPYTER_DATASCIENCE_CPU_PY312_IMAGE") | .value][0]')"

if [[ -z "${WORKBENCH_IMAGE}" || "${WORKBENCH_IMAGE}" == "null" ]]; then
    echo "ERROR: RELATED_IMAGE_ODH_WORKBENCH_JUPYTER_DATASCIENCE_CPU_PY312_IMAGE not found in RHOAI CSV" >&2
    echo "Installed CSVs in redhat-ods-operator:" >&2
    oc get csv -n redhat-ods-operator --no-headers >&2
    exit 1
fi

echo "${WORKBENCH_IMAGE}"
