"""Typed TLS scenarios and environment settings, like the AutoML/AutoRAG suites.

Scenario catalogs are tracked in this directory. Environment values are resolved
only for selected tests, so collecting scenarios never contacts the cluster.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

from autox_tests.lib.tls.settings import (
    MissingTLSConfiguration,
    endpoint_credentials,
    service_ca,
    timeout_seconds,
)

_CONFIGS_DIR = Path(__file__).parent


def _read_catalog(kind):
    path = _CONFIGS_DIR / f"{kind}_test_configs.json"
    data = json.loads(path.read_text())
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError(f"{path} must contain a list of scenario objects")
    ids = [item.get("id") for item in data]
    if any(not isinstance(value, str) or not value for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError(f"{path} requires non-empty, unique scenario IDs")
    return data


def _filter_by_tags(configs):
    requested = {
        tag.strip().lower()
        for tag in os.environ.get("AUTORAG_FUNCTIONAL_TESTS_TAGS", "").split(",")
        if tag.strip()
    }
    return [c for c in configs if requested <= {tag.lower() for tag in c.tags}]


@dataclass(frozen=True)
class TLSRuntimeConfig:
    """One runtime/DSPA scenario; payload excludes pytest selection tags."""

    __test__ = False

    id: str
    engine: str = "maas"
    trust: str = "mounted"
    expected: str = "success"
    redirect: bool = False
    certificate: str = "server"
    dspa: bool = False
    dspa_mode: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.engine not in {
            "maas",
            "agent",
            "direct",
            "s3",
            "milvus",
            "pgvector",
            "neo4j",
        }:
            raise ValueError(f"Unsupported TLS engine: {self.engine}")
        if self.expected not in {"success", "tls_error", "ca_error"}:
            raise ValueError(f"Unsupported TLS outcome: {self.expected}")
        if self.certificate not in {"server", "wrong-host", "expired"}:
            raise ValueError(f"Unsupported TLS certificate variant: {self.certificate}")
        if self.dspa and self.dspa_mode not in {"platform", "plain-dir"}:
            raise ValueError("DSPA requires platform or plain-dir mode")

    def payload(self):
        data = asdict(self)
        data.pop("tags")
        if not self.dspa:
            data.pop("dspa")
            data.pop("dspa_mode")
        return data


def get_runtime_configs(kind, *, filter_tags=True):
    configs = [TLSRuntimeConfig(**item) for item in _read_catalog(kind)]
    return _filter_by_tags(configs) if filter_tags else configs


def resolve_service_profile(test_config, *, backend_secret=None, core=None, namespace=None):
    """Reuse an existing Secret or standard endpoint settings from .env.rag."""
    local_certificate_fixture = test_config.engine in {
        "s3",
        "milvus",
    } and test_config.certificate in {"wrong-host", "expired"}
    local_s3_fixture = local_certificate_fixture and test_config.engine == "s3"
    local_milvus_fixture = local_certificate_fixture and test_config.engine == "milvus"
    if local_certificate_fixture:
        secret, credentials, ca = None, {}, {}
    else:
        secret, credentials = endpoint_credentials(
            test_config.engine, test_config.certificate, backend_secret=backend_secret
        )
        ca = service_ca(
            test_config.engine, test_config.certificate,
            backend_secret=backend_secret, core=core, namespace=namespace,
        )
    environment = {
        "INPUT_DATA_BUCKET_NAME": os.environ.get("INPUT_DATA_BUCKET_NAME", "").strip(),
    }
    if test_config.engine != "s3":
        environment = {}
    if local_s3_fixture:
        environment = {"INPUT_DATA_BUCKET_NAME": "tls-fixture"}
    if test_config.engine == "s3" and not local_s3_fixture:
        missing = {
            name for name, value in environment.items() if not value
        }
        if missing:
            raise ValueError(
                "Set "
                + ", ".join(sorted(missing))
                + " for the automatically uploaded S3 probe object"
            )
    image = os.environ.get("AUTOX_TLS_AUTORAG_IMAGE", "").strip()
    if not image:
        raise ValueError("AUTOX_TLS_AUTORAG_IMAGE is required for runtime probes")
    return {
        "id": test_config.engine + "-" + test_config.certificate,
        "engine": test_config.engine,
        "local_s3_fixture": local_s3_fixture,
        "local_milvus_fixture": local_milvus_fixture,
        "image": image,
        "secret_name": secret,
        "credentials": credentials,
        **ca,
        "environment": environment,
        "s3_key": "probe.txt" if local_s3_fixture else None,
        "timeout": timeout_seconds(180, "RHOAI_NOTEBOOK_JOB_TIMEOUT"),
    }


def resolve_environment_values(value, secret_resolver=None, dataset_values=None):
    """Resolve environment, generated dataset and credential references without eval."""
    if isinstance(value, list):
        return [
            resolve_environment_values(item, secret_resolver, dataset_values)
            for item in value
        ]
    if isinstance(value, dict):
        if "secret" in value:
            if secret_resolver is None:
                raise MissingTLSConfiguration("Credential resolver is required")
            return secret_resolver(value["secret"])
        if "data" in value:
            name = value["data"]
            if dataset_values is None or name not in dataset_values:
                raise ValueError("Generated TLS dataset is required")
            return dataset_values[name]
        if "env" not in value:
            return {
                key: resolve_environment_values(item, secret_resolver, dataset_values)
                for key, item in value.items()
            }
        name = value["env"]
        raw = os.environ.get(name, "").strip()
        if not raw:
            if "default" in value:
                return value["default"]
            raise MissingTLSConfiguration(f"Set {name} in .env.rag")
        if value.get("format") == "list":
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = [part.strip() for part in raw.split(",") if part.strip()]
            if (
                not isinstance(result, list)
                or not result
                or any(not isinstance(item, str) or not item for item in result)
            ):
                raise ValueError(f"{name} must be a non-empty list of strings")
            return result
        return raw
    return value


@dataclass(frozen=True)
class TLSFlowConfig:
    """One tracked product-pipeline scenario, resolved by its session fixtures."""

    __test__ = False

    id: str
    family: str
    expected: str
    tags: list[str]
    definition: dict = field(repr=False)

    def resolve(self, secret_resolver=None, dataset_values=None):
        from autox_tests.lib.tls.flows import validate_flows

        flow = resolve_environment_values(
            self.definition, secret_resolver, dataset_values
        )
        validate_flows([flow])
        return flow


def get_flow_configs(*, filter_tags=True):
    configs = []
    for item in _read_catalog("flow"):
        if (
            item.get("family") != "autorag"
            or item.get("expected") != "tls_error"
            or item.get("fault") != "maas_wrong_ca"
        ):
            raise ValueError("TLS product flows require a negative AutoRAG CA fault")
        tags = item.get(
            "tags",
            [
                "pipeline",
                item["family"],
                "positive" if item["expected"] == "success" else "negative",
            ],
        )
        configs.append(
            TLSFlowConfig(item["id"], item["family"], item["expected"], tags, item)
        )
    return _filter_by_tags(configs) if filter_tags else configs
