"""Resolve reusable TLS settings without touching cluster resources."""

import json
import os
from pathlib import Path
import ssl


class CredentialValues(dict):
    """Avoid exposing credentials in pytest fixture/argument representations."""

    def __repr__(self):
        return "<redacted credentials>"


class MissingTLSConfiguration(ValueError):
    """An optional external-service scenario has not been configured."""


_ENDPOINTS = {
    "s3": (
        ("AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        ("AWS_DEFAULT_REGION",),
        {"AWS_DEFAULT_REGION": "us-east-1"},
    ),
    "maas": (("MAAS_BASE_URL", "MAAS_API_KEY"), (), {}),
    "milvus": (("MILVUS_URI",), ("MILVUS_TOKEN",), {}),
    "pgvector": (
        ("PGVECTOR_HOST", "PGVECTOR_PASSWORD"),
        ("PGVECTOR_PORT", "PGVECTOR_DB", "PGVECTOR_USER"),
        {
            "PGVECTOR_PORT": "5432",
            "PGVECTOR_DB": "postgres",
            "PGVECTOR_USER": "postgres",
        },
    ),
    "neo4j": (
        ("NEO4J_URI", "NEO4J_PASSWORD"),
        ("NEO4J_USERNAME", "NEO4J_DATABASE"),
        {"NEO4J_USERNAME": "neo4j", "NEO4J_DATABASE": "neo4j"},
    ),
}


def service_prefix(engine, certificate="server"):
    if engine in {"s3", "milvus"} and certificate != "server":
        raise ValueError("S3/Milvus certificate fixtures are generated automatically")
    suffix = (
        "" if certificate == "server" else "_" + certificate.upper().replace("-", "_")
    )
    return f"AUTOX_TLS_{engine.upper()}{suffix}"


def endpoint_credentials(engine, certificate="server"):
    """Prefer an explicit existing Secret; otherwise use standard .env keys once."""
    prefix = service_prefix(engine, certificate)
    secret_setting = "MAAS_SECRET_NAME" if engine == "maas" else prefix + "_SECRET"
    name = os.environ.get(secret_setting, "").strip()
    if name:
        return name, {}
    # Bad-host/expired cases must never accidentally target the healthy endpoint.
    if certificate != "server":
        raise MissingTLSConfiguration(
            f"Set {prefix}_SECRET for the separate endpoint fixture"
        )
    required, optional, defaults = _ENDPOINTS[engine]
    if not any(os.environ.get(key, "").strip() for key in required):
        raise MissingTLSConfiguration(f"Set {' / '.join(required)} or {secret_setting}")
    missing = [key for key in required if not os.environ.get(key, "").strip()]
    if missing:
        raise ValueError("Missing endpoint settings: " + ", ".join(missing))
    values = CredentialValues(
        {
            key: os.environ[key]
            for key in required + optional
            if os.environ.get(key, "").strip()
        }
    )
    for key, value in defaults.items():
        values.setdefault(key, value)
    if engine == "s3" and not values["AWS_S3_ENDPOINT"].startswith("https://"):
        raise ValueError("AWS_S3_ENDPOINT must use HTTPS for TLS tests")
    if engine == "maas" and not values["MAAS_BASE_URL"].startswith("https://"):
        raise ValueError("MAAS_BASE_URL must use HTTPS for TLS tests")
    return None, values


def read_ca(value, setting):
    """Read a caller-trusted CA file/PEM; never fetch trust from the target server."""
    try:
        pem = (
            value
            if value.lstrip().startswith("-----BEGIN CERTIFICATE-----")
            else Path(value).expanduser().read_text()
        )
        ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=pem)
    except (OSError, ValueError, ssl.SSLError, UnicodeError):
        raise ValueError(
            f"{setting} must identify a readable CA PEM file or contain valid CA PEM"
        ) from None
    return pem


def ca_setting_names(engine):
    specific = (
        "MILVUS_SERVER_CERT"
        if engine == "milvus"
        else ("AWS_CA_BUNDLE" if engine == "s3" else engine.upper() + "_CA_CERT")
    )
    return tuple(dict.fromkeys((specific, "AWS_CA_BUNDLE")))


def service_ca(engine, certificate="server"):
    prefix = service_prefix(engine, certificate)
    name = os.environ.get(prefix + "_CA_CONFIGMAP", "").strip()
    if name:
        return {
            "ca_configmap": name,
            "ca_key": os.environ.get(prefix + "_CA_KEY") or "ca-bundle.crt",
        }
    specific = ca_setting_names(engine)[0]
    for setting in ca_setting_names(engine):
        value = os.environ.get(setting, "").strip()
        if value:
            return {"ca_pem": read_ca(value, setting)}
    raise ValueError(
        f"Set {specific} or AWS_CA_BUNDLE, or reference {prefix}_CA_CONFIGMAP"
    )


def model_id(kind):
    name = f"AUTORAG_{kind}_MODELS"
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            values = [item.strip() for item in raw.split(",") if item.strip()]
        if (
            isinstance(values, list)
            and values
            and all(isinstance(item, str) and item for item in values)
        ):
            return values[0]
        raise ValueError(f"{name} must be a non-empty list of model IDs")
    raise ValueError(f"Set {name}")


def maas_credentials():
    name, values = endpoint_credentials("maas")
    if not name:
        values.update(
            {
                "TLS_GENERATION_MODEL": model_id("GENERATION"),
                "TLS_EMBEDDING_MODEL": model_id("EMBEDDING"),
            }
        )
    # No trust overrides: the real DSPA test must consume platform trust unchanged.
    return name, values


def timeout_seconds(default):
    value = os.environ.get("AUTOX_TLS_TIMEOUT", "").strip()
    try:
        timeout = int(value) if value else default
        if timeout <= 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            "AUTOX_TLS_TIMEOUT must be a positive number of seconds"
        ) from None
    return timeout


def flow_secret_credentials(engine):
    if engine == "vector_db":
        engine = None
    if engine is None and os.environ.get("AUTOX_TLS_VECTOR_DB", "").strip():
        engine = os.environ["AUTOX_TLS_VECTOR_DB"].strip().lower()
        if engine not in {"milvus", "pgvector", "neo4j"}:
            raise ValueError("AUTOX_TLS_VECTOR_DB must be milvus, pgvector or neo4j")
    if engine is None:
        configured = []
        for backend in ("milvus", "pgvector", "neo4j"):
            try:
                endpoint_credentials(backend)
                configured.append(backend)
            except MissingTLSConfiguration:
                pass
        if len(configured) != 1:
            raise ValueError(
                "Set AUTOX_TLS_VECTOR_DB or configure exactly one database backend"
            )
        engine = configured[0]
    reference, values = endpoint_credentials(engine)
    if values and engine in {"milvus", "pgvector", "neo4j"}:
        for setting in ca_setting_names(engine):
            if os.environ.get(setting, "").strip():
                pem = read_ca(os.environ[setting], setting)
                if engine == "milvus":
                    # Pipeline versions may forward either key; both carry the same PEM.
                    values["MILVUS_SERVER_CERT"] = pem
                    values["MILVUS_CA_CERT"] = pem
                else:
                    values[engine.upper() + "_CA_CERT"] = pem
                break
    # Local AWS_CA_BUNDLE paths are host-only and must never enter product Secrets.
    return reference, values
