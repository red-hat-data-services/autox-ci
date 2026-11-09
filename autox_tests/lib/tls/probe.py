"""Standalone probe copied verbatim into product containers (stdlib at import time)."""

import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import ssl
import tempfile
import threading
import uuid
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

TRUST_KEYS = (
    "MAAS_CA_BUNDLE",
    "AWS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


class ScenarioNotApplicable(Exception):
    """The endpoint does not meet a scenario's CA-isolation prerequisite."""


def require_private_s3_ca(report):
    """Verify default trust independently, without invoking the product client."""
    from botocore.httpsession import get_cert_path

    report["stage"] = "s3_default_trust_preflight"
    url = urlsplit(os.environ["AWS_S3_ENDPOINT"])
    if url.scheme != "https" or not url.hostname:
        raise ValueError("S3 CA preflight requires an HTTPS endpoint")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    # Use the SDK's default CA file, never the scenario's injected trust/env.
    context.load_verify_locations(cafile=get_cert_path(True))
    try:
        with socket.create_connection(
            (url.hostname, url.port or 443), timeout=10
        ) as connection:
            with context.wrap_socket(connection, server_hostname=url.hostname):
                pass
    except ssl.SSLCertVerificationError as exc:
        report["s3_default_trust"] = {"trusted": False, "verify_code": exc.verify_code}
        if exc.verify_code not in {18, 19, 20, 21}:
            raise ValueError(
                "S3 fixture failed default-trust preflight for a reason other than an untrusted issuer"
            ) from None
        return
    report["s3_default_trust"] = {"trusted": True}
    raise ScenarioNotApplicable(
        "S3 endpoint is already trusted by the image default CA store; this scenario requires a private/untrusted issuer"
    )


def validate_dspa_trust(expected_dir):
    """SSL_CERT_DIR is a path list, not necessarily a single directory."""
    directories = [
        str(Path(value))
        for value in os.environ.get("SSL_CERT_DIR", "").split(os.pathsep)
        if value
    ]
    if str(Path(expected_dir)) not in directories:
        raise AssertionError(
            "DSPA did not supply the expected CA directory in SSL_CERT_DIR"
        )


def error_kind(exc):
    """Only certificate-specific errors count as expected rejections."""
    seen, messages = set(), []
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        if isinstance(current, ssl.SSLCertVerificationError):
            return "tls_error"
        current = current.__cause__ or current.__context__
    message = " ".join(messages)
    if "hostname" in message and any(
        part in message for part in ("doesn't match", "does not match")
    ):
        return "tls_error"
    if any(
        part in message
        for part in (
            "certificate_verify_failed",
            "certificate verify failed",
            "certificate verification failed",
            "hostname mismatch",
            "certificate has expired",
            "self-signed certificate",
            "unable to get local issuer certificate",
            "certificate is not trusted",
            "certificate signed by unknown authority",
        )
    ):
        return "tls_error"
    if any(
        part in message
        for part in (
            "no certificate or crl found",
            "no start line",
            "pem lib",
            "invalid certificate",
            "certificate file",
            "not a readable pem",
        )
    ):
        return "ca_error"
    return "unexpected_error"


def error_diagnostics(exc):
    """Emit only numeric codes and allowlisted signals, never SDK exception text."""
    chain, seen, messages = [], set(), []
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    message = " ".join(messages)
    signals = {
        "connection_unavailable": (
            "fail connecting to server",
            "failed to connect",
            "server unavailable",
            "illegal connection params",
            "failed to create new connection",
            "connection refused",
            "failed to connect to all addresses",
        ),
        "authentication": (
            "unauthenticated",
            "authentication failed",
            "invalid token",
            "unauthorized",
        ),
        "permission_denied": (
            "permission denied",
            "permissiondenied",
            "privilege",
            "not authorized",
        ),
        "timeout": ("deadline_exceeded", "deadline exceeded", "timed out", "timeout"),
        "grpc_http2": (
            "http/2",
            "http2",
            "http1",
            "http status code",
            "unexpected content-type",
        ),
        "tls_handshake": ("ssl handshake", "tls handshake", "handshake failed"),
        "unsupported_feature": (
            "not supported",
            "unsupported",
            "unimplemented",
            "unknown function",
        ),
        "dimension_mismatch": (
            "dimension mismatch",
            "dimension not match",
            "dimension is not equal",
        ),
        "collection_missing": (
            "collection not found",
            "collection does not exist",
            "can't find collection",
        ),
    }
    categories = [
        name
        for name, fragments in signals.items()
        if any(part in message for part in fragments)
    ]
    kind = error_kind(exc)
    if kind != "unexpected_error":
        categories.insert(0, kind)
    details = {"categories": categories or ["unclassified"], "exception_chain": chain}
    code = getattr(exc, "code", None)
    if type(code) is int:
        details["code"] = code
    return details


@contextmanager
def capture_native_ca_errors(report, *, certificate=False):
    """Capture C-level stderr locally; retain only specific CA parser signals."""
    markers = {
        "no_root_certificates_loaded": "could not load any root certificate",
        "root_certificates_load_failed": "failed to load root certificates",
        "root_certificates_not_loaded": "could not load root certificates",
    }
    with tempfile.TemporaryFile() as output:
        saved = os.dup(2)
        try:
            os.dup2(output.fileno(), 2)
            yield
        finally:
            try:
                os.dup2(saved, 2)
            finally:
                os.close(saved)
            output.seek(0)
            # Chunk reads avoid retaining potentially large SDK output in memory.
            signals, certificate_signals, tail = set(), set(), b""
            while chunk := output.read(8192):
                text = (tail + chunk).decode("utf-8", errors="replace").lower()
                signals.update(key for key, value in markers.items() if value in text)
                if certificate:
                    if (
                        "peer name" in text and "not in peer certificate" in text
                    ) or "hostname verification check failed" in text:
                        certificate_signals.add("hostname_mismatch")
                    if "certificate has expired" in text:
                        certificate_signals.add("certificate_expired")
                tail = (tail + chunk)[-128:]
            if certificate:
                report["native_certificate_signals"] = sorted(certificate_signals)
            else:
                report["native_ca_signals"] = sorted(signals)


def native_milvus_ca_rejection(case, report):
    """Generic gRPC unavailability is insufficient evidence of CA rejection."""
    configuration = report.get("milvus_trust_configuration", {})
    return (
        case["engine"] == "milvus"
        and case["trust"] == "invalid"
        and report.get("stage") == "vector_store_initialization"
        and configuration.get("explicit_ca_consumed") is True
        and configuration.get("ca_pem_parseable") is False
        and bool(report.get("native_ca_signals"))
    )


def configure_milvus_trust(get_config, trust, bundle, wrong, report):
    """Use the installed library's CA contract and verify it consumed the PEM."""
    names = ("MILVUS_SERVER_CERT", "MILVUS_CA_CERT")
    for name in names:
        os.environ.pop(name, None)
    report["stage"] = "milvus_trust_configuration"
    config = get_config()
    fields = [name for name in ("server_cert", "ca_cert") if hasattr(config, name)]
    diagnostic = {"available_fields": fields, "explicit_ca_consumed": False}
    report["milvus_trust_configuration"] = diagnostic
    if trust not in ("explicit", "wrong", "invalid"):
        return config
    if not fields:
        raise RuntimeError(
            "Milvus configuration exposes no supported certificate field"
        )
    pem = Path(wrong if trust == "wrong" else bundle).read_text()
    try:
        ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=pem)
        diagnostic["ca_pem_parseable"] = True
    except (ssl.SSLError, ValueError):
        diagnostic["ca_pem_parseable"] = False
    # Diagnostic only: exercise the product even when the fixture is malformed.
    declared = {
        item[0]
        for item in getattr(config, "env_vars", ())
        if isinstance(item, (tuple, list)) and item
    }
    candidates = [name for name in names if name in declared] or list(names)
    for name in candidates:
        for other in names:
            os.environ.pop(other, None)
        os.environ[name] = pem
        config = get_config()
        matching = [field for field in fields if getattr(config, field) == pem]
        if matching:
            diagnostic.update(
                {
                    "environment_variable": name,
                    "config_field": matching[0],
                    "explicit_ca_consumed": True,
                }
            )
            return config
    raise RuntimeError("Milvus configuration did not consume the explicit CA PEM")


def cleanup_vector_store(store, report, operation_failed):
    """Always close; retain the operation failure when cleanup also fails."""
    previous_stage = report.get("stage")
    failures = []
    for stage, action in (
        ("cleanup_collection", store.clean_collection),
        ("close_store", store.close),
    ):
        try:
            action()
        except Exception as exc:
            failures.append((stage, exc))
            report.setdefault("cleanup_errors", []).append(
                {"stage": stage, **error_diagnostics(exc)}
            )
    if failures and not operation_failed:
        report["stage"] = failures[0][0]
        raise failures[0][1]
    if previous_stage is not None:
        report["stage"] = previous_stage


def apply_trust(case, directory, ca, other):
    """Apply exactly one declared trust path; never fix a client's configuration."""
    for name in TRUST_KEYS:
        os.environ.pop(name, None)
    trust = case["trust"]
    bundle = directory / "ca.crt"
    bundle.write_text(ca)
    wrong = directory / "other.pem"
    wrong.write_text(other)
    if trust in (
        "mounted",
        "conflicting_env",
        "conflicting_file",
        "multiple_dirs",
        "hashed_and_plain",
    ):
        os.environ["SSL_CERT_DIR"] = str(directory)
    if trust == "ssl_file":
        os.environ["SSL_CERT_FILE"] = str(bundle)
    elif trust == "default_store":
        # Controlled default-store analogue. Public roots are additionally
        # checked below; this is not advertised as a public-Internet test.
        import certifi

        controlled_store = directory / "default-store.pem"
        controlled_store.write_text(Path(certifi.where()).read_text() + "\n" + ca)
        certifi.where = lambda: str(controlled_store)
    elif trust == "maas_env":
        os.environ["MAAS_CA_BUNDLE"] = str(bundle)
    elif trust == "aws_env":
        os.environ["AWS_CA_BUNDLE"] = str(bundle)
    elif trust == "conflicting_env":
        os.environ["MAAS_CA_BUNDLE"] = str(wrong)
    elif trust == "conflicting_file":
        os.environ["SSL_CERT_FILE"] = str(wrong)
    elif trust in ("hashed", "hashed_and_plain"):
        pass
    elif trust == "argument_over_env":
        os.environ["MAAS_CA_BUNDLE"] = str(wrong)
    elif trust == "explicit_over_env":
        os.environ["AWS_CA_BUNDLE"] = str(wrong)
    elif trust == "invalid":
        bundle.write_text("invalid PEM data")
        os.environ["MAAS_CA_BUNDLE"] = str(bundle)
        os.environ["AWS_CA_BUNDLE"] = str(bundle)
    elif trust == "missing_file":
        os.environ["MAAS_CA_BUNDLE"] = str(directory / "absent.pem")
    if trust == "multiple_dirs":
        os.environ["SSL_CERT_DIR"] = (
            str(directory / "absent") + os.pathsep + str(directory)
        )
    if trust in ("hashed", "hashed_and_plain"):
        hashed = directory / "hashed"
        hashed.mkdir()
        material = Path(os.environ.get("TLS_MATERIAL_DIR", "/opt/autox-tls/material"))
        subject_hash = (material / "ca.hash").read_text().strip()
        shutil.copyfile(bundle, hashed / (subject_hash + ".0"))
        os.environ["SSL_CERT_DIR"] = str(hashed)
        if trust == "hashed_and_plain":
            import certifi

            bundle.write_text(Path(certifi.where()).read_text())
            os.environ["SSL_CERT_DIR"] += os.pathsep + str(directory)
    return str(bundle), str(wrong)


def start_maas(material, certificate, redirect):
    """Real TLS and OpenAI protocol, with deterministic responses and request counts."""
    counts = {"models": 0, "redirect": 0, "chat": 0, "embeddings": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, body):
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if redirect and self.path == "/v1/models":
                counts["redirect"] += 1
                self.send_response(307)
                self.send_header("Location", "/v1/models/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            counts["models"] += 1
            self.reply({"object": "list", "data": []})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path.endswith("/embeddings"):
                counts["embeddings"] += 1
                self.reply(
                    {
                        "object": "list",
                        "model": "tls-model",
                        "data": [
                            {
                                "object": "embedding",
                                "index": 0,
                                "embedding": [0.1, 0.2, 0.3],
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                )
            else:
                counts["chat"] += 1
                self.reply(
                    {
                        "id": "tls-chat",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "tls-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "tls-ok"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(material / (certificate + ".pem"), material / "server.key")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, counts


def start_s3(material, certificate):
    """Loopback-only HTTPS fixture; no real credentials or external S3 access."""
    counts = {"http": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            counts["http"] += 1
            body = b"x"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            material / (certificate + ".pem"), material / "server.key"
        )
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    except BaseException:
        server.server_close()
        raise
    return server, thread, counts


def start_milvus_tls(material, certificate):
    """Real loopback gRPC TLS listener; not a functional Milvus service."""
    import grpc
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=2)
    server = grpc.server(executor)
    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "autox.tls.Probe",
                {
                    "Ping": grpc.unary_unary_rpc_method_handler(
                        lambda request, context: b"ok"
                    )
                },
            ),
        )
    )
    try:
        credentials = grpc.ssl_server_credentials(
            (
                (
                    (material / "server.key").read_bytes(),
                    (material / (certificate + ".pem")).read_bytes(),
                ),
            )
        )
        port = server.add_secure_port("127.0.0.1:0", credentials)
        if not port:
            raise RuntimeError("Failed to bind local gRPC TLS fixture")
        server.start()
    except BaseException:
        server.stop(0).wait(timeout=5)
        executor.shutdown(wait=True)
        raise
    return server, executor, port


def milvus_certificate_reason(exc, report):
    reasons = set(report.get("native_certificate_signals", ()))
    reason = s3_certificate_reason(exc)
    if reason:
        reasons.add(reason)
    return next(iter(reasons)) if len(reasons) == 1 else None


def s3_certificate_reason(exc):
    """Extract only hostname/expiry signals; never retain raw SDK error text."""
    pending, seen, reasons = [exc], set(), set()
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        code = getattr(current, "verify_code", None)
        text = str(current).lower()
        if (
            ("peer name" in text and "not in peer certificate" in text)
            or "hostname verification check failed" in text
            or code in {62, 64}
            or any(
                part in text
                for part in (
                    "hostname mismatch",
                    "ip address mismatch",
                    "doesn't match",
                    "does not match",
                    "not valid for",
                )
            )
        ):
            reasons.add("hostname_mismatch")
        if code == 10 or "certificate has expired" in text:
            reasons.add("certificate_expired")
        pending.extend(
            (current.__cause__, current.__context__, getattr(current, "reason", None))
        )
        pending.extend(current.args)
        pending.extend(getattr(current, "kwargs", {}).values())
    return next(iter(reasons)) if len(reasons) == 1 else None


def maas_request(case, url, bundle):
    from ai4rag.utils.clients.maas_client import create_maas_client
    from openai import OpenAI

    kwargs = {}
    if case["trust"] in ("argument", "argument_over_env"):
        kwargs["ca_bundle"] = bundle
    api_key = os.environ.get("MAAS_API_KEY", "tls-test-key")
    if case["engine"] == "direct":
        base = url.rstrip("/")
        client = OpenAI(
            base_url=base if base.endswith("/v1") else base + "/v1",
            api_key=api_key,
            max_retries=0,
            timeout=15,
        )
    else:
        client = create_maas_client(url, api_key, **kwargs)
    with client:
        assert_verified_transport(client._client)
        context = vars(client).get("_ai4rag_tls_verify")
        if case["trust"] in ("mounted", "multiple_dirs") and isinstance(
            context, ssl.SSLContext
        ):
            import certifi

            public = set(
                ssl.create_default_context(cafile=certifi.where()).get_ca_certs(
                    binary_form=True
                )
            )
            assert public <= set(context.get_ca_certs(binary_form=True)), (
                "Mounted CA replaced public roots"
            )
        client.models.list()
        model = case.get("generation_model", "tls-model")
        embedding = case.get("embedding_model", "tls-model")
        client.embeddings.create(model=embedding, input="tls probe")
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "TLS probe"}]
        )
        if case["engine"] == "agent":
            from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
            from ai4rag.rag.template.agentic_rag_template import AgenticRAG

            agent = AgenticRAG(
                foundation_model=OpenAIFoundationModel(client=client, model_id=model),
                retriever=None,
            )
            chat = agent._chat_model()
            try:
                assert_verified_transport(chat.root_client._client)
                assert_verified_transport(chat.root_async_client._client)
                assert chat.invoke("TLS probe").content

                async def invoke():
                    try:
                        return await chat.ainvoke("TLS probe")
                    finally:
                        await chat.root_async_client.close()

                assert asyncio.run(invoke()).content
            finally:
                if chat.root_client is not client:
                    chat.root_client.close()


def assert_verified_transport(http_client):
    """A successful request after verify=False fallback must never pass a CA test."""
    context = http_client._transport._pool._ssl_context
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname, (
        "Client disabled certificate or hostname verification"
    )


def service_request(case, bundle, wrong, report=None):
    """Exercise product factories with real S3/DB endpoints supplied by a Secret."""
    report = {} if report is None else report
    report["stage"] = "service_configuration"
    engine, trust = case["engine"], case["trust"]
    if engine == "s3":
        from ai4rag.utils.clients.s3 import create_s3_client

        assert os.environ["AWS_S3_ENDPOINT"].startswith("https://"), (
            "S3 fixture must enforce TLS"
        )

        if trust in ("mounted", "missing"):
            require_private_s3_ca(report)
        verify = wrong if trust == "wrong" else bundle
        kwargs = (
            {"verify": verify}
            if trust in ("explicit", "explicit_over_env", "wrong")
            else {}
        )
        report["stage"] = "s3_client_initialization"
        bucket, key = os.environ["INPUT_DATA_BUCKET_NAME"], case["s3_key"]
        client = create_s3_client(**kwargs)
        try:
            report["stage"] = "s3_get_object"
            response = client.get_object(Bucket=bucket, Key=key)
            try:
                report["stage"] = "s3_read_object"
                assert response["Body"].read(1), (
                    "Automatically prepared S3 probe object must be non-empty"
                )
            finally:
                response["Body"].close()
        finally:
            client.close()
        return
    from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
    from ai4rag.rag.vector_store import get_vector_store, get_vector_store_config

    class Embedding(BaseEmbeddingModel):
        def __init__(self):
            super().__init__(
                client=None, model_id="tls-probe", params={"embedding_dimension": 4}
            )

        def embed_query(self, query):
            return [0.1, 0.2, 0.3, 0.4]

        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

    if engine == "milvus":
        config = configure_milvus_trust(
            lambda: get_vector_store_config(engine), trust, bundle, wrong, report
        )
    else:
        variable = engine.upper() + "_CA_CERT"
        os.environ.pop(variable, None)
        if trust in ("explicit", "wrong", "invalid"):
            os.environ[variable] = Path(
                wrong if trust == "wrong" else bundle
            ).read_text()
        config = get_vector_store_config(engine)
    if engine == "milvus":
        assert config.uri.startswith("https://"), (
            "Milvus fixture must use HTTPS/gRPC TLS"
        )
    if engine == "neo4j" and trust == "missing":
        assert config.uri.startswith(("bolt+s://", "neo4j+s://")), (
            "Neo4j without explicit CA must use a verified +s URI"
        )
    if engine == "pgvector" and trust == "missing":
        from ai4rag.utils.network import is_local_or_cluster_host

        assert not is_local_or_cluster_host(config.host), (
            "PGVector missing CA cases require a non-local, non-cluster host to enforce verification"
        )
    # A fresh name prevents a probe from touching any application collection.
    report["stage"] = "vector_store_initialization"
    collection_name = "ai4rag_tls_" + uuid.uuid4().hex
    report["test_collection"] = collection_name
    capture = (
        capture_native_ca_errors(report)
        if engine == "milvus" and trust == "invalid"
        else nullcontext()
    )
    with capture:
        store = get_vector_store(
            embedding_model=Embedding(),
            config=config,
            collection_name=collection_name,
        )
    operation_failed = False
    try:
        report["stage"] = "document_preparation"
        from ai4rag.rag.chunking.chunk import AI4RAGChunk

        chunk = AI4RAGChunk(
            text="TLS connectivity probe",
            metadata={"document_id": "autox-tls", "sequence_number": 0},
        )
        report["stage"] = "add_documents"
        store.add_documents([chunk])
    except Exception:
        operation_failed = True
        raise
    finally:
        cleanup_vector_store(store, report, operation_failed)


def execute(case):
    """Return a structured report; configuration/unrelated failures never pass as TLS errors."""
    versions = {}
    for package in (
        "ai4rag",
        "openai",
        "httpx",
        "boto3",
        "pymilvus",
        "asyncpg",
        "neo4j",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "id": case["id"],
        "engine": case["engine"],
        "trust": case["trust"],
        "versions": versions,
        "stage": "trust_configuration",
    }
    server = thread = grpc_server = grpc_executor = None
    try:
        with tempfile.TemporaryDirectory(prefix="autox-tls-") as temporary:
            directory = Path(temporary)
            if case.get("dspa"):
                report["stage"] = "dspa_configuration"
                expected_dir = case.get("mount_dir", "/kfp/certs")
                mode = case.get("dspa_mode", "platform")
                report["dspa_configuration"] = {
                    "mode": mode,
                    "removed_overrides": [],
                    "expected_dir": expected_dir,
                    "ssl_cert_dir": os.environ.get("SSL_CERT_DIR", ""),
                    "explicit_overrides": [
                        name
                        for name in ("MAAS_CA_BUNDLE", "SSL_CERT_FILE")
                        if os.environ.get(name)
                    ],
                }
                validate_dspa_trust(expected_dir)
                if mode not in ("platform", "plain-dir"):
                    raise ValueError("Unknown DSPA trust mode")
                if mode == "plain-dir":
                    # This task is isolated: never modify the image or cluster settings.
                    for name in ("MAAS_CA_BUNDLE", "SSL_CERT_FILE"):
                        if os.environ.pop(name, None):
                            report["dspa_configuration"]["removed_overrides"].append(
                                name
                            )
                bundle = str(Path(expected_dir) / "ca.crt")
                ca = Path(bundle).read_text()
                wrong = ""
            else:
                material = Path(
                    os.environ.get("TLS_MATERIAL_DIR", "/opt/autox-tls/material")
                )
                ca = Path(
                    os.environ.get("TLS_SERVICE_CA_FILE", str(material / "ca.pem"))
                ).read_text()
                other = (material / "other.pem").read_text()
                bundle, wrong = apply_trust(case, directory, ca, other)
            report["ca_sha256"] = hashlib.sha256(ca.encode()).hexdigest()
            report["trust_environment"] = {
                name: os.environ[name] for name in TRUST_KEYS if name in os.environ
            }
            if case["engine"] in ("maas", "agent", "direct"):
                report["stage"] = "maas_requests"
                if case.get("live") or case.get("dspa"):
                    url = os.environ["MAAS_BASE_URL"]
                else:
                    # Avoid image-level proxy configuration altering loopback probes.
                    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
                    server, thread, counts = start_maas(
                        material,
                        case.get("certificate", "server"),
                        case.get("redirect", False),
                    )
                    url = f"https://127.0.0.1:{server.server_port}"
                maas_request(case, url, bundle)
                if server:
                    report["requests"] = dict(counts)
                    assert counts["chat"] >= (3 if case["engine"] == "agent" else 1)
                    assert counts["embeddings"] >= 1
                    if case.get("redirect"):
                        assert counts["redirect"] >= 1
            else:
                if case.get("local_s3_fixture"):
                    if case["engine"] != "s3" or case.get("certificate") not in {
                        "wrong-host",
                        "expired",
                    }:
                        raise ValueError("Invalid local S3 certificate fixture")
                    report["stage"] = "s3_fixture_setup"
                    server, thread, counts = start_s3(material, case["certificate"])
                    os.environ.update(
                        {
                            "AWS_S3_ENDPOINT": f"https://127.0.0.1:{server.server_port}",
                            "AWS_ACCESS_KEY_ID": "tls-fixture-key",
                            "AWS_SECRET_ACCESS_KEY": "tls-fixture-secret",
                            "AWS_DEFAULT_REGION": "us-east-1",
                            "AWS_SESSION_TOKEN": "",
                            "INPUT_DATA_BUCKET_NAME": "tls-fixture",
                            "NO_PROXY": "127.0.0.1,localhost",
                            "no_proxy": "127.0.0.1,localhost",
                        }
                    )
                    case = {**case, "s3_key": "probe.txt"}
                    report["endpoint_fixture"] = "generated-loopback"
                if case.get("local_milvus_fixture"):
                    if case["engine"] != "milvus" or case.get("certificate") not in {
                        "wrong-host",
                        "expired",
                    }:
                        raise ValueError("Invalid local Milvus certificate fixture")
                    report["stage"] = "milvus_fixture_setup"
                    grpc_server, grpc_executor, port = start_milvus_tls(
                        material, case["certificate"]
                    )
                    os.environ.update(
                        {
                            "MILVUS_URI": f"https://127.0.0.1:{port}",
                            "MILVUS_TOKEN": "tls-fixture-token",
                            "NO_PROXY": "127.0.0.1,localhost",
                            "no_proxy": "127.0.0.1,localhost",
                        }
                    )
                    report["endpoint_fixture"] = "generated-loopback-grpc"
                capture = (
                    capture_native_ca_errors(report, certificate=True)
                    if case["engine"] == "milvus"
                    and case.get("certificate") in {"wrong-host", "expired"}
                    else nullcontext()
                )
                with capture:
                    service_request(case, bundle, wrong, report)
            report["outcome"] = "success"
    except ScenarioNotApplicable as exc:
        report["outcome"] = "not_applicable"
        report["skip_reason"] = str(exc)
    except Exception as exc:
        report["outcome"] = error_kind(exc)
        if report.get("stage") == "s3_default_trust_preflight":
            # Network/configuration/preflight failures cannot satisfy a rejection test.
            report["outcome"] = "unexpected_error"
        if report["outcome"] == "unexpected_error" and native_milvus_ca_rejection(
            case, report
        ):
            report["outcome"] = "ca_error"
            report["rejection_evidence"] = "native_ca_parse_failure"
        if case["trust"] == "missing_file" and isinstance(exc, FileNotFoundError):
            report["outcome"] = "ca_error"
        # Do not print exception text: SDK errors can include endpoint credentials.
        report["exception_type"] = type(exc).__name__
        report["error_diagnostics"] = error_diagnostics(exc)
        if case["engine"] == "s3" and case.get("certificate") in {
            "wrong-host",
            "expired",
        }:
            report["certificate_rejection_reason"] = s3_certificate_reason(exc)
        if case["engine"] == "milvus" and case.get("certificate") in {
            "wrong-host",
            "expired",
        }:
            reason = milvus_certificate_reason(exc, report)
            report["certificate_rejection_reason"] = reason
            config = report.get("milvus_trust_configuration", {})
            if (
                reason
                and config.get("explicit_ca_consumed") is True
                and config.get("ca_pem_parseable") is True
                and report.get("stage")
                in {"vector_store_initialization", "add_documents"}
            ):
                report["outcome"] = "tls_error"
    finally:
        if grpc_server:
            grpc_server.stop(0).wait(timeout=5)
            grpc_executor.shutdown(wait=True)
        if server:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    report["passed"] = report["outcome"] == case["expected"]
    if case["engine"] in {"s3", "milvus"} and case.get("certificate") in {
        "wrong-host",
        "expired",
    }:
        expected_reason = {
            "wrong-host": "hostname_mismatch",
            "expired": "certificate_expired",
        }[case["certificate"]]
        report["passed"] = (
            report["passed"]
            and report.get("certificate_rejection_reason") == expected_reason
        )
    if case["engine"] == "milvus" and case.get("certificate") in {
        "wrong-host",
        "expired",
    }:
        config = report.get("milvus_trust_configuration", {})
        report["passed"] = (
            report["passed"]
            and config.get("explicit_ca_consumed") is True
            and config.get("ca_pem_parseable") is True
            and report.get("stage") in {"vector_store_initialization", "add_documents"}
        )
    return report


if __name__ == "__main__":
    result = execute(json.loads(os.environ["TLS_SCENARIO"]))
    print("AUTOX_TLS_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(
        0 if result["passed"] or result["outcome"] == "not_applicable" else 1
    )
