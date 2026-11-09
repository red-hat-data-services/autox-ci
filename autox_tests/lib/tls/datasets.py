"""Owned S3 uploads for runtime probes and the injected MaaS CA fault."""

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from autox_tests.lib.tls.settings import keep_resources


logger = logging.getLogger(__name__)
DATA_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetObject:
    bucket_argument: str
    key: str
    body: bytes
    content_type: str


@dataclass(frozen=True)
class DatasetPlan:
    values: dict
    objects: list[DatasetObject]


def build_dataset_plan(definition, prefix):
    """Build only the selected flow's data, without environment changes or S3 access."""
    if (
        definition.get("family") != "autorag"
        or definition.get("dataset") != "rag-light"
        or definition.get("fault") != "maas_wrong_ca"
    ):
        raise ValueError(
            "Automatic TLS datasets require a supported dataset profile for a MaaS CA fault"
        )
    prefix = prefix.rstrip("/") + "/rag-light"
    source = DATA_ROOT / "autorag/data/neo4j_light"
    objects, mapping = [], {}
    for path in sorted((source / "documents").glob("*.md")):
        relative = path.relative_to(source).as_posix()
        key = prefix + "/" + relative
        mapping["datasets/rag/neo4j_light/" + relative] = key
        objects.append(
            DatasetObject(
                "input_data_bucket_name", key, path.read_bytes(), "text/markdown"
            )
        )
    if not mapping:
        raise ValueError("The bundled TLS RAG corpus is empty")
    benchmark = json.loads((source / "benchmark_data.json").read_text())
    for row in benchmark:
        row["correct_answer_document_keys"] = [
            mapping[key] for key in row["correct_answer_document_keys"]
        ]
    benchmark_key = prefix + "/benchmark_data.json"
    objects.append(
        DatasetObject(
            "test_data_bucket_name",
            benchmark_key,
            json.dumps(benchmark).encode(),
            "application/json",
        )
    )
    return DatasetPlan(
        {
            "input_data_keys": list(mapping.values()),
            "test_data_key": benchmark_key,
        },
        objects,
    )


class OwnedDatasets:
    """Never create buckets, overwrite existing objects, or delete by prefix."""

    def __init__(self, clients):
        self.clients = clients
        self.owner = uuid.uuid4().hex
        self.prefix = "autox-tls-datasets/" + self.owner
        self.uploaded = {}

    def prepare(self, flow, plan):
        for item in plan.objects:
            bucket = flow["arguments"][item.bucket_argument]
            if not bucket:
                raise ValueError(f"Dataset upload needs {item.bucket_argument}")
            location = (item.bucket_argument, bucket, item.key)
            if location in self.uploaded:
                continue
            # Atomic create: even a pre-existing object at this random key is never replaced.
            response = self.clients[item.bucket_argument].put_object(
                Bucket=bucket,
                Key=item.key,
                Body=item.body,
                ContentType=item.content_type,
                IfNoneMatch="*",
                Metadata={"autox-tls-owner": self.owner},
            )
            self.uploaded[location] = {
                "ETag": response.get("ETag"),
                "VersionId": response.get("VersionId"),
            }

    def close(self):
        if keep_resources():
            return
        for (bucket_argument, bucket, key), identity in self.uploaded.items():
            client = self.clients[bucket_argument]
            target = {"Bucket": bucket, "Key": key}
            if identity["VersionId"] is not None:
                target["VersionId"] = identity["VersionId"]
            try:
                current = client.head_object(**target)
                if (
                    current.get("Metadata", {}).get("autox-tls-owner") != self.owner
                    or not identity["ETag"]
                    or current.get("ETag") != identity["ETag"]
                ):
                    logger.warning(
                        "TLS dataset changed; leaving s3://%s/%s untouched", bucket, key
                    )
                    continue
                client.delete_object(**target)
            except Exception:
                # Cleanup must not replace a pipeline/upload failure or expose SDK secrets.
                logger.warning(
                    "Could not clean test-owned TLS dataset s3://%s/%s", bucket, key
                )
