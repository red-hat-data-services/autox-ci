"""Local unit tests for AutoRAG utility helpers."""

import pytest

from autox_tests.autorag.configs.configs import get_test_configs_for_run
from autox_tests.autorag.utils import get_uploaded_document_names


pytestmark = pytest.mark.config


def test_get_uploaded_document_names_uses_empty_prefix_for_whole_bucket(monkeypatch) -> None:
    """An empty input_data_keys list makes the pipeline scan the complete bucket."""
    seen = {}

    def fake_list_s3_objects(s3_client, bucket, prefix):
        seen.update(s3_client=s3_client, bucket=bucket, prefix=prefix)
        return [
            {"Key": "documents/a.md"},
            {"Key": "documents/b.txt"},
            {"Key": "documents/"},
        ]

    monkeypatch.setattr(
        "autox_tests.lib.s3_data.list_s3_objects", fake_list_s3_objects
    )

    assert get_uploaded_document_names("client", "input-bucket", []) == {"a.md", "b.txt"}
    assert seen == {"s3_client": "client", "bucket": "input-bucket", "prefix": ""}


def test_get_uploaded_document_names_unions_all_input_prefixes(monkeypatch) -> None:
    """Multiple configured prefixes must match ai4rag's union-based discovery."""
    seen_prefixes = []

    def fake_list_s3_objects(_s3_client, _bucket, prefix):
        seen_prefixes.append(prefix)
        return [{"Key": f"{prefix}document.md"}]

    monkeypatch.setattr(
        "autox_tests.lib.s3_data.list_s3_objects", fake_list_s3_objects
    )

    assert get_uploaded_document_names("client", "input-bucket", ["set-a", "set-b/"]) == {
        "document.md"
    }
    assert seen_prefixes == ["set-a/", "set-b/"]


def test_too_many_input_prefixes_scenario_has_eleven_entries() -> None:
    """The negative scenario must exceed ai4rag's supported ten-prefix limit."""
    config = next(c for c in get_test_configs_for_run("negative") if c.id == "TC-F-4")

    assert len(config.input_data_keys or []) == 11
