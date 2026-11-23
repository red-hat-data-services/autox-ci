"""Local unit tests for AutoRAG utility helpers."""

from pathlib import Path

import pytest

from autox_tests.autorag.configs.configs import (
    get_indexing_configs_for_run,
    get_test_configs_for_run,
)
from autox_tests.autorag.utils import get_uploaded_document_names, upload_test_datasets


pytestmark = pytest.mark.config


class _RecordingS3:
    """Minimal S3 stand-in that records uploads instead of calling the API."""

    def __init__(self) -> None:
        self.uploaded: list[tuple[str, str]] = []

    def upload_file(self, filename, bucket, key, ExtraArgs=None) -> None:
        self.uploaded.append((bucket, key))


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


def test_upload_test_datasets_skips_prefix_without_matching_local_dataset(tmp_path) -> None:
    """A key whose parent resembles no local dataset must stay absent in S3."""
    (tmp_path / "audio" / "documents").mkdir(parents=True)
    (tmp_path / "audio" / "documents" / "story.mp3").write_text("audio")
    (tmp_path / "rh_summit" / "documents").mkdir(parents=True)
    (tmp_path / "rh_summit" / "documents" / "notes.md").write_text("text")

    s3 = _RecordingS3()
    uploaded = upload_test_datasets(
        s3,
        "inputs",
        ["non/existing/path/to/documents", "datasets/rag/rh_summit_2026/documents"],
        tmp_path,
    )

    assert uploaded == ["datasets/rag/rh_summit_2026/documents/notes.md"]
    assert s3.uploaded == [("inputs", "datasets/rag/rh_summit_2026/documents/notes.md")]


def test_negative_indexing_scenario_prefix_is_not_uploaded() -> None:
    """IDX-F-2 expects documents-discovery to fail on a prefix that has no objects."""
    config = next(
        c for c in get_indexing_configs_for_run("negative") if c.id == "IDX-F-2"
    )
    s3 = _RecordingS3()

    uploaded = upload_test_datasets(
        s3,
        "inputs",
        config.input_data_keys or [],
        Path(__file__).parent / "data",
    )

    assert uploaded == []
    assert s3.uploaded == []
