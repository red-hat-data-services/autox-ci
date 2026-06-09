"""Unit tests for .env credential loading."""

from __future__ import annotations

import os

import pytest

from benchmark_common.credentials import credentials_dict_from_env, load_credentials_overlay


def test_credentials_dict_from_env_automl(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("BENCHMARK_", "RHOAI_", "KFP_", "AWS_", "AUTOML_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("BENCHMARK_KFP_HOST", "https://kfp.example.com")
    monkeypatch.setenv("RHOAI_PROJECT_NAME", "my-ns")
    monkeypatch.setenv("KFP_API_TOKEN", "tok")
    monkeypatch.setenv("AUTOML_TRAIN_DATA_BUCKET_NAME", "train-bucket")
    monkeypatch.setenv("RHOAI_TEST_S3_SECRET_NAME", "s3-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    overlay = credentials_dict_from_env()
    assert overlay["kfp"]["host"] == "https://kfp.example.com"
    assert overlay["kfp"]["namespace"] == "my-ns"
    assert overlay["kfp"]["token"] == "tok"
    assert overlay["storage"]["train_data_bucket_name"] == "train-bucket"
    assert overlay["pipeline"]["train_data_secret_name"] == "s3-secret"
    assert overlay["s3"]["aws_access_key_id"] == "key"


def test_load_credentials_overlay_from_fixture_env(automl_env_file, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("BENCHMARK_", "RHOAI_", "KFP_", "AWS_", "AUTOML_")):
            monkeypatch.delenv(key, raising=False)
    overlay, source = load_credentials_overlay(env_file=automl_env_file)
    assert str(automl_env_file) in source
    assert overlay["kfp"]["experiment_name"] == "automl-benchmark-test"
