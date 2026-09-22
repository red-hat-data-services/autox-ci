"""Shared KFP and S3 client factories for functional test suites."""

import logging
import os

logger = logging.getLogger(__name__)


def make_kfp_client(config):
    """Create a KFP client from a config dict; returns None if config is None."""
    if config is None:
        logger.info("Skipping KFP client creation due to missing config.")
        return None
    import kfp

    host = config["rhoai_kfp_url"]
    if not host.endswith("/"):
        host = host + "/"
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    client = kfp.Client(
        host=host,
        namespace=config["rhoai_project"],
        existing_token=config.get("rhoai_token"),
        verify_ssl=verify_ssl,
    )
    _disable_gcp_token_refresh(client)
    return client


def _disable_gcp_token_refresh(client) -> None:
    """Turn an expired bearer token into a readable error instead of an opaque TypeError.

    On a 401 the KFP SDK calls ``_refresh_api_client_token()``, which only knows how to
    mint *GCP* credentials. Off GCP it returns ``None`` and the SDK assigns that to
    ``api_key['authorization']``; every later request then dies inside urllib3 with
    ``TypeError: expected string or bytes-like object, got 'NoneType'``. Once the token
    expires mid-session that hits every remaining test, and none of the messages mention
    authentication. Raising here surfaces the real cause on the first 401.
    """

    def _expired_bearer_token():
        raise RuntimeError(
            "KFP API returned 401 Unauthorized: the bearer token (RHOAI_TOKEN) is no longer "
            "valid — it most likely expired mid-run. Refresh it (`oc whoami -t`) in the env "
            "file and re-run. Long suites outlive short-lived OpenShift tokens."
        )

    client._refresh_api_client_token = _expired_bearer_token


def make_s3_client(config):
    """Create a boto3 S3 client from a config dict; returns None if not configured."""
    if config is None or not config.get("s3_endpoint"):
        logger.info("Skipping S3 client creation due to missing config.")
        return None
    try:
        import boto3
    except ImportError:
        logger.info("Skipping S3 client creation due to missing 'boto3' package.")
        return None
    verify_ssl = os.environ.get("S3_SSL_VERIFY", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return boto3.client(
        "s3",
        endpoint_url=config["s3_endpoint"],
        aws_access_key_id=config["s3_access_key"],
        aws_secret_access_key=config["s3_secret_key"],
        region_name=config["s3_region"],
        verify=verify_ssl,
    )
