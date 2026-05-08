"""Load benchmark YAML fragment from an in-cluster / kubeconfig-backed ConfigMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmark_common.merge import deep_merge
from benchmark_common.yaml_io import load_yaml_dict

logger = logging.getLogger(__name__)


def read_configmap_yaml(name: str, namespace: str, key: str) -> dict[str, Any]:
    from kubernetes import client, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()

    api = client.CoreV1Api()
    cm = api.read_namespaced_config_map(name=name, namespace=namespace)
    raw = (cm.data or {}).get(key)
    if not raw:
        raise KeyError(f"ConfigMap {namespace}/{name} has no data key {key!r}")
    import yaml

    data = yaml.safe_load(raw)
    return data if isinstance(data, dict) else {}


def apply_kubernetes_configmap_overlay(cfg: dict[str, Any]) -> dict[str, Any]:
    kc = cfg.get("kubernetes_configmap")
    if not kc or not isinstance(kc, dict):
        return cfg
    name, ns = kc.get("name"), kc.get("namespace")
    if not name or not ns:
        return cfg
    key = str(kc.get("key", "benchmark.yaml"))
    overlay = read_configmap_yaml(str(name), str(ns), key)
    merged = deep_merge(cfg, overlay)
    logger.info("Merged Kubernetes ConfigMap %s/%s key=%s", ns, name, key)
    return merged


def load_benchmark_config_file(config_path: Path | str) -> tuple[dict[str, Any], Path]:
    """Load YAML from disk, then optionally merge ConfigMap overlay."""
    path = Path(config_path).resolve()
    cfg = load_yaml_dict(path)
    cfg = apply_kubernetes_configmap_overlay(cfg)
    return cfg, path.parent.resolve()
