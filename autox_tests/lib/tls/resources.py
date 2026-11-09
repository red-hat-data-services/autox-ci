"""Create and clean only test-owned credential Secrets, without editing existing ones."""

import uuid
import warnings

from kubernetes import client


class OwnedCredentialSecrets:
    def __init__(self, core, namespace):
        self.core = core
        self.namespace = namespace
        self.created = []
        self.cache = {}

    def reference(self, existing, values):
        if existing:
            return existing
        key = tuple(sorted(values.items()))
        if not key:
            raise ValueError("Cannot create an empty credential Secret")
        if key not in self.cache:
            name = "autox-tls-credentials-" + uuid.uuid4().hex[:12]
            self.core.create_namespaced_secret(
                self.namespace,
                client.V1Secret(
                    metadata=client.V1ObjectMeta(
                        name=name, labels={"app.kubernetes.io/part-of": "autox-tls"}
                    ),
                    string_data=values,
                ),
            )
            self.created.append(name)
            self.cache[key] = name
        return self.cache[key]

    def close(self):
        for name in reversed(self.created):
            try:
                self.core.delete_namespaced_secret(name, self.namespace)
            except client.ApiException as exc:
                if exc.status != 404:
                    warnings.warn(
                        f"Unable to clean test-owned Secret/{name}: HTTP {exc.status}",
                        stacklevel=2,
                    )
        self.created.clear()
        self.cache.clear()
