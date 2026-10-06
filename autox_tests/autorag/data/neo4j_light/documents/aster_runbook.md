# Aster incident runbook

When an Aster incident is reported, the operations team first identifies the affected
service, then checks that service's runbook. The Neo4j knowledge graph connects each
service to its runbook and to the incidents it addresses.

For the Billing API, the runbook recommends checking the payment-provider connection
before restarting any workloads. This prevents unnecessary restarts during external
payment-provider outages.
