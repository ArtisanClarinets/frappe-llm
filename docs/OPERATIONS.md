# Operations & Governance Checklist

This playbook captures the day-two responsibilities for the Frappe alignment stack. Keep it in sync
with production changes and audit the checklist during quarterly reviews.

## Logging & Retention

* Log request metadata: timestamp, hashed user identifier, prompt length, context length, latency,
  and final status (success/error). Avoid storing raw prompts or completions; hash strings with a
  salt rotated monthly.
* Retain structured logs for **30 days** in cold storage for incident response. Summaries/metrics can
  persist beyond that window.
* Mask secrets and user-provided access tokens prior to logging. Upstream gateways should enforce the
  same guarantees.

## Evaluation Discipline

* Maintain a **red-team regression set** that captures historical failure modes (unsafe SQL, naming
  conventions, migration ordering, etc.). Re-run this suite after every training or merge.
* Schedule a nightly `lm_eval` cron job (ARC-Easy + HellaSwag, zero-shot) on both the SFT baseline and
  the latest DPO merge. Alert when scores regress beyond agreed thresholds.
* Supplement harness metrics with lightweight "Frappe correctness" regex checks (DocType names,
  migration patches) to catch syntax regressions that automated metrics miss.

## Incident Response

* Keep the previous merged checkpoint available (both SFT and DPO). Roll back by switching the
  serving symlink to the last known-good artifact.
* If preference tuning regresses severely, disable the DPO adapter and revert to the merged SFT model
  until the root cause is understood.
* Document every incident with root-cause analysis, remediation steps, and mitigations in the internal
  runbook. Share summaries with stakeholders within 48 hours.

## Change Management

* Before deploying a new dataset or model, complete a peer review of data lineage, annotations, and
  bias assessments.
* Track dataset versions in source control (JSONL snapshots + manifest). Training jobs must reference
  immutable paths or explicit digests.
* When enabling new retrieval sources, ensure governance approval and document provenance in
  `docs/RAG.md`.
