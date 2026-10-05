# Telecom v3: ordered synthetic support procedures

This corpus contains 60 KB articles, 240 simulated-resolved training tickets and
30 historical responses. Its 330 indexed records exclude dev/test tickets. V1/V2
are retained. Train/dev/test complaint files are byte-identical to V2; no
classifier retraining was needed for this procedure-content change.

Each article has five ordered steps: two customer observations, an authorised
support investigation, a fully conditional remedy, and a completion check.
The diagnostic gate, restrictions and escalation criteria remain mandatory.
Known observations and completed attempts are skipped. Alternative procedures
contribute only a gated remedy and completion criterion, rather than a second
full checklist. The resulting number of steps can be lower when checks are known.

The source authoring file is
[`../telecom_v3_authoring/procedures.json`](../telecom_v3_authoring/procedures.json).
The publisher is `python -m scripts publish-v3`; it refuses to overwrite an
existing version. To make further changes, publish to a new corpus directory.
Generated `knowledge_base.jsonl`, `tickets.jsonl`, and `processed/documents.jsonl`
are validated ingestion artifacts. `quality_report.json` records hashes,
provenance, reference URLs and counts; it is not a plan-accuracy score.

Tickets retain `ticket_id`, KB references and `simulated_resolved` outcomes.
Their resolution steps were authored alongside the KB procedures and deliberately
synchronised. They are **not independently observed customer repairs** and do
not independently corroborate the article. Historical `[Tn]` citations are used
only when the corresponding retrieved ticket contains the exact step and links
the selected KB. KB references use `[Sn]`. Both open the source in the UI.
Some KB families, such as shared outages, have no matching training history;
their guidance cites the KB without inventing a ticket.

General wiring/outage guidance was consulted from BT and AT&T; reference URLs
are in the quality report. All procedures remain AI-authored, fictional-provider
material without telecom-expert review. Exact-source validation proves copied
spans and conditional rendering, not technical truth, diagnosis or repair success.
An optional LLM writes introductory wording; v3's ordered steps stay local and
the UI labels that distinction. No live account/network action is performed.

Activate using an existing configured PostgreSQL+pgvector database:

```powershell
$env:CORPUS_DIR = "data/synthetic/telecom_v3"
$env:LLM_ENABLED = "false"
python -m scripts chunk
python -m scripts index
python -m scripts check
python -m uvicorn app.main:app --host 127.0.0.1 --port 8011
```

Also set `CORPUS_DIR` in `.env` before restarting normal/container startup.
Updating KB content requires reindexing, not training the embedding model.
The initial ready index has 330 documents, chunks and normalised embeddings.
Older frozen evaluations predate V3 and are not V3 quality measurements.
