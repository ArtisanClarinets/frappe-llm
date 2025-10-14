# Retrieval-Augmented Generation (RAG)

The RAG stack couples a FAISS dense retriever (BGE-M3 embeddings) with the `jina-reranker-v3`
cross-encoder. Use this document to manage sources, rebuild indices, and tune runtime behaviour.

## Rebuilding the Index

```bash
# Rebuild using default roots (../frappe and ./docs when present)
python rag/index_build.py --out /srv/frappe-llm/rag/index

# Provide explicit repositories
python rag/index_build.py --roots /srv/frappe,/srv/erpnext,/srv/frappe-llm/docs --out /srv/frappe-llm/rag/index
```

The builder stores `index.faiss`, `metadata.jsonl`, and `manifest.json`. Always commit the manifest so
changes in coverage are reviewable.

## Chunking Strategy

* Default chunking uses 120-line windows with 20-line overlap. Adjust `--chunk-size` and `--overlap`
  to match your documentation style.
* Highly structured modules (Python controllers, ERPNext hooks) may benefit from smaller windows
  (e.g., 80 lines) to avoid mixing unrelated functions.

## Reranker Tuning

* Retrieval pulls the top 50 candidates from FAISS, then reranks to the best 8 passages.
* Increase `initial_k` in `rag/retriever.py` if recall is insufficient. Larger values increase reranker
  latency but can improve quality for broad queries.
* Decrease `top_k` in serving code if VRAM pressure or prompt length becomes an issue.

## Context Packing

`rag.retriever.pack_context` formats passages as:

```
[path/to/file.py:12-44]
Code or documentation snippet
```

Concatenate the returned string with the user prompt to ensure citations remain traceable.

## Source Governance

* Only ingest repositories that have passed security review.
* Track provenance for each added root (commit SHA, fetch timestamp) in deployment change logs.
* Rebuild the index whenever a source repo updates materially; add a note to deployment tickets with
  the manifest diff.
