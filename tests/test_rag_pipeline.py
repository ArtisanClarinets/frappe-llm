import numpy as np

from rag import index_build
from rag.retriever import RAGRetriever, pack_context, retrieve


class DummyEmbedder:
    def encode(self, texts, batch_size=1):
        vectors = []
        for text in texts:
            text = text.lower()
            tokens = text.split()
            length = len(tokens)
            alpha = text.count("alpha")
            beta = text.count("beta")
            gamma = text.count("gamma")
            vectors.append(np.array([length, alpha + beta, gamma], dtype=np.float32))
        return {"dense_vecs": np.vstack(vectors)}


class DummyReranker:
    def rerank(self, query, candidates, top_k=8):
        scored = []
        for candidate in candidates:
            text = candidate["text"].lower()
            score = text.count("gamma") * 10 + text.count("beta")
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score)
            scored.append(enriched)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]


def test_rag_pipeline_end_to_end(tmp_path):
    docs = {
        "doc_alpha.txt": "Alpha content with limited signals.\n",
        "doc_beta.txt": "Beta reference mentions beta and gamma once.\n",
        "doc_gamma.txt": "Gamma heavy gamma gamma beta text for highest match.\n",
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    out_dir = tmp_path / "index"
    embedder = DummyEmbedder()
    index_build.build_index([tmp_path], out_dir, chunk_size=200, overlap=0, embedder=embedder)

    retriever = RAGRetriever(out_dir, embedder=embedder, reranker=DummyReranker(), initial_k=3)
    results = retriever.retrieve("gamma feature", top_k=2)
    assert len(results) == 2
    assert results[0]["relative_path"].endswith("doc_gamma.txt")

    context = pack_context(results, max_tokens=50)
    assert "doc_gamma.txt" in context
    assert "Gamma heavy" in context

    manual = retrieve("beta", k=1, retriever=retriever)
    assert manual[0]["relative_path"].endswith("doc_gamma.txt") or manual[0]["relative_path"].endswith("doc_beta.txt")
