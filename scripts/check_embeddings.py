"""Real-model Phase C smoke test; no database writes or corpus indexing."""
import json
import numpy as np

from app.retrieval.embeddings import get_embedding_service


def main() -> None:
    service = get_embedding_service()
    texts = ["My internet connection drops every evening.",
             "My broadband disconnects each night.",
             "Please explain the duplicate charge on my invoice."]
    vectors = np.asarray(service.embed_documents(texts))
    query = np.asarray(service.embed_query(texts[0]))
    if vectors.shape != (3, 384) or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise RuntimeError("Embedding shape or normalization check failed")
    if not np.allclose(query, vectors[0], atol=1e-5):
        raise RuntimeError("Query and document embeddings differ unexpectedly")
    print(json.dumps({"status": "ok", "device": "cpu", "shape": list(vectors.shape),
                      "query_dimension": len(query), "unit_norms": True,
                      "query_document_consistent": True,
                      "model_reused": service is get_embedding_service(),
                      "paraphrase_cosine": float(vectors[0] @ vectors[1]),
                      "unrelated_cosine": float(vectors[0] @ vectors[2]),
                      "note": "Illustrative smoke test, not retrieval accuracy or a benchmark"}, indent=2))


if __name__ == "__main__":
    main()
