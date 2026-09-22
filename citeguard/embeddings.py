from sentence_transformers import SentenceTransformer

# small model, runs fine on CPU, no GPU required
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(texts):
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)
