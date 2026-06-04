#!/usr/bin/env python3
"""A′ proof — embed with multilingual-e5-small via MLX (the Qwen runtime
family, NOT ONNX Runtime), then dump vectors as JSON so the Rust Memory Plant
engine can ingest them through `add_document_with_embeddings` + `search`.

This validates the on-device pipeline minus the phone:
    text --(e5 via MLX)--> 384-d vector --> Memory Plant (Rust) --> ranked hits

e5 needs prefixes: "query: " for searches, "passage: " for stored docs.
Outputs L2-normalized vectors (cosine == dot product downstream).
"""
import json
import sys

import mlx.core as mx
from mlx_embeddings.utils import load

MODEL = "intfloat/multilingual-e5-small"

# ru / en / kz passages (stored docs) + a ru query. The ru query must rank the
# ru passage first, above the en/kz distractors.
DOCS = [
    ("ru_tolstoy", "passage: Роман «Война и мир» написал Лев Толстой."),
    ("en_cell",    "passage: The mitochondria is the powerhouse of the cell."),
    ("kz_astana",  "passage: Қазақстанның астанасы — Астана қаласы."),
    ("ru_recipe",  "passage: Чтобы сварить борщ, нужны свёкла, капуста и мясо."),
]
QUERY = ("query: Кто написал роман Война и мир?", "ru_tolstoy")  # (text, expected top doc)


def l2(v):
    v = mx.array(v)
    n = mx.sqrt(mx.sum(v * v))
    return (v / mx.maximum(n, 1e-12)).tolist()


def embed(model, tokenizer, text):
    inp = tokenizer.batch_encode_plus(
        [text], return_tensors="mlx", padding=True, truncation=True, max_length=512
    )
    out = model(inp["input_ids"], attention_mask=inp["attention_mask"])
    # mlx-embeddings exposes mean-pooled, normalized sentence vectors as
    # `text_embeds`; fall back to pooling last_hidden_state if absent.
    emb = getattr(out, "text_embeds", None)
    if emb is None:
        lhs = out.last_hidden_state
        mask = inp["attention_mask"][..., None]
        emb = (lhs * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1)
    return l2(emb[0].tolist())


def main():
    model, tokenizer = load(MODEL)
    docs = [{"id": did, "text": txt, "emb": embed(model, tokenizer, txt)} for did, txt in DOCS]
    q_text, expected = QUERY
    out = {
        "model": MODEL,
        "dim": len(docs[0]["emb"]),
        "docs": docs,
        "query": {"text": q_text, "emb": embed(model, tokenizer, q_text), "expected_top": expected},
    }
    path = sys.argv[1] if len(sys.argv) > 1 else "e5_mlx_vectors.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}: dim={out['dim']}, docs={len(docs)}, expected_top={expected}")


if __name__ == "__main__":
    main()
