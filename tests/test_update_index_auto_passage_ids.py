"""Appending to an existing index must not be blocked by the IDs its own builder chose.

``add_text()`` without an ID stamps ``str(len(self.chunks))`` -- this builder's insertion position
-- and ``update_index()`` then compared that stamp against the *target* index, so the first
appended chunk was always rejected as a duplicate of passage ``0``.  Auto-assigned IDs have to be
resolved against the index being updated; the caller's example for that flow
(``examples/dynamic_update_no_recompute.py``) has to hand every chunk a globally unique ``id`` to
get the same result.
"""

import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import leann.api as api
import leann_backend_ivf  # noqa: F401  -- registers the "ivf" backend
import numpy as np
import pytest
from leann.api import LeannBuilder

VECTORS = {
    text: np.random.default_rng(ord(text)).standard_normal(8).astype(np.float32)
    for text in "abcdefgh"
}


def _embed(texts, *args, **kwargs):
    return np.stack([VECTORS[text] for text in texts])


def _builder(**kwargs):
    return LeannBuilder(backend_name="ivf", dimensions=8, nlist=2, **kwargs)


def _build(index: Path, texts, **kwargs):
    with patch.object(api, "compute_embeddings", side_effect=_embed):
        builder = _builder(**kwargs)
        for text in texts:
            builder.add_text(text)
        builder.build_index(str(index))


def _append(index: Path, texts, remove_passage_ids=None, **kwargs):
    with patch.object(api, "compute_embeddings", side_effect=_embed):
        builder = _builder(**kwargs)
        for text in texts:
            builder.add_text(text)
        builder.update_index(str(index), remove_passage_ids=remove_passage_ids)


def _text_of(index: Path, passage_id: str) -> str:
    """Resolve a passage ID the way the searcher does: offset map, then one JSONL line."""
    with open(f"{index}.passages.idx", "rb") as f:
        offsets = pickle.load(f)
    with open(f"{index}.passages.jsonl", "rb") as f:
        f.seek(offsets[passage_id])
        return json.loads(f.readline().decode("utf-8"))["text"]


def _id_map(index: Path) -> dict:
    with open(index.parent / f"{index.stem}.ivf_id_map.json", encoding="utf-8") as f:
        return json.load(f)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_append_without_ids_adds_to_the_existing_index():
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "docs.leann"
        _build(index, "abc")

        _append(index, "de")

        assert _text_of(index, "0") == "a"
        assert _text_of(index, "2") == "c"
        assert _text_of(index, "3") == "d"
        assert _text_of(index, "4") == "e"


def test_append_after_a_removal_does_not_reuse_an_id_that_is_still_there():
    """Deleting a passage makes the index shorter, so an ID derived from its size is already used."""
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "docs.leann"
        _build(index, "abc")

        _append(index, "d", remove_passage_ids=["0"])

        assert _text_of(index, "1") == "b"
        assert _text_of(index, "2") == "c"
        assert _text_of(index, "0") == "d"
        stored = _id_map(index)["id_to_passage"]
        assert sorted(stored.values()) == ["0", "1", "2"]


def test_a_caller_chosen_duplicate_id_is_still_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "docs.leann"
        _build(index, "abc")

        with patch.object(api, "compute_embeddings", side_effect=_embed):
            builder = _builder()
            builder.add_text("d", metadata={"id": "1"})
            with pytest.raises(ValueError, match="already exists"):
                builder.update_index(str(index))


def test_a_content_hash_id_is_still_the_id_of_that_content():
    """Content-hash IDs are derived from the text, not from a position, so they stay authoritative."""
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "docs.leann"
        scheme = {"passage_id_scheme": "content-hash"}
        _build(index, "abc", **scheme)

        _append(index, "d", **scheme)

        assert _text_of(index, _hash("d")) == "d"
        assert _text_of(index, _hash("a")) == "a"


def test_a_non_compact_hnsw_index_appends_without_ids_too():
    """The rejection happens in ``update_index`` before any backend runs, so HNSW hits it as well."""
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "docs.leann"
        with patch.object(api, "compute_embeddings", side_effect=_embed):
            builder = LeannBuilder(backend_name="hnsw", dimensions=8, is_compact=False)
            for text in "abc":
                builder.add_text(text)
            builder.build_index(str(index))

        with patch.object(api, "compute_embeddings", side_effect=_embed):
            builder = LeannBuilder(backend_name="hnsw", dimensions=8, is_compact=False)
            for text in "de":
                builder.add_text(text)
            builder.update_index(str(index))

        assert _text_of(index, "3") == "d"
        assert _text_of(index, "4") == "e"
