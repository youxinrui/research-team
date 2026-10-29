from __future__ import annotations

from research_team.rag.chunker import chunk_text, load_documents


def test_chunk_text_groups_short_paragraphs():
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = chunk_text(text, "doc.md", chunk_size=500, overlap=80)

    assert len(chunks) == 1
    assert chunks[0].id == "doc.md#chunk-0"
    assert chunks[0].source == "doc.md"
    assert "第一段" in chunks[0].text


def test_chunk_text_splits_long_text():
    text = "甲" * 1200
    chunks = chunk_text(text, "long.md", chunk_size=500, overlap=80)

    assert len(chunks) >= 3
    assert all(chunk.index == i for i, chunk in enumerate(chunks))
    assert all(len(chunk.text) <= 500 for chunk in chunks)


def test_load_documents_only_reads_text_formats(tmp_path):
    (tmp_path / "a.md").write_text("Alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Beta", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("ignored", encoding="utf-8")

    documents = dict(load_documents(tmp_path))

    assert documents == {"a.md": "Alpha", "b.txt": "Beta"}
