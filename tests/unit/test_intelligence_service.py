from __future__ import annotations

from unittest.mock import MagicMock, patch

from document_process.intelligence_service import (
    DocumentIntelligenceService,
    Section,
    _safe_parse_json,
)
from document_process.models import BoundingBox, LayoutRegion, OrderedTextBlock


def _make_region(
    region_id: str,
    region_type: str,
    y0: float,
    label: str,
    page: int = 1,
    crop_path: str | None = None,
) -> LayoutRegion:
    return LayoutRegion(
        region_id=region_id,
        region_type=region_type,
        page_number=page,
        bbox=BoundingBox(x0=0.0, y0=y0, x1=100.0, y1=y0 + 20.0),
        metadata={"label": label, "detector": "test"},
        crop_path=crop_path,
    )


def _make_block(block_id: str, text: str, region_ids: list[str], page: int = 1) -> OrderedTextBlock:
    return OrderedTextBlock(
        block_id=block_id,
        page_number=page,
        text=text,
        region_ids=region_ids,
        reading_order=1,
    )


# ── _assign_titles ────────────────────────────────────────────────────────────


def test_assign_titles_stamps_parent_fields(tmp_settings):
    service = DocumentIntelligenceService(tmp_settings)

    title_region = _make_region("r1", "text_block", 0.0, "title")
    subtitle_region = _make_region("r2", "text_block", 25.0, "subtitle")
    text_region = _make_region("r3", "text_block", 50.0, "text")
    table_region = _make_region("r4", "table", 80.0, "table")

    blocks = [
        _make_block("b1", "Introduction", ["r1"]),
        _make_block("b2", "Overview", ["r2"]),
    ]
    regions = [title_region, subtitle_region, text_region, table_region]
    service._assign_titles(regions, blocks)

    # Non-title regions should have parent fields stamped
    assert text_region.metadata["parent_title"] == "Introduction"
    assert text_region.metadata["parent_subtitle"] == "Overview"
    assert table_region.metadata["parent_title"] == "Introduction"
    assert table_region.metadata["parent_subtitle"] == "Overview"

    # Title and subtitle regions themselves should NOT have parent_title
    assert "parent_title" not in title_region.metadata
    assert "parent_title" not in subtitle_region.metadata


def test_assign_titles_defaults_to_untitled(tmp_settings):
    service = DocumentIntelligenceService(tmp_settings)
    text_region = _make_region("r1", "text_block", 10.0, "text")
    service._assign_titles([text_region], [])
    assert text_region.metadata["parent_title"] == "untitled"
    assert text_region.metadata["parent_subtitle"] is None


def test_assign_titles_resets_subtitle_on_new_title(tmp_settings):
    service = DocumentIntelligenceService(tmp_settings)

    t1 = _make_region("r1", "text_block", 0.0, "title")
    sub = _make_region("r2", "text_block", 20.0, "subtitle")
    body1 = _make_region("r3", "text_block", 40.0, "text")
    t2 = _make_region("r4", "text_block", 60.0, "title")
    body2 = _make_region("r5", "text_block", 80.0, "text")

    blocks = [
        _make_block("b1", "Chapter 1", ["r1"]),
        _make_block("b2", "Intro", ["r2"]),
        _make_block("b3", "Chapter 2", ["r4"]),
    ]
    service._assign_titles([t1, sub, body1, t2, body2], blocks)

    assert body1.metadata["parent_title"] == "Chapter 1"
    assert body1.metadata["parent_subtitle"] == "Intro"
    # After new title, subtitle resets to None
    assert body2.metadata["parent_title"] == "Chapter 2"
    assert body2.metadata["parent_subtitle"] is None


# ── _group_into_sections ──────────────────────────────────────────────────────


def test_group_into_sections_counts(tmp_settings):
    service = DocumentIntelligenceService(tmp_settings)

    # Pre-stamp metadata as _assign_titles would
    r_text = _make_region("r1", "text_block", 10.0, "text")
    r_text.metadata["parent_title"] = "Methods"
    r_text.metadata["parent_subtitle"] = None

    r_table = _make_region("r2", "table", 30.0, "table")
    r_table.metadata["parent_title"] = "Methods"
    r_table.metadata["parent_subtitle"] = None

    r_chart = _make_region("r3", "figure", 50.0, "chart")
    r_chart.metadata["parent_title"] = "Methods"
    r_chart.metadata["parent_subtitle"] = None

    r_fig = _make_region("r4", "figure", 70.0, "figure")
    r_fig.metadata["parent_title"] = "Methods"
    r_fig.metadata["parent_subtitle"] = None

    blocks = [_make_block("b1", "Hello world", ["r1"])]
    sections = service._group_into_sections([r_text, r_table, r_chart, r_fig], blocks)

    assert len(sections) == 1
    section = sections[0]
    assert section.title == "Methods"
    assert section.text_blocks == 1
    assert section.tables == 1
    assert section.charts == 1
    assert section.figures == 1


def test_group_into_sections_flat_text_assembled(tmp_settings):
    service = DocumentIntelligenceService(tmp_settings)

    r1 = _make_region("r1", "text_block", 10.0, "text")
    r1.metadata["parent_title"] = "Intro"
    r1.metadata["parent_subtitle"] = None

    r2 = _make_region("r2", "text_block", 30.0, "text")
    r2.metadata["parent_title"] = "Intro"
    r2.metadata["parent_subtitle"] = None

    blocks = [
        _make_block("b1", "First sentence.", ["r1"]),
        _make_block("b2", "Second sentence.", ["r2"]),
    ]
    sections = service._group_into_sections([r1, r2], blocks)

    assert len(sections) == 1
    assert "Intro" in sections[0].flat_text
    assert "First sentence." in sections[0].flat_text
    assert "Second sentence." in sections[0].flat_text


def test_group_into_sections_multiple_sections(tmp_settings):
    service = DocumentIntelligenceService(tmp_settings)

    r1 = _make_region("r1", "text_block", 10.0, "text")
    r1.metadata["parent_title"] = "Intro"
    r1.metadata["parent_subtitle"] = None

    r2 = _make_region("r2", "table", 30.0, "table")
    r2.metadata["parent_title"] = "Results"
    r2.metadata["parent_subtitle"] = None

    sections = service._group_into_sections([r1, r2], [])

    assert len(sections) == 2
    assert sections[0].title == "Intro"
    assert sections[1].title == "Results"


# ── _safe_parse_json ──────────────────────────────────────────────────────────


def test_safe_parse_json_valid():
    result = _safe_parse_json('{"strategy": "layout_aware", "chunk_size": 2000}')
    assert result["strategy"] == "layout_aware"
    assert result["chunk_size"] == 2000


def test_safe_parse_json_markdown_fences():
    text = '```json\n{"doc_type": "report", "domain": "finance"}\n```'
    result = _safe_parse_json(text)
    assert result["doc_type"] == "report"
    assert result["domain"] == "finance"


def test_safe_parse_json_embedded_object():
    text = 'Here is the result: {"summary": "A document.", "has_data": true} and nothing else.'
    result = _safe_parse_json(text)
    assert result["summary"] == "A document."


def test_safe_parse_json_broken_recovers_fields():
    text = 'This is broken JSON but has "summary": "overview text" inside.'
    result = _safe_parse_json(text)
    assert result.get("parse_error") is True
    assert result.get("summary") == "overview text"


def test_safe_parse_json_completely_unparseable():
    result = _safe_parse_json("not json at all, just garbage text !!!!")
    assert isinstance(result, dict)
    assert result.get("parse_error") is True


# ── _decide_strategy hard overrides ──────────────────────────────────────────


def test_decide_strategy_disabled_returns_defaults(tmp_settings):
    service = DocumentIntelligenceService(tmp_settings)
    result = service._decide_strategy({}, [], 10)
    assert result["strategy"] == "semantic_fixed"
    assert result["chunk_size"] == tmp_settings.preprocess_chunk_size
    assert result["overlap"] == tmp_settings.preprocess_chunk_overlap
    assert "override" not in result


def test_decide_strategy_table_override(tmp_settings):
    settings = tmp_settings.model_copy(update={"use_adaptive_chunking": True})
    service = DocumentIntelligenceService(settings)

    # 1 section with 3 tables → tables(3) > sections(1) / 2 = 0.5
    section = Section(title="financials", subtitle=None, tables=3)

    with patch.object(service, "_build_descriptor_client") as mock_builder:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = (
            '{"strategy": "semantic_fixed", "chunk_size": 1800, '
            '"overlap": 200, "keep_tables_intact": false, "reason": "default"}'
        )
        mock_builder.return_value = mock_client

        result = service._decide_strategy({"well_structured": True}, [section], 10)

    assert result["strategy"] == "layout_aware"
    assert result["keep_tables_intact"] is True
    assert "override" in result


def test_decide_strategy_unstructured_override(tmp_settings):
    settings = tmp_settings.model_copy(update={"use_adaptive_chunking": True})
    service = DocumentIntelligenceService(settings)

    # 2 sections, not well-structured → triggers recursive_large override
    sections = [
        Section(title="untitled", subtitle=None, text_blocks=5),
        Section(title="appendix", subtitle=None, text_blocks=2),
    ]
    descriptor = {"well_structured": False, "doc_type": "report", "domain": "unknown"}

    with patch.object(service, "_build_descriptor_client") as mock_builder:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = (
            '{"strategy": "semantic_fixed", "chunk_size": 1500, '
            '"overlap": 150, "keep_tables_intact": false, "reason": "default"}'
        )
        mock_builder.return_value = mock_client

        result = service._decide_strategy(descriptor, sections, 5)

    assert result["strategy"] == "recursive_large"
    assert result["chunk_size"] >= 2000
    assert "override" in result
