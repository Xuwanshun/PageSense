"""
Unit tests for document_process/vlm.py — VLM visual description enrichment.

All OpenAI vision API calls are mocked. No real network calls, no API key needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import Settings
from document_process.models import VisualRegionSummary
from document_process.vlm import _describe_crop, enrich_summaries_with_vlm

# ── Helpers ───────────────────────────────────────────────────────────────────


def _settings(**overrides) -> Settings:
    base = dict(
        openai_api_key="sk-test-fake",
        use_vlm_summaries=True,
        vlm_model="gpt-4o",
        s3_bucket_name=None,
    )
    base.update(overrides)
    return Settings(**base)


def _summary(
    region_id: str = "r1",
    region_type: str = "table",
    summary_text: str = "some nearby OCR text",
    crop_path: str | None = None,
) -> VisualRegionSummary:
    return VisualRegionSummary(
        summary_id=f"summary_{region_id}",
        region_id=region_id,
        page_number=1,
        region_type=region_type,
        summary_text=summary_text,
        crop_path=crop_path,
    )


def _fake_openai_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ── enrich_summaries_with_vlm tests ──────────────────────────────────────────


class TestEnrichSummariesWithVlm:
    def test_summary_with_crop_gets_vlm_description(self, tmp_path):
        crop = tmp_path / "region_1.png"
        crop.write_bytes(b"PNG_FAKE_DATA")

        summaries = [_summary(crop_path=str(crop), summary_text="Detected table region on page 1.")]

        with patch(
            "document_process.vlm._describe_crop", return_value=("Q1–Q4 revenue table showing 25% growth", True)
        ) as mock_describe:
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        assert len(result) == 1
        assert result[0].summary_text == "Q1–Q4 revenue table showing 25% growth"
        assert result[0].is_meaningful is True
        mock_describe.assert_called_once()

    def test_summary_without_crop_not_sent_to_vlm(self):
        summaries = [_summary(crop_path=None)]

        with patch("document_process.vlm._describe_crop") as mock_describe:
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        mock_describe.assert_not_called()
        assert result[0].summary_text == "some nearby OCR text"  # unchanged

    def test_summary_with_missing_crop_file_not_sent_to_vlm(self, tmp_path):
        summaries = [_summary(crop_path=str(tmp_path / "nonexistent.png"))]

        with patch("document_process.vlm._describe_crop") as mock_describe:
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        mock_describe.assert_not_called()
        assert result[0].summary_text == "some nearby OCR text"  # unchanged

    def test_vlm_failure_keeps_original_summary(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG_FAKE_DATA")

        summaries = [_summary(crop_path=str(crop), summary_text="original text")]

        with patch("document_process.vlm._describe_crop", side_effect=RuntimeError("API timeout")):
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        assert result[0].summary_text == "original text"  # fallback preserved
        assert result[0].is_meaningful is True  # failure doesn't mark as not meaningful

    def test_input_summaries_not_mutated(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG_FAKE_DATA")

        original = _summary(crop_path=str(crop), summary_text="original")
        original_text = original.summary_text

        with patch("document_process.vlm._describe_crop", return_value=("new description", True)):
            enrich_summaries_with_vlm([original], settings=_settings())

        assert original.summary_text == original_text  # input not mutated

    def test_multiple_summaries_processed_independently(self, tmp_path):
        crop1 = tmp_path / "r1.png"
        crop1.write_bytes(b"PNG1")
        crop2 = tmp_path / "r2.png"
        crop2.write_bytes(b"PNG2")

        descriptions = {"r1": "VLM description for r1", "r2": "VLM description for r2"}

        def fake_describe(crop_path, region_type, context_text, settings):
            text = descriptions["r1"] if "r1" in str(crop_path) else descriptions["r2"]
            return (text, True)

        summaries = [
            _summary(region_id="r1", crop_path=str(crop1)),
            _summary(region_id="r2", crop_path=str(crop2)),
        ]

        with patch("document_process.vlm._describe_crop", side_effect=fake_describe):
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        assert result[0].summary_text == "VLM description for r1"
        assert result[1].summary_text == "VLM description for r2"

    def test_returns_all_summaries_including_unenriched(self, tmp_path):
        crop = tmp_path / "r1.png"
        crop.write_bytes(b"PNG")

        summaries = [
            _summary(region_id="r1", crop_path=str(crop)),  # has crop → enriched
            _summary(region_id="r2", crop_path=None),  # no crop → kept as-is
        ]

        with patch("document_process.vlm._describe_crop", return_value=("VLM text", True)):
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        assert len(result) == 2
        assert result[0].region_id == "r1"
        assert result[1].region_id == "r2"

    def test_not_meaningful_image_marked_and_text_preserved(self, tmp_path):
        crop = tmp_path / "logo.png"
        crop.write_bytes(b"PNG_LOGO")

        summaries = [_summary(crop_path=str(crop), summary_text="Qualcomm logo")]

        with patch("document_process.vlm._describe_crop", return_value=("", False)):
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        assert result[0].is_meaningful is False
        assert result[0].summary_text == "Qualcomm logo"  # original OCR text preserved

    def test_not_meaningful_does_not_update_summary_text(self, tmp_path):
        crop = tmp_path / "icon.png"
        crop.write_bytes(b"PNG_ICON")

        original_text = "some nearby OCR text"
        summaries = [_summary(crop_path=str(crop), summary_text=original_text)]

        with patch("document_process.vlm._describe_crop", return_value=("", False)):
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        assert result[0].summary_text == original_text  # text unchanged when not meaningful

    def test_meaningful_summary_has_is_meaningful_true_by_default(self):
        summary = _summary(crop_path=None)
        assert summary.is_meaningful is True

    def test_mixed_meaningful_and_not(self, tmp_path):
        crop1 = tmp_path / "chart.png"
        crop1.write_bytes(b"PNG")
        crop2 = tmp_path / "logo.png"
        crop2.write_bytes(b"PNG")

        summaries = [
            _summary(region_id="r1", crop_path=str(crop1), summary_text="chart text"),
            _summary(region_id="r2", crop_path=str(crop2), summary_text="logo text"),
        ]

        def fake_describe(crop_path, region_type, context_text, settings):
            if "chart" in str(crop_path):
                return ("Revenue grew 15% YoY", True)
            return ("", False)

        with patch("document_process.vlm._describe_crop", side_effect=fake_describe):
            result = enrich_summaries_with_vlm(summaries, settings=_settings())

        assert result[0].is_meaningful is True
        assert result[0].summary_text == "Revenue grew 15% YoY"
        assert result[1].is_meaningful is False
        assert result[1].summary_text == "logo text"  # unchanged


# ── _describe_crop tests ──────────────────────────────────────────────────────


class TestDescribeCrop:
    def test_raises_when_no_api_key(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings(openai_api_key=None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            _describe_crop(crop, "table", "context text", settings)

    def test_calls_openai_with_image_and_system_prompt(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNGDATA")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("a table about sales")

        with patch("openai.OpenAI", return_value=mock_client):
            description, is_meaningful = _describe_crop(crop, "table", "nearby text", settings)

        assert description == "a table about sales"
        assert is_meaningful is True
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "table" in messages[0]["content"].lower()  # table-specific prompt used

    def test_figure_prompt_used_for_figure_type(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNGDATA")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("a line chart")

        with patch("openai.OpenAI", return_value=mock_client):
            _describe_crop(crop, "figure", "", settings)

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert "figure" in messages[0]["content"].lower()  # figure-specific prompt used

    def test_image_sent_as_base64_data_url(self, tmp_path):
        import base64

        crop = tmp_path / "crop.png"
        image_bytes = b"FAKE_PNG_BYTES"
        crop.write_bytes(image_bytes)
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("description")

        with patch("openai.OpenAI", return_value=mock_client):
            _describe_crop(crop, "table", "context", settings)

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        image_part = next(p for p in user_content if p["type"] == "image_url")
        expected_b64 = base64.b64encode(image_bytes).decode()
        assert expected_b64 in image_part["image_url"]["url"]

    def test_real_context_text_included_in_prompt(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("desc")

        with patch("openai.OpenAI", return_value=mock_client):
            _describe_crop(crop, "table", "Revenue Q1 2024 table context", settings)

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        text_parts = [p for p in user_content if p["type"] == "text"]
        assert any("Revenue Q1 2024" in p["text"] for p in text_parts)

    def test_placeholder_context_text_excluded_from_prompt(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("desc")

        with patch("openai.OpenAI", return_value=mock_client):
            _describe_crop(crop, "table", "Detected table region on page 3.", settings)

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        # Placeholder text should not be passed to the model as context
        text_parts = [p for p in user_content if p["type"] == "text"]
        assert not any("Detected" in p["text"] for p in text_parts)

    def test_uses_vlm_model_from_settings(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings(vlm_model="gpt-4o-mini")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("desc")

        with patch("openai.OpenAI", return_value=mock_client):
            _describe_crop(crop, "table", "", settings)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_strips_whitespace_from_response(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("  description with spaces  \n")

        with patch("openai.OpenAI", return_value=mock_client):
            description, is_meaningful = _describe_crop(crop, "table", "", settings)

        assert description == "description with spaces"
        assert is_meaningful is True

    def test_skip_sentinel_returns_not_meaningful(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("SKIP")

        with patch("openai.OpenAI", return_value=mock_client):
            description, is_meaningful = _describe_crop(crop, "figure", "", settings)

        assert is_meaningful is False
        assert description == ""

    def test_skip_sentinel_case_insensitive(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("  skip  ")

        with patch("openai.OpenAI", return_value=mock_client):
            _, is_meaningful = _describe_crop(crop, "figure", "", settings)

        assert is_meaningful is False

    def test_system_prompt_contains_skip_instruction(self, tmp_path):
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"PNG")
        settings = _settings()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_openai_response("desc")

        with patch("openai.OpenAI", return_value=mock_client):
            _describe_crop(crop, "figure", "", settings)

        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "SKIP" in system_content  # sentinel instruction included in prompt
