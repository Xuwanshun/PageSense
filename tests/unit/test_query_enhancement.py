from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.query_enhancement import classify_query, decompose_query, hyde_enhance

# ── classify_query ────────────────────────────────────────────────────────────


def test_classify_query_returns_simple(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "simple"
        mock_build.return_value = mock_client

        result = classify_query("What is the capital of France?", tmp_settings)

    assert result == "simple"
    mock_client.generate_text.assert_called_once()


def test_classify_query_returns_complex(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "complex"
        mock_build.return_value = mock_client

        result = classify_query("What is X and how does it compare to Y?", tmp_settings)

    assert result == "complex"


def test_classify_query_defaults_to_simple_on_unexpected_output(tmp_settings):
    """Any response that does not contain 'complex' is treated as simple."""
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "I cannot determine"
        mock_build.return_value = mock_client

        result = classify_query("What is X?", tmp_settings)

    assert result == "simple"


def test_classify_query_case_insensitive(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "COMPLEX"
        mock_build.return_value = mock_client

        result = classify_query("What is X and also Y?", tmp_settings)

    assert result == "complex"


# ── hyde_enhance ──────────────────────────────────────────────────────────────


def test_hyde_enhance_returns_stripped_text(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "  The system processes data in real time.  "
        mock_build.return_value = mock_client

        result = hyde_enhance("How does the system process data?", tmp_settings)

    assert result == "The system processes data in real time."


def test_hyde_enhance_passes_question_as_user_prompt(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "Some hypothetical answer."
        mock_build.return_value = mock_client

        question = "What is the retention policy?"
        hyde_enhance(question, tmp_settings)

    call_kwargs = mock_client.generate_text.call_args.kwargs
    assert call_kwargs["user_prompt"] == question


# ── decompose_query ───────────────────────────────────────────────────────────


def test_decompose_query_parses_numbered_list(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = (
            "1. What is the revenue for Q1?\n"
            "2. What is the cost structure?\n"
            "3. How does revenue compare year-over-year?"
        )
        mock_build.return_value = mock_client

        result = decompose_query("What is the revenue and cost structure?", tmp_settings)

    assert len(result) == 3
    assert result[0] == "What is the revenue for Q1?"
    assert result[1] == "What is the cost structure?"
    assert result[2] == "How does revenue compare year-over-year?"


def test_decompose_query_strips_parenthesis_numbering(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "1) First sub-query\n2) Second sub-query"
        mock_build.return_value = mock_client

        result = decompose_query("First and second topic?", tmp_settings)

    assert result == ["First sub-query", "Second sub-query"]


def test_decompose_query_falls_back_to_original_on_empty_response(tmp_settings):
    """If the LLM returns blank output, the original question is returned."""
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "   "
        mock_build.return_value = mock_client

        original = "What is X and Y?"
        result = decompose_query(original, tmp_settings)

    assert result == [original]


def test_decompose_query_ignores_blank_lines(tmp_settings):
    with patch("rag.query_enhancement.build_openai_client") as mock_build:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "1. Alpha\n\n2. Beta\n\n"
        mock_build.return_value = mock_client

        result = decompose_query("Alpha and Beta?", tmp_settings)

    assert result == ["Alpha", "Beta"]
