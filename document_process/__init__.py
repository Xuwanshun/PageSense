"""Document preprocessing pipeline aligned to the architecture diagram."""

from document_process.pipeline import (
    DocumentPreprocessingPipeline,
    PreprocessingResult,
    preprocess_document,
)

__all__ = ["DocumentPreprocessingPipeline", "PreprocessingResult", "preprocess_document"]
