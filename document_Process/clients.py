"""
OpenAI API client wrappers.

WHY the error handling was added
---------------------------------
The original code made OpenAI API calls with no try/except. Any network
hiccup, rate-limit response, or invalid API key would crash the entire
pipeline with a raw openai exception — no helpful message, no context.

Now every OpenAI call is wrapped to catch the most common failure modes
and re-raise them as RuntimeError with a clear, actionable message.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from config import Settings

logger = logging.getLogger(__name__)
ModelT = TypeVar("ModelT", bound=BaseModel)


class OpenAIJSONModelClient:
    def __init__(self, *, model: str, api_key: str, base_url: str | None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_structured(self, *, system_prompt: str, user_prompt: str, response_model: type[ModelT]) -> ModelT:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI rate limit reached. Wait a moment and retry, or reduce request frequency."
            ) from exc
        except APITimeoutError as exc:
            raise RuntimeError("OpenAI API request timed out. Check your network connection.") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Could not connect to OpenAI API: {exc}") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI API returned an error (HTTP {exc.status_code}): {exc.message}") from exc

        content = str((response.choices[0].message.content or "").strip())
        return _validate_response_model(response_model, _extract_json_from_text(content))

    def generate_text(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI rate limit reached. Wait a moment and retry, or reduce request frequency."
            ) from exc
        except APITimeoutError as exc:
            raise RuntimeError("OpenAI API request timed out. Check your network connection.") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"Could not connect to OpenAI API: {exc}") from exc
        except APIStatusError as exc:
            raise RuntimeError(f"OpenAI API returned an error (HTTP {exc.status_code}): {exc.message}") from exc

        return str(response.choices[0].message.content or "")


def build_openai_client(settings: Settings) -> OpenAIJSONModelClient:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. " "Add it to your .env file or set it as an environment variable."
        )
    return OpenAIJSONModelClient(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def request_openai_embeddings(*, model: str, texts: list[str], api_key: str, base_url: str | None) -> list[list[float]]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.embeddings.create(model=model, input=texts)
    except RateLimitError as exc:
        raise RuntimeError(
            "OpenAI rate limit reached while generating embeddings. "
            "Wait a moment and retry, or reduce the number of texts per batch."
        ) from exc
    except APITimeoutError as exc:
        raise RuntimeError("OpenAI embedding request timed out. Check your network connection.") from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to OpenAI API for embeddings: {exc}") from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"OpenAI API returned an error (HTTP {exc.status_code}) during embedding: {exc.message}"
        ) from exc

    return [item.embedding for item in response.data]


def _extract_json_from_text(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise RuntimeError("OpenAI model did not return valid JSON.") from None


def _validate_response_model(response_model: type[ModelT], payload: dict[str, Any]) -> ModelT:
    try:
        return response_model.model_validate(payload)
    except Exception:
        normalized = _normalize_model_payload(payload)
        return response_model.model_validate(normalized)


def _normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    relevant_region_ids = normalized.get("relevant_region_ids")
    if relevant_region_ids is None:
        normalized["relevant_region_ids"] = []
    elif isinstance(relevant_region_ids, str):
        normalized["relevant_region_ids"] = [relevant_region_ids]
    return normalized
