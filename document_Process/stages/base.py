from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class StageRunInfo:
    stage_name: str
    stage_version: str
    duration_seconds: float
    item_count_in: int
    item_count_out: int
    skipped_count: int
    failed_count: int
    cache_hit: bool


@runtime_checkable
class StageProtocol(Protocol[InputT, OutputT]):
    stage_name: str
    stage_version: str

    def run(self, *args: Any, **kwargs: Any) -> OutputT: ...

    def cache_key(self, *args: Any, **kwargs: Any) -> str: ...


@runtime_checkable
class AsyncStageProtocol(Protocol[InputT, OutputT]):
    stage_name: str
    stage_version: str

    async def run(self, *args: Any, **kwargs: Any) -> OutputT: ...

    def cache_key(self, *args: Any, **kwargs: Any) -> str: ...
