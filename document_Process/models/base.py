from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RegionType = Literal["text_block", "table", "figure", "chart", "image"]


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def intersection_area(self, other: BoundingBox) -> float:
        overlap_x0 = max(self.x0, other.x0)
        overlap_y0 = max(self.y0, other.y0)
        overlap_x1 = min(self.x1, other.x1)
        overlap_y1 = min(self.y1, other.y1)
        if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
            return 0.0
        return (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)

    def is_valid(self) -> bool:
        return self.x1 > self.x0 and self.y1 > self.y0

    @classmethod
    def from_list(cls, values: list[float]) -> BoundingBox:
        return cls(
            x0=float(values[0]),
            y0=float(values[1]),
            x1=float(values[2]),
            y1=float(values[3]),
        )

    @classmethod
    def merge(cls, boxes: list[BoundingBox]) -> BoundingBox | None:
        if not boxes:
            return None
        return cls(
            x0=min(box.x0 for box in boxes),
            y0=min(box.y0 for box in boxes),
            x1=max(box.x1 for box in boxes),
            y1=max(box.y1 for box in boxes),
        )


class ProcessingIssue(BaseModel):
    code: str
    message: str
    level: Literal["warning", "error"]
    page_number: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
