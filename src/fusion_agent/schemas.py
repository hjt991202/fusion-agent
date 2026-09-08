"""Validated contracts shared by CLI, tools and agents."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Method = Literal["weighted", "wavelet"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ImageInfo(Contract):
    name: str
    width: int
    height: int
    channels: int
    dtype: str
    sha256: str
    warnings: list[str] = Field(default_factory=list)


class PairInfo(Contract):
    visible: ImageInfo
    infrared: ImageInfo
    same_size: bool


class FusionPlan(Contract):
    methods: list[Method] = Field(default_factory=lambda: ["weighted", "wavelet"], min_length=1)
    visible_weight: float = Field(default=0.5, ge=0, le=1)
    resize: bool = False
    rationale: str = "Compare two CPU baselines."


class MetricResult(Contract):
    entropy: float = Field(ge=0)
    standard_deviation: float = Field(ge=0)
    average_gradient: float = Field(ge=0)
    spatial_frequency: float = Field(ge=0)


class ToolRecord(Contract):
    tool: str
    seconds: float = Field(ge=0)
    details: str


class AnalysisReport(Contract):
    task: str
    mode: str
    inputs: PairInfo
    plan: FusionPlan
    metrics: dict[str, MetricResult]
    recommendation: str
    reason: str
    limitations: list[str]
    records: list[ToolRecord]
    files: dict[str, str]
