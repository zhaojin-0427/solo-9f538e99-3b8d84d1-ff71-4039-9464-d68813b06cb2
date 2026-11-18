"""请求 / 响应 Pydantic 模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# 状态码：inside 内部点 / boundary 边界待判 / outside 流域外
PointStatus = Literal["inside", "boundary", "outside"]


class GridIn(BaseModel):
    grid_id: str = Field(..., min_length=1, max_length=128, description="网格唯一编号")
    geometry: dict[str, Any] = Field(..., description="Polygon 或 MultiPolygon 的 GeoJSON geometry")


class PublishRequest(BaseModel):
    expected_version: int | None = Field(
        None,
        ge=0,
        description="提交方所见的当前版号；首次发布传 null 或 0，之后传当前版号",
    )
    watershed: dict[str, Any] = Field(..., description="流域多边形 GeoJSON geometry")
    grids: list[GridIn] = Field(..., min_length=2, max_length=20, description="2~20 个带唯一编号的网格")


class ValidationIssue(BaseModel):
    type: Literal["overlap", "gap", "excess"]
    location: dict[str, float]
    representative_point: list[float]
    area: float
    message: str


class PublishResponse(BaseModel):
    version: int
    expected_version: int | None
    published_at: str
    grid_count: int
    status: Literal["published"] = "published"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    issues: list[ValidationIssue] | None = None


class QueryPoint(BaseModel):
    point_id: str | None = Field(None, max_length=128, description="观测点编号，可选，便于结果对应")
    lon: float
    lat: float


class QueryRequest(BaseModel):
    version: int | None = Field(
        None, ge=1, description="固定查询版号；不传或传 null 表示当前版，一次查询内所有点同版"
    )
    points: list[QueryPoint] = Field(..., min_length=1, max_length=100, description="至多 100 个观测点")


class PointResult(BaseModel):
    point_id: str | None
    lon: float
    lat: float
    status: PointStatus
    grid_id: str | None = Field(None, description="内部点返回唯一网格编号；其余为 null")
    grid_ids: list[str] = Field(
        default_factory=list, description="边界点列出所有相接网格（待判）；域外为空"
    )


class QueryResponse(BaseModel):
    version: int = Field(..., description="本次查询实际使用的版号")
    is_current: bool
    results: list[PointResult]


# ---------- 河道按网格切分 ----------

class RiverIn(BaseModel):
    river_id: str = Field(..., min_length=1, max_length=128, description="河道唯一编号")
    geometry: dict[str, Any] = Field(
        ..., description="LineString 或 MultiLineString 的 GeoJSON geometry"
    )


class SplitRequest(BaseModel):
    version: int | None = Field(
        None,
        ge=1,
        description="固定切分所用已发布网格版号；不传或传 null 表示受理时当前版",
    )
    rivers: list[RiverIn] = Field(
        ..., min_length=1, max_length=10, description="1～10 条编号唯一的河道线"
    )


class InsidePiece(BaseModel):
    grid_id: str = Field(..., description="该格内线段所属网格编号")
    geometry: dict[str, Any] = Field(..., description="格内部线段 GeoJSON（LineString/MultiLineString）")
    length: float = Field(..., description="该格内部线段长度")


class PendingPiece(BaseModel):
    geometry: dict[str, Any] = Field(..., description="沿网格边界延伸的线段 GeoJSON")
    length: float
    grid_ids: list[str] = Field(
        ..., description="与该边界线段相接的全部网格编号（待判）"
    )


class OutsidePiece(BaseModel):
    geometry: dict[str, Any] = Field(..., description="流域外 / 孔洞内线段 GeoJSON")
    length: float


class RiverSplitResult(BaseModel):
    river_id: str
    source_length: float = Field(..., description="原始河道线总长度")
    inside: list[InsidePiece] = Field(
        default_factory=list, description="各网格内部线段及长度，沿边界部分不在其中"
    )
    pending: list[PendingPiece] = Field(
        default_factory=list, description="沿网格边界延伸、单列待判的线段"
    )
    outside: list[OutsidePiece] = Field(
        default_factory=list, description="流域外及孔洞内线段，不计入任何格"
    )


class SplitResponse(BaseModel):
    version: int = Field(..., description="本次切分实际使用的网格版号")
    is_current: bool
    results: list[RiverSplitResult]


class VersionInfo(BaseModel):
    version: int
    expected_version: int | None
    published_at: str
    grid_count: int
