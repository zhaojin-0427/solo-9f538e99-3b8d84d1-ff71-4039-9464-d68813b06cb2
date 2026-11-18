"""FastAPI 入口：流域网格发布与观测点批量查询。"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .config import settings
from .db import Database, VersionConflict
from .geometry import (
    GeometryError,
    classify_point,
    parse_line,
    parse_surface,
    split_line_by_grids,
    validate_lines,
    validate_submission,
)
from .schemas import (
    InsidePiece,
    OutsidePiece,
    PointResult,
    PublishRequest,
    PublishResponse,
    QueryRequest,
    QueryResponse,
    RiverSplitResult,
    SplitRequest,
    SplitResponse,
    PendingPiece,
    VersionInfo,
)
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

app = FastAPI(
    title="流域防汛网格服务",
    description=(
        "提交流域多边形与整版网格（互不重叠且恰好覆盖流域），按所见版号乐观锁发布；"
        "观测点按固定版号批量落格，河道中心线按固定版号逐网格切分，旧版持久保留可查。"
    ),
    version="1.1.0",
)

db = Database(settings.database_path)


def _error(status: int, error: str, detail: str, issues: list | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": error, "detail": detail, "issues": issues},
    )


@app.exception_handler(GeometryError)
async def geometry_error_handler(_: Request, exc: GeometryError) -> JSONResponse:
    return _error(422, "INVALID_GEOMETRY", str(exc))


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["运维"])
async def health() -> dict:
    return {"status": "ok", "current_version": db.current_version()}


@app.get("/versions", response_model=list[VersionInfo], tags=["版本"])
async def versions() -> list[dict]:
    return db.list_versions()


@app.post(
    "/publish",
    response_model=PublishResponse,
    tags=["发布"],
    responses={
        409: {"model": dict, "description": "版号冲突，整版拒绝，当前版不变"},
        422: {"model": dict, "description": "几何非法 / 网格重叠或未恰好覆盖流域，整版拒绝"},
    },
)
async def publish(req: PublishRequest) -> JSONResponse | PublishResponse:
    # 1) 解析流域
    watershed = parse_surface(req.watershed, "流域多边形")

    # 2) 解析网格并检查编号唯一
    grids: list[tuple[str, object]] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    for grid in req.grids:
        if grid.grid_id in seen_ids and grid.grid_id not in duplicates:
            duplicates.append(grid.grid_id)
        seen_ids.add(grid.grid_id)
        geom = parse_surface(grid.geometry, f"网格 {grid.grid_id}")
        grids.append((grid.grid_id, geom))
    if duplicates:
        return _error(
            422,
            "DUPLICATE_GRID_ID",
            f"网格编号必须唯一，重复编号：{', '.join(duplicates)}",
        )

    # 3) 几何整版校验：重叠 / 缺口 / 越界，任何问题整版拒绝
    issues = validate_submission(watershed, grids, settings.area_tol_rel)
    if issues:
        return _error(
            422,
            "COVERAGE_INVALID",
            "网格存在重叠或未恰好覆盖流域，整版拒绝（问题位置见 issues）",
            issues,
        )

    # 4) 乐观锁发布（首次发布 expected_version 传 null 或 0）
    expected = None if req.expected_version in (None, 0) else req.expected_version
    try:
        new_version = db.publish(expected, watershed, grids)
    except VersionConflict as exc:
        return _error(409, "VERSION_CONFLICT", str(exc))

    info = next(v for v in db.list_versions() if v["version"] == new_version)
    return PublishResponse(
        version=new_version,
        expected_version=expected,
        published_at=info["published_at"],
        grid_count=info["grid_count"],
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["查询"],
    responses={404: {"model": dict, "description": "指定版本不存在或尚无已发布版本"}},
)
async def query(req: QueryRequest) -> JSONResponse | QueryResponse:
    current = db.current_version()
    if req.version is None:
        if current is None:
            return _error(404, "NO_PUBLISHED_VERSION", "尚无已发布版本，请先调用 /publish")
        version = current
    else:
        version = req.version
        if not any(v["version"] == version for v in db.list_versions()):
            return _error(404, "VERSION_NOT_FOUND", f"版本 {version} 不存在")

    watershed, grids = db.load_snapshot(version)
    diag = max(watershed.bounds[2] - watershed.bounds[0],
               watershed.bounds[3] - watershed.bounds[1])
    eps = diag * settings.point_eps_rel

    results: list[PointResult] = []
    for p in req.points:
        verdict = classify_point(p.lon, p.lat, watershed, grids, eps)
        results.append(
            PointResult(
                point_id=p.point_id,
                lon=p.lon,
                lat=p.lat,
                status=verdict["status"],
                grid_id=verdict["grid_id"],
                grid_ids=verdict["grid_ids"],
            )
        )

    return QueryResponse(version=version, is_current=(version == current), results=results)


def _geojson(geom: BaseGeometry) -> dict:
    return mapping(geom)


@app.post(
    "/split",
    response_model=SplitResponse,
    tags=["切分"],
    responses={
        404: {"model": dict, "description": "指定版本不存在或尚无已发布版本"},
        422: {"model": dict, "description": "河道线非法 / 编号重复 / 自交或互相重叠，整次拒绝"},
    },
)
async def split(req: SplitRequest) -> JSONResponse | SplitResponse:
    # 1) 先做全部输入校验：任何一条线非法都整次拒绝，不产生部分结果
    rivers: list[tuple[str, object]] = []
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    for river in req.rivers:
        if river.river_id in seen_ids and river.river_id not in duplicates:
            duplicates.append(river.river_id)
        seen_ids.add(river.river_id)
        geom = parse_line(river.geometry, f"河道 {river.river_id}")
        rivers.append((river.river_id, geom))
    if duplicates:
        return _error(
            422,
            "DUPLICATE_RIVER_ID",
            f"河道编号必须唯一，重复编号：{', '.join(duplicates)}",
        )
    validate_lines(rivers, settings.line_overlap_tol_rel)

    # 2) 解析版号并一次性加载该版快照；之后全部计算只基于内存中的同一快照，
    #    计算期间即使有新发布，结果也不会与版号错配
    current = db.current_version()
    if req.version is None:
        if current is None:
            return _error(404, "NO_PUBLISHED_VERSION", "尚无已发布版本，请先调用 /publish")
        version = current
    else:
        version = req.version
        if not any(v["version"] == version for v in db.list_versions()):
            return _error(404, "VERSION_NOT_FOUND", f"版本 {version} 不存在")

    _watershed, grids = db.load_snapshot(version)
    diag = max(
        _watershed.bounds[2] - _watershed.bounds[0],
        _watershed.bounds[3] - _watershed.bounds[1],
    )
    eps = diag * settings.point_eps_rel

    # 3) 逐条按网格裁切（输入已整体通过校验，此处不再可能出现部分失败）
    results: list[RiverSplitResult] = []
    for river_id, line in rivers:
        outcome = split_line_by_grids(line, grids, eps)
        inside_pieces = [
            InsidePiece(
                grid_id=grid_id,
                geometry=_geojson(piece),
                length=piece.length,
            )
            for grid_id, pieces in outcome["inside"].items()
            for piece in pieces
        ]
        pending_pieces = [
            PendingPiece(
                geometry=_geojson(item["geometry"]),
                length=item["geometry"].length,
                grid_ids=item["grid_ids"],
            )
            for item in outcome["pending"]
        ]
        outside_pieces = [
            OutsidePiece(geometry=_geojson(piece), length=piece.length)
            for piece in outcome["outside"]
        ]
        results.append(
            RiverSplitResult(
                river_id=river_id,
                source_length=line.length,
                inside=inside_pieces,
                pending=pending_pieces,
                outside=outside_pieces,
            )
        )

    return SplitResponse(
        version=version, is_current=(version == current), results=results
    )
