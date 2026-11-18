"""Shapely 几何处理：整版网格校验、观测点落格判定与河道线按网格切分。"""
from __future__ import annotations

import math
from typing import Any

from shapely.geometry import shape, Point, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union, split as shapely_split
from shapely.strtree import STRtree
from shapely.validation import explain_validity

# 缺口 / 重叠报告最多返回的局部数量，避免错误负载过大
MAX_ISSUE_PARTS = 20
SUPPORTED_SURFACE_TYPES = {"Polygon", "MultiPolygon"}
SUPPORTED_LINE_TYPES = {"LineString", "MultiLineString"}


class GeometryError(ValueError):
    """请求中的 GeoJSON 无法解析或不是合法的目标几何。"""


def _bbox_dict(geom: BaseGeometry) -> dict[str, float]:
    minx, miny, maxx, maxy = geom.bounds
    return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}


def _issue(kind: str, geom: BaseGeometry, message: str) -> dict[str, Any]:
    return {
        "type": kind,
        "location": _bbox_dict(geom),
        "representative_point": list(geom.representative_point().coords[0]),
        "area": geom.area,
        "message": message,
    }


def parse_surface(geojson: dict[str, Any], label: str) -> BaseGeometry:
    """把 GeoJSON geometry 解析为 Polygon/MultiPolygon，并做有效性检查。"""
    if not isinstance(geojson, dict):
        raise GeometryError(f"{label} 必须是 GeoJSON geometry 对象")
    geom_type = geojson.get("type")
    if geom_type not in SUPPORTED_SURFACE_TYPES:
        raise GeometryError(
            f"{label} 的 type 必须是 Polygon 或 MultiPolygon，收到 {geom_type!r}"
        )
    try:
        geom = shape(geojson)
    except Exception as exc:  # shapely 对非法 GeoJSON 抛 GEOSException/ValueError
        raise GeometryError(f"{label} 不是合法的 GeoJSON：{exc}") from exc
    if not geom.is_valid:
        raise GeometryError(f"{label} 几何无效：{explain_validity(geom)}")
    if geom.is_empty or geom.area <= 0.0:
        raise GeometryError(f"{label} 为空或面积为零")
    return geom


def _coords_finite(geojson: dict[str, Any]) -> bool:
    """递归检查 GeoJSON 坐标树中所有数值是否为有限实数（拒绝 NaN/Infinity）。"""
    stack: list[Any] = [geojson.get("coordinates")]
    while stack:
        node = stack.pop()
        if isinstance(node, (int, float)):
            if not math.isfinite(float(node)):
                return False
        elif isinstance(node, list):
            stack.extend(node)
        elif node is None:
            return False
        else:
            return False
    return True


def parse_line(geojson: dict[str, Any], label: str) -> BaseGeometry:
    """把 GeoJSON geometry 解析为 LineString/MultiLineString 并做基础检查。

    仅检查：类型、坐标有限、可解析、几何有效、总长度为正。
    编号唯一与线间重叠属于整次请求级校验，由 validate_lines 统一处理。
    """
    if not isinstance(geojson, dict):
        raise GeometryError(f"{label} 必须是 GeoJSON geometry 对象")
    geom_type = geojson.get("type")
    if geom_type not in SUPPORTED_LINE_TYPES:
        raise GeometryError(
            f"{label} 的 type 必须是 LineString 或 MultiLineString，收到 {geom_type!r}"
        )
    if not _coords_finite(geojson):
        raise GeometryError(f"{label} 坐标必须全部为有限数字，不得为 NaN/Infinity/null")
    try:
        geom = shape(geojson)
    except Exception as exc:  # shapely 对非法 GeoJSON 抛 GEOSException/ValueError
        raise GeometryError(f"{label} 不是合法的 GeoJSON：{exc}") from exc
    if not geom.is_valid:
        raise GeometryError(f"{label} 几何无效：{explain_validity(geom)}")
    # OGC 中自交折线仍可能 is_valid，必须用 is_simple 才能识别自交 / 自触
    if not geom.is_simple:
        raise GeometryError(f"{label} 存在自交或自接触，河道线不得自交")
    if geom.is_empty or geom.length <= 0.0:
        raise GeometryError(f"{label} 为空或总长度为零")
    return geom


def _line_components(geom: BaseGeometry) -> list[BaseGeometry]:
    """把 LineString/MultiLineString 展开为非空的 LineString 分量。"""
    if geom.geom_type == "LineString":
        return [geom] if geom.length > 0.0 else []
    return [part for part in geom.geoms if part.geom_type == "LineString" and part.length > 0.0]


def validate_lines(lines: list[tuple[str, BaseGeometry]], overlap_tol_rel: float) -> None:
    """整批河道线的请求级校验：自交已在解析时拒绝，这里查线间内部重叠。

    任一失败抛 GeometryError，由端点整次拒绝（不返回任何部分结果）。
    """
    components: list[tuple[str, BaseGeometry]] = []
    for river_id, geom in lines:
        for part in _line_components(geom):
            components.append((river_id, part))

    # 容差按较短线长度缩放，吸收共线浮点噪音，同时仍能识别实质性重叠
    for i in range(len(components)):
        id_a, geom_a = components[i]
        for j in range(i + 1, len(components)):
            id_b, geom_b = components[j]
            tol = overlap_tol_rel * min(geom_a.length, geom_b.length)
            inter = geom_a.intersection(geom_b)
            if not inter.is_empty and inter.length > tol:
                raise GeometryError(
                    f"河道 {id_a} 与 {id_b} 存在重叠线段，重叠长度 {inter.length:g}，整次请求拒绝"
                )


def validate_submission(
    watershed: BaseGeometry,
    grids: list[tuple[str, BaseGeometry]],
    area_tol_rel: float,
) -> list[dict[str, Any]]:
    """校验网格互不重叠且恰好覆盖流域，返回问题清单（空清单代表通过）。"""
    issues: list[dict[str, Any]] = []
    area_tol = max(watershed.area * area_tol_rel, 0.0)

    # 1) 两两检查内部重叠（相切不算重叠）
    for i in range(len(grids)):
        id_a, geom_a = grids[i]
        for j in range(i + 1, len(grids)):
            id_b, geom_b = grids[j]
            inter = geom_a.intersection(geom_b)
            if not inter.is_empty and inter.area > area_tol:
                issues.append(
                    _issue(
                        "overlap",
                        inter,
                        f"网格 {id_a} 与 {id_b} 内部重叠，重叠面积 {inter.area:g}",
                    )
                )

    # 2) 合并后与流域做对称差，区分缺口与越界
    union = unary_union([geom for _, geom in grids])
    sym_diff = union.symmetric_difference(watershed)
    gap = sym_diff.intersection(watershed)  # 流域内未被网格覆盖的部分
    excess = sym_diff.difference(watershed)  # 网格超出流域的部分

    if not gap.is_empty and gap.area > area_tol:
        parts = list(gap.geoms) if gap.geom_type == "MultiPolygon" else [gap]
        parts = [p for p in parts if p.area > area_tol]
        for part in parts[:MAX_ISSUE_PARTS]:
            issues.append(_issue("gap", part, "网格未覆盖流域，存在缺口"))
        if len(parts) > MAX_ISSUE_PARTS:
            issues.append(
                {
                    "type": "gap",
                    "location": _bbox_dict(gap),
                    "representative_point": list(
                        gap.representative_point().coords[0]
                    ),
                    "area": gap.area,
                    "message": f"缺口过多，共 {len(parts)} 处，仅报告前 {MAX_ISSUE_PARTS} 处",
                }
            )

    if not excess.is_empty and excess.area > area_tol:
        parts = (
            list(excess.geoms)
            if excess.geom_type == "MultiPolygon"
            else [excess]
        )
        parts = [p for p in parts if p.area > area_tol]
        for part in parts[:MAX_ISSUE_PARTS]:
            issues.append(_issue("excess", part, "网格超出流域边界"))
        if len(parts) > MAX_ISSUE_PARTS:
            issues.append(
                {
                    "type": "excess",
                    "location": _bbox_dict(excess),
                    "representative_point": list(
                        excess.representative_point().coords[0]
                    ),
                    "area": excess.area,
                    "message": f"越界块过多，共 {len(parts)} 处，仅报告前 {MAX_ISSUE_PARTS} 处",
                }
            )

    return issues


def _make_point(lon: Any, lat: Any) -> Point:
    try:
        point = Point(float(lon), float(lat))
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"观测点坐标必须是数字，收到 {lon!r}, {lat!r}") from exc
    if not point.is_valid:
        raise GeometryError("观测点不是合法的点几何")
    return point


def classify_point(
    lon: Any,
    lat: Any,
    watershed: BaseGeometry,
    grids: list[tuple[str, BaseGeometry]],
    eps: float,
) -> dict[str, Any]:
    """判定单个观测点：内部点落唯一网格；边界点列出相接网格待判；域外返回域外。"""
    point = _make_point(lon, lat)

    # 先用空间索引找出外接框相交的候选网格，再精确判定
    tree = STRtree([geom for _, geom in grids])
    candidate_idx = list(tree.query(point))
    inside_ids: list[str] = []
    touching: list[str] = []
    for idx in candidate_idx:
        grid_id, geom = grids[int(idx)]
        # contains 不含边界；touches 表示交点只落在边界上
        if geom.contains(point):
            inside_ids.append(grid_id)
        elif geom.touches(point):
            touching.append(grid_id)

    # 已发布版本保证网格互不重叠；正常情况下内部点唯一，异常数据则转待判
    if len(inside_ids) == 1 and not touching:
        grid_id = inside_ids[0]
        return {"status": "inside", "grid_id": grid_id, "grid_ids": [grid_id]}
    if inside_ids or touching:
        pending = sorted(set(inside_ids) | set(touching))
        return {"status": "boundary", "grid_id": None, "grid_ids": pending}

    # 浮点兜底：点在流域边界极近邻内（且流域本身包含/相接该点时）按待判处理
    if watershed.dwithin(point, eps):
        near = sorted(
            grid_id
            for grid_id, geom in grids
            if geom.dwithin(point, eps)
        )
        if near:
            return {"status": "boundary", "grid_id": None, "grid_ids": near}

    return {"status": "outside", "grid_id": None, "grid_ids": []}


def _lineal_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """从任意叠加结果中提取有正长度的 LineString 分量（忽略 Point/MultiPoint）。"""
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom] if geom.length > 0.0 else []
    if geom.geom_type == "MultiLineString":
        return [part for part in geom.geoms if part.length > 0.0]
    if geom.geom_type == "GeometryCollection":
        out: list[BaseGeometry] = []
        for part in geom.geoms:
            if part.geom_type == "LineString" and part.length > 0.0:
                out.append(part)
            elif part.geom_type == "MultiLineString":
                out.extend(p for p in part.geoms if p.length > 0.0)
        return out
    return []


def _line_endpoints(geom: BaseGeometry) -> list[Point]:
    """收集 LineString/MultiLineString/GeometryCollection 中各线段的两个端点。

    这些端点即线进出每个网格的边界穿越点（含网格角点与跨边点），
    用作点切割器可把覆盖内部分切成「每段只落在一个格内部」的原子段。
    """
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom.interpolate(0.0), geom.interpolate(1.0, normalized=True)]
    if hasattr(geom, "geoms"):
        out: list[Point] = []
        for part in geom.geoms:
            out.extend(_line_endpoints(part))
        return out
    return []


def split_line_by_grids(
    line: BaseGeometry,
    grids: list[tuple[str, BaseGeometry]],
    eps: float,
) -> dict[str, Any]:
    """把一条河道线按已发布网格切成三类互斥的结果。

    - inside：严格落在某网格内部的原子线段（不含沿网格边界的部分），按格归并；
    - pending：沿任一网格边界延伸的线段，附带所有相接网格编号，不计入格内；
    - outside：流域外 / 网格孔洞内的线段，不计入格内。

    实现要点：
    1. ``line ∩ 网格并集`` 取覆盖内部，``line − 网格并集`` 取域外，二者天然互斥；
    2. 用「线与每个网格相交结果的端点」组成点切割器，把覆盖内部切成原子段，
       点切割器与线只可能点接触，不会触发 GEOS 的重叠切割异常；
    3. 每个原子段在 1/4、1/2、3/4 处采样，多数表决唯一归属格（已发布网格内部
       互不重叠）；三个采样点都落不进任何格内部的段即贴边界段，单列待判。
    仅与边界点接触（零长度交点）不产生线段，故不漏计也不重计；
    三类长度之和等于原线长度（GEOS 精确叠加，浮点误差远小于 1e-8 相对容差）。
    """
    union = unary_union([geom for _, geom in grids])

    # 域外：网格覆盖面之外（含流域外与孔洞内）
    outside = _lineal_parts(line.difference(union))

    # 覆盖内部，并按所有网格穿越点切成原子段
    covered = _lineal_parts(line.intersection(union))
    cut_points: list[Point] = []
    for _, grid_geom in grids:
        for pt in _line_endpoints(line.intersection(grid_geom)):
            if not any(pt.equals(existing) for existing in cut_points):
                cut_points.append(pt)

    atoms: list[BaseGeometry] = []
    if cut_points:
        cutter = MultiPoint(cut_points)
        for part in covered:
            try:
                atoms.extend(_lineal_parts(shapely_split(part, cutter)))
            except ValueError:
                # 极端共线情况下退化为不切分，交给采样归属判别
                atoms.append(part)
    else:
        atoms = covered

    inside: dict[str, list[BaseGeometry]] = {}
    pending: list[dict[str, Any]] = []
    for atom in atoms:
        votes: dict[str, int] = {}
        for fraction in (0.5, 0.25, 0.75):
            sample = atom.interpolate(fraction, normalized=True)
            owners = [
                grid_id
                for grid_id, grid_geom in grids
                if grid_geom.contains(sample)
            ]
            if len(owners) == 1:
                votes[owners[0]] = votes.get(owners[0], 0) + 1
        if votes:
            owner = max(votes, key=votes.get)
            inside.setdefault(owner, []).append(atom)
        else:
            # 采样点均不在任何格内部 → 整段贴在网格边界上，单列待判
            touching_ids = sorted(
                grid_id
                for grid_id, grid_geom in grids
                if grid_geom.boundary.dwithin(atom, eps)
            )
            pending.append({"geometry": atom, "grid_ids": touching_ids})

    return {"inside": inside, "pending": pending, "outside": outside}
