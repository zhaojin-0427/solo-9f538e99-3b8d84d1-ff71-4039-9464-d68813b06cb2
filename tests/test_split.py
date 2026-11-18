"""/split 端点端到端测试：河道线按已发布网格切分。"""
from __future__ import annotations

import importlib
import threading

import pytest
from fastapi.testclient import TestClient

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    import app.config as config
    import app.db as dbmod
    import app.main as main
    importlib.reload(config)
    importlib.reload(dbmod)
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, main


def poly(coords):
    return {"type": "Polygon", "coordinates": [coords]}


def line(coords):
    return {"type": "LineString", "coordinates": coords}


def mline(lines):
    return {"type": "MultiLineString", "coordinates": lines}


# 0,0 → 4,4 流域，左右两个 2×4 网格，共享边 x=2
WATERSHED = poly([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
LEFT = poly([(0, 0), (2, 0), (2, 4), (0, 4), (0, 0)])
RIGHT = poly([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)])


def publish(c, expected=None):
    body = {
        "expected_version": expected,
        "watershed": WATERSHED,
        "grids": [
            {"grid_id": "L", "geometry": LEFT},
            {"grid_id": "R", "geometry": RIGHT},
        ],
    }
    r = c.post("/publish", json=body)
    assert r.status_code == 200, r.text


def split_body(rivers, version=None):
    return {"version": version, "rivers": rivers}


def river(rid, coords_or_geom):
    return {"river_id": rid, "geometry": coords_or_geom}


def summed_length(result):
    return (
        sum(p["length"] for p in result["inside"])
        + sum(p["length"] for p in result["pending"])
        + sum(p["length"] for p in result["outside"])
    )


# ---------- 基本切分与长度守恒 ----------

def test_split_diagonal_across_two_grids(client):
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("d", line([[0, 0], [4, 4]]))]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 1 and body["is_current"] is True
    res = body["results"][0]
    assert res["river_id"] == "d"
    by_grid = {p["grid_id"]: p for p in res["inside"]}
    assert set(by_grid) == {"L", "R"}
    assert by_grid["L"]["length"] == pytest.approx(2 * 2 ** 0.5)
    assert by_grid["R"]["length"] == pytest.approx(2 * 2 ** 0.5)
    # 每段都是合法 GeoJSON LineString
    assert by_grid["L"]["geometry"]["type"] == "LineString"
    assert res["pending"] == [] and res["outside"] == []
    assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)


def test_split_length_conservation_relative_error_under_1e_minus_8(client):
    c, _ = client
    publish(c)
    # 非理性浮点坐标 + 多次转折（x 单调递增保证不自交），压浮点误差
    geom = line([
        [0.123456789, 0.987654321],
        [1.3333333, 3.7777777],
        [2.5555555, 1.1111111],
        [3.7777777, 2.9999999],
    ])
    r = c.post("/split", json=split_body([river("f", geom)]))
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    rel_err = abs(summed_length(res) - res["source_length"]) / res["source_length"]
    assert rel_err <= 1e-8


def test_split_multilinestring(client):
    c, _ = client
    publish(c)
    geom = mline([[[0, 1], [4, 1]], [[-1, 3], [5, 3]]])
    r = c.post("/split", json=split_body([river("m", geom)]))
    assert r.status_code == 200, r.text
    res = r.json()["results"][0]
    # 每条分量都被切成 L/R 两段，域外各 1
    assert len(res["inside"]) == 4
    assert len(res["outside"]) == 2
    assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)


# ---------- 边界：待判、点接触、不漏不重 ----------

def test_split_line_along_shared_boundary_is_pending(client):
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("e", line([[2, 0], [2, 4]]))]))
    res = r.json()["results"][0]
    assert res["inside"] == []
    assert len(res["pending"]) == 1
    p = res["pending"][0]
    assert p["length"] == pytest.approx(4.0)
    assert sorted(p["grid_ids"]) == ["L", "R"]
    assert res["outside"] == []
    assert summed_length(res) == pytest.approx(4.0, rel=1e-12)


def test_split_line_along_outer_boundary_is_pending(client):
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("top", line([[0, 4], [4, 4]]))]))
    res = r.json()["results"][0]
    assert res["inside"] == [] and res["outside"] == []
    assert sum(p["length"] for p in res["pending"]) == pytest.approx(4.0)
    for p in res["pending"]:
        assert sorted(p["grid_ids"]) == ["L", "R"]


def test_split_point_contact_only_not_counted(client):
    """仅与网格角点接触：无任何线段产生，长度全部落在域外，不重计。"""
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("touch", line([[-1, -1], [0, 0]]))]))
    res = r.json()["results"][0]
    assert res["inside"] == [] and res["pending"] == []
    assert len(res["outside"]) == 1
    assert res["outside"][0]["length"] == pytest.approx(2 ** 0.5)


def test_split_mixed_inside_pending_outside(client):
    c, _ = client
    publish(c)
    # 格内 → 沿共享边 → 格内
    geom = line([[0, 1], [2, 1], [2, 3], [4, 3]])
    r = c.post("/split", json=split_body([river("mix", geom)]))
    res = r.json()["results"][0]
    gids = sorted(p["grid_id"] for p in res["inside"])
    assert gids == ["L", "R"]
    assert len(res["pending"]) == 1 and res["pending"][0]["length"] == pytest.approx(2.0)
    assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)


def test_split_reenter_same_grid_not_lost(client):
    """河道多次进出同一格：每次进入都要各计一段，不漏不重。"""
    c, _ = client
    publish(c)
    # 在 L 格（x<2）内进出两次：格内两段各长 2；域外三段长度 1 + 2 + 1
    geom = line([[-1, 0.5], [1, 0.5], [1, 1.5], [-1, 1.5],
                 [-1, 2.5], [1, 2.5]])
    r = c.post("/split", json=split_body([river("zig", geom)]))
    res = r.json()["results"][0]
    l_pieces = [p for p in res["inside"] if p["grid_id"] == "L"]
    assert len(l_pieces) == 2
    assert sum(p["length"] for p in l_pieces) == pytest.approx(4.0)
    assert sum(p["length"] for p in res["outside"]) == pytest.approx(4.0)
    assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)


def test_split_diagonal_through_grid_corner_not_double_counted(tmp_path, monkeypatch):
    """回归：斜线穿过网格角点时，角点两侧线段必须各归一格，绝不重计。

    2×2 网格共享角点 (2,2)，线 (1,1)→(3,3) 穿过该角点：
    左下格与右上格各得一段，长度各半，无待判、无域外，总长守恒。
    """
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "corner.db"))
    import app.config as config
    import app.db as dbmod
    import app.main as main
    importlib.reload(config)
    importlib.reload(dbmod)
    importlib.reload(main)
    with TestClient(main.app) as c:
        ws = poly([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
        grids = [
            ("BL", poly([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])),
            ("BR", poly([(2, 0), (4, 0), (4, 2), (2, 2), (2, 0)])),
            ("TL", poly([(0, 2), (2, 2), (2, 4), (0, 4), (0, 2)])),
            ("TR", poly([(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)])),
        ]
        r = c.post("/publish", json={
            "expected_version": None, "watershed": ws,
            "grids": [{"grid_id": gid, "geometry": g} for gid, g in grids],
        })
        assert r.status_code == 200, r.text
        r = c.post("/split", json=split_body([river("x", line([[1, 1], [3, 3]]))]))
        res = r.json()["results"][0]
        owners = sorted(p["grid_id"] for p in res["inside"])
        assert owners == ["BL", "TR"]
        assert sum(p["length"] for p in res["inside"]) == pytest.approx(2 * 2 ** 0.5)
        assert res["pending"] == [] and res["outside"] == []
        assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)


def test_split_crosses_many_grid_corners_length_conserved(tmp_path, monkeypatch):
    """回归：一条线连续穿过多个网格角点（4×3 网格），长度守恒且无重计。"""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "many.db"))
    import app.config as config
    import app.db as dbmod
    import app.main as main
    importlib.reload(config)
    importlib.reload(dbmod)
    importlib.reload(main)
    with TestClient(main.app) as c:
        ws = poly([(0, 0), (8, 0), (8, 6), (0, 6), (0, 0)])
        grids = []
        for i in range(4):
            for j in range(3):
                gid = f"g{i}{j}"
                g = poly([(i * 2, j * 2), (i * 2 + 2, j * 2),
                          (i * 2 + 2, j * 2 + 2), (i * 2, j * 2 + 2),
                          (i * 2, j * 2)])
                grids.append({"grid_id": gid, "geometry": g})
        r = c.post("/publish", json={
            "expected_version": None, "watershed": ws, "grids": grids})
        assert r.status_code == 200, r.text
        # 斜率 1，依次穿过 (2,2)、(4,4) 两个内部格点
        r = c.post("/split", json=split_body([river("k", line([[1, 1], [5, 5]]))]))
        res = r.json()["results"][0]
        # 每段长度计入且仅计入一次：格内总长 = 原线长（全程在流域内、不贴边）
        assert sum(p["length"] for p in res["inside"]) == pytest.approx(
            res["source_length"], rel=1e-12)
        assert res["pending"] == [] and res["outside"] == []
        assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)


# ---------- 域外与孔洞 ----------

def test_split_fully_outside(client):
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("out", line([[20, 20], [21, 21]]))]))
    res = r.json()["results"][0]
    assert res["inside"] == [] and res["pending"] == []
    assert sum(p["length"] for p in res["outside"]) == pytest.approx(2 ** 0.5)


def test_split_hole_segment_is_outside(tmp_path, monkeypatch):
    """带孔洞的流域：孔洞内线段单列域外，洞边界线段待判，长度守恒。"""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "hole.db"))
    import app.config as config
    import app.db as dbmod
    import app.main as main
    importlib.reload(config)
    importlib.reload(dbmod)
    importlib.reload(main)
    with TestClient(main.app) as c:
        # 孔洞 x∈[7,9]、y∈[4,6]，整体落在右半区；左右网格共享边 x=5，
        # 洞环只属于右网格，避免环在共享边上自触
        watershed = {"type": "Polygon", "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[7, 4], [9, 4], [9, 6], [7, 6], [7, 4]]]}
        left = {"type": "Polygon", "coordinates": [
            [[0, 0], [5, 0], [5, 10], [0, 10], [0, 0]]]}
        right = {"type": "Polygon", "coordinates": [
            [[5, 0], [10, 0], [10, 10], [5, 10], [5, 0]],
            [[7, 4], [9, 4], [9, 6], [7, 6], [7, 4]]]}
        r = c.post("/publish", json={
            "expected_version": None,
            "watershed": watershed,
            "grids": [{"grid_id": "WL", "geometry": left},
                      {"grid_id": "WR", "geometry": right}],
        })
        assert r.status_code == 200, r.text

        # 横线 y=5 从左网格穿到右网格并横穿孔洞：
        # 格内 0→7 共 7（含共享边待判），孔洞内 7→9 共 2 为域外，9→10 格内 1
        r = c.post("/split", json=split_body([river("h", line([[2, 5], [10, 5]]))]))
        res = r.json()["results"][0]
        assert sum(p["length"] for p in res["outside"]) == pytest.approx(2.0)
        # 格内 = 总长 8 − 域外 2 − 共享边待判 0（共享边交点只是点，零长度）
        assert sum(p["length"] for p in res["inside"]) == pytest.approx(6.0)
        assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)

        # 完全落在孔洞内的线：整体域外
        r = c.post("/split", json=split_body([river("inhole", line([[8, 4.5], [8, 5.5]]))]))
        res = r.json()["results"][0]
        assert res["inside"] == [] and res["pending"] == []
        assert sum(p["length"] for p in res["outside"]) == pytest.approx(1.0)

        # 沿孔洞底边 y=4 的线：待判且相接网格为右网格
        r = c.post("/split", json=split_body([river("holeedge", line([[7, 4], [9, 4]]))]))
        res = r.json()["results"][0]
        assert sum(p["length"] for p in res["pending"]) == pytest.approx(2.0)
        for p in res["pending"]:
            assert p["grid_ids"] == ["WR"]
        assert summed_length(res) == pytest.approx(res["source_length"], rel=1e-12)


# ---------- 输入校验：整次拒绝，无部分结果 ----------

def test_split_river_count_limits(client):
    c, _ = client
    publish(c)
    empty = {"rivers": []}
    assert c.post("/split", json=empty).status_code == 422
    many = {"rivers": [river(f"r{i}", line([[0, 0], [1, 1]])) for i in range(11)]}
    assert c.post("/split", json=many).status_code == 422


def test_split_duplicate_river_id_rejected(client):
    c, _ = client
    publish(c)
    body = split_body([
        river("x", line([[0, 0], [1, 1]])),
        river("x", line([[2, 2], [3, 3]])),
    ])
    r = c.post("/split", json=body)
    assert r.status_code == 422
    assert r.json()["error"] == "DUPLICATE_RIVER_ID"


def test_split_wrong_geometry_type_rejected(client):
    c, _ = client
    publish(c)
    body = split_body([river("p", poly([(0, 0), (1, 0), (1, 1), (0, 0)]))])
    r = c.post("/split", json=body)
    assert r.status_code == 422
    assert r.json()["error"] == "INVALID_GEOMETRY"


def test_split_zero_length_rejected(client):
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("z", line([[1, 1], [1, 1]]))]))
    assert r.status_code == 422
    assert r.json()["error"] == "INVALID_GEOMETRY"


def test_split_non_finite_coordinates_rejected(client):
    c, _ = client
    publish(c)
    # 用原始报文模拟客户端提交非有限坐标（严格 JSON 下这些本就非法，
    # 服务端无论如何都必须识别并拒绝，而不是崩成 500）
    for token in ("Infinity", "-Infinity", "NaN"):
        raw = (
            '{"rivers": [{"river_id": "bad", '
            '"geometry": {"type": "LineString", '
            f'"coordinates": [[0, 0], [{token}, 1]]}}}}]}}'
        )
        r = c.post("/split", content=raw, headers={"Content-Type": "application/json"})
        assert r.status_code == 422, token
        assert r.json()["error"] == "INVALID_GEOMETRY", token


def test_split_self_intersection_rejected(client):
    c, _ = client
    publish(c)
    # 折线自交（蝴蝶结）
    geom = line([[0, 0], [2, 2], [2, 0], [0, 2]])
    r = c.post("/split", json=split_body([river("cross", geom)]))
    assert r.status_code == 422
    assert r.json()["error"] == "INVALID_GEOMETRY"


def test_split_overlapping_rivers_rejected(client):
    c, _ = client
    publish(c)
    body = split_body([
        river("a", line([[0, 1], [4, 1]])),
        river("b", line([[1, 1], [3, 1]])),
    ])
    r = c.post("/split", json=body)
    assert r.status_code == 422
    assert r.json()["error"] == "INVALID_GEOMETRY"


def test_split_multilinestring_self_overlap_rejected(client):
    c, _ = client
    publish(c)
    geom = mline([[[0, 0], [4, 0]], [[1, 0], [3, 0]]])
    r = c.post("/split", json=split_body([river("dup", geom)]))
    assert r.status_code == 422


def test_split_invalid_input_returns_no_partial_results(client):
    """整次拒绝：响应里不含任何 results 字段。"""
    c, _ = client
    publish(c)
    body = split_body([
        river("ok", line([[0, 1], [1, 1]])),
        river("bad", line([[5, 5], [5, 5]])),  # 零长度
    ])
    r = c.post("/split", json=body)
    assert r.status_code == 422
    assert "results" not in r.json()


# ---------- 版号：缺省当前版、旧版、不存在、未发布 ----------

def test_split_defaults_to_current_version(client):
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("d", line([[0, 0], [4, 4]]))]))
    assert r.status_code == 200
    assert r.json()["version"] == 1 and r.json()["is_current"] is True


def test_split_before_publish_is_404(client):
    c, _ = client
    r = c.post("/split", json=split_body([river("d", line([[0, 0], [1, 1]]))]))
    assert r.status_code == 404
    assert r.json()["error"] == "NO_PUBLISHED_VERSION"


def test_split_version_not_found(client):
    c, _ = client
    publish(c)
    r = c.post("/split", json=split_body([river("d", line([[0, 0], [1, 1]]))], version=9))
    assert r.status_code == 404
    assert r.json()["error"] == "VERSION_NOT_FOUND"


def test_split_old_version_stable_and_recalc_consistent(client):
    """指定旧版重算结果一致，且与新版结果不同。"""
    c, _ = client
    publish(c)
    # 第 2 版：上下两个网格
    top = poly([(0, 2), (4, 2), (4, 4), (0, 4), (0, 2)])
    bottom = poly([(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)])
    r = c.post("/publish", json={
        "expected_version": 1,
        "watershed": WATERSHED,
        "grids": [{"grid_id": "B", "geometry": bottom},
                  {"grid_id": "T", "geometry": top}],
    })
    assert r.status_code == 200
    body = split_body([river("d", line([[1, 1], [3, 3]]))], version=1)
    first = c.post("/split", json=body).json()
    second = c.post("/split", json=body).json()
    assert first == second  # 同版重算完全一致
    gids_v1 = sorted(p["grid_id"] for p in first["results"][0]["inside"])
    assert gids_v1 == ["L", "R"]
    cur = c.post("/split", json=split_body([river("d", line([[1, 1], [3, 3]]))])).json()
    assert cur["version"] == 2 and cur["is_current"] is True
    gids_v2 = sorted(p["grid_id"] for p in cur["results"][0]["inside"])
    assert gids_v2 == ["B", "T"]


def test_split_snapshot_isolated_from_concurrent_publish(client):
    """计算期间网格改版：响应版号与结果必须取自同一快照，不错配。"""
    c, main = client
    publish(c)
    top = poly([(0, 2), (4, 2), (4, 4), (0, 4), (0, 2)])
    bottom = poly([(0, 0), (4, 0), (4, 2), (0, 2), (0, 0)])
    from shapely.geometry import shape

    def publisher():
        # 与切分并发发布第 2 版；若版号冲突被拒也不影响切分自洽性断言
        try:
            main.db.publish(1, main.db.load_snapshot(1)[0],
                            [("B", shape(bottom)), ("T", shape(top))])
        except Exception:
            pass

    t = threading.Thread(target=publisher)
    t.start()
    # 切分固定版号 1，结果网格编号只能来自第 1 版（L/R），绝不能出现 B/T
    for _ in range(5):
        resp = c.post("/split", json=split_body(
            [river("d", line([[0, 0], [4, 4]]))], version=1)).json()
        assert resp["version"] == 1
        gids = {p["grid_id"] for p in resp["results"][0]["inside"]}
        assert gids <= {"L", "R"}
    t.join()


# ---------- 原功能回归 ----------

def test_publish_and_query_still_work(client):
    c, _ = client
    publish(c)
    q = c.post("/query", json={"points": [
        {"point_id": "in", "lon": 1.0, "lat": 2.0},
        {"point_id": "edge", "lon": 2.0, "lat": 2.0},
        {"point_id": "out", "lon": 9.0, "lat": 9.0},
    ]})
    assert q.status_code == 200
    by_id = {x["point_id"]: x for x in q.json()["results"]}
    assert by_id["in"]["status"] == "inside" and by_id["in"]["grid_id"] == "L"
    assert by_id["edge"]["status"] == "boundary"
    assert by_id["out"]["status"] == "outside"
    assert c.get("/health").status_code == 200
    assert c.get("/versions").status_code == 200
    assert c.get("/docs").status_code == 200
