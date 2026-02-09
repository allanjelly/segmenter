from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable, Sequence

from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPointLocator, vtkPolyData
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkIdList, vtkPoints
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkFiltersModeling import vtkDijkstraGraphGeodesicPath
from vtkmodules.vtkFiltersCore import vtkClipPolyData
from vtkmodules.vtkCommonDataModel import vtkPlane


@dataclass(frozen=True)
class GeodesicResult:
    point_ids: list[int]
    polyline: vtkPolyData


def build_point_locator(surface: vtkPolyData) -> vtkPointLocator:
    locator = vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    return locator


def closest_point_id(locator: vtkPointLocator, point: Iterable[float]) -> int:
    x, y, z = point
    return int(locator.FindClosestPoint(x, y, z))


def compute_geodesic(surface: vtkPolyData, start_id: int, end_id: int) -> GeodesicResult:
    dijkstra = vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(surface)
    dijkstra.SetStartVertex(start_id)
    dijkstra.SetEndVertex(end_id)
    dijkstra.Update()

    polyline = dijkstra.GetOutput()
    point_ids = [int(dijkstra.GetIdList().GetId(i)) for i in range(dijkstra.GetIdList().GetNumberOfIds())]
    return GeodesicResult(point_ids=point_ids, polyline=polyline)


def _build_point_adjacency(surface: vtkPolyData) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(surface.GetNumberOfPoints())]

    def add_cell(ids: Iterable[int]) -> None:
        ids_list = list(ids)
        count = len(ids_list)
        for i in range(count):
            a = ids_list[i]
            for j in range(i + 1, count):
                b = ids_list[j]
                adjacency[a].add(b)
                adjacency[b].add(a)

    polys = surface.GetPolys()
    polys.InitTraversal()
    id_list = vtkIdList()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() >= 2:
            add_cell(id_list.GetId(i) for i in range(id_list.GetNumberOfIds()))

    strips = surface.GetStrips()
    strips.InitTraversal()
    id_list.Reset()
    while strips.GetNextCell(id_list):
        if id_list.GetNumberOfIds() >= 2:
            add_cell(id_list.GetId(i) for i in range(id_list.GetNumberOfIds()))

    return adjacency


def compute_anisotropic_geodesic(
    surface: vtkPolyData,
    start_id: int,
    end_id: int,
    normal: Sequence[float],
    penalty_strength: float,
) -> GeodesicResult | None:
    if surface.GetNumberOfPoints() == 0:
        return None
    if start_id == end_id:
        return None

    adjacency = _build_point_adjacency(surface)
    unit_normal = normalize(normal)
    if unit_normal == (0.0, 0.0, 0.0):
        unit_normal = (0.0, 0.0, 0.0)

    num_points = surface.GetNumberOfPoints()
    dist = [float("inf")] * num_points
    prev = [-1] * num_points
    dist[start_id] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start_id)]

    while heap:
        current_dist, current = heapq.heappop(heap)
        if current_dist != dist[current]:
            continue
        if current == end_id:
            break
        px, py, pz = surface.GetPoint(current)
        for neighbor in adjacency[current]:
            nx, ny, nz = surface.GetPoint(neighbor)
            dx = nx - px
            dy = ny - py
            dz = nz - pz
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length == 0.0:
                continue
            if unit_normal == (0.0, 0.0, 0.0):
                penalty = 1.0
            else:
                dir_dot = abs((dx / length) * unit_normal[0] + (dy / length) * unit_normal[1] + (dz / length) * unit_normal[2])
                penalty = 1.0 + penalty_strength * dir_dot
            weight = length * penalty
            next_dist = current_dist + weight
            if next_dist < dist[neighbor]:
                dist[neighbor] = next_dist
                prev[neighbor] = current
                heapq.heappush(heap, (next_dist, neighbor))

    if prev[end_id] == -1:
        return None

    path_ids: list[int] = []
    current = end_id
    while current != -1:
        path_ids.append(current)
        if current == start_id:
            break
        current = prev[current]
    if path_ids[-1] != start_id:
        return None
    path_ids.reverse()

    vtk_points = vtkPoints()
    lines = vtkCellArray()
    lines.InsertNextCell(len(path_ids))
    for idx, point_id in enumerate(path_ids):
        vtk_points.InsertNextPoint(surface.GetPoint(point_id))
        lines.InsertCellPoint(idx)

    polyline = vtkPolyData()
    polyline.SetPoints(vtk_points)
    polyline.SetLines(lines)
    return GeodesicResult(point_ids=path_ids, polyline=polyline)


def compute_weighted_geodesic(
    surface: vtkPolyData,
    start_id: int,
    end_id: int,
    weights: Sequence[float],
) -> GeodesicResult:
    if surface.GetNumberOfPoints() != len(weights):
        raise ValueError("Weight count must match number of points")

    scalars = vtkFloatArray()
    scalars.SetName("geodesic_cost")
    scalars.SetNumberOfComponents(1)
    scalars.SetNumberOfTuples(surface.GetNumberOfPoints())
    for i, value in enumerate(weights):
        scalars.SetValue(i, float(value))

    point_data = surface.GetPointData()
    previous_scalars = point_data.GetScalars()
    point_data.SetScalars(scalars)

    dijkstra = vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(surface)
    dijkstra.SetStartVertex(start_id)
    dijkstra.SetEndVertex(end_id)
    dijkstra.SetUseScalarWeights(True)
    dijkstra.Update()

    polyline = dijkstra.GetOutput()
    point_ids = [int(dijkstra.GetIdList().GetId(i)) for i in range(dijkstra.GetIdList().GetNumberOfIds())]

    point_data.SetScalars(previous_scalars)
    return GeodesicResult(point_ids=point_ids, polyline=polyline)


def compute_geodesic_via(
    surface: vtkPolyData,
    start_id: int,
    via_id: int,
    end_id: int,
) -> GeodesicResult:
    first = compute_geodesic(surface, start_id, via_id)
    second = compute_geodesic(surface, via_id, end_id)

    append = vtkAppendPolyData()
    append.AddInputData(first.polyline)
    append.AddInputData(second.polyline)
    append.Update()

    point_ids = first.point_ids + second.point_ids[1:]
    return GeodesicResult(point_ids=point_ids, polyline=append.GetOutput())


def plane_normal(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> tuple[float, float, float]:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    nx = ab[1] * ac[2] - ab[2] * ac[1]
    ny = ab[2] * ac[0] - ab[0] * ac[2]
    nz = ab[0] * ac[1] - ab[1] * ac[0]
    return (nx, ny, nz)


def normalize(vec: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = vec
    mag = (x * x + y * y + z * z) ** 0.5
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return (x / mag, y / mag, z / mag)


def centroid(points: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sz = sum(p[2] for p in points)
    count = float(len(points))
    return (sx / count, sy / count, sz / count)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Sequence[float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)




def cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def plane_side(
    plane_point: Sequence[float],
    normal: Sequence[float],
    point: Sequence[float],
) -> int:
    value = dot(normal, sub(point, plane_point))
    return 1 if value >= 0 else -1


def polyline_midpoint(polyline: vtkPolyData) -> tuple[float, float, float]:
    points = polyline.GetPoints()
    if points is None or points.GetNumberOfPoints() == 0:
        return (0.0, 0.0, 0.0)
    mid_index = points.GetNumberOfPoints() // 2
    return points.GetPoint(mid_index)


def compute_clipped_geodesic(
    surface: vtkPolyData,
    start_point: Sequence[float],
    end_point: Sequence[float],
    plane_point: Sequence[float],
    normal: Sequence[float],
    keep_side: int,
) -> GeodesicResult | None:
    plane = vtkPlane()
    plane.SetOrigin(plane_point)
    plane_normal = normal if keep_side > 0 else (-normal[0], -normal[1], -normal[2])
    plane.SetNormal(plane_normal)

    clipper = vtkClipPolyData()
    clipper.SetInputData(surface)
    clipper.SetClipFunction(plane)
    clipper.SetInsideOut(False)
    clipper.Update()

    clipped = clipper.GetOutput()
    if clipped is None or clipped.GetNumberOfPoints() == 0:
        return None

    clipped_locator = build_point_locator(clipped)
    start_id = closest_point_id(clipped_locator, start_point)
    end_id = closest_point_id(clipped_locator, end_point)
    result = compute_geodesic(clipped, start_id, end_id)
    if result.polyline is None or result.polyline.GetNumberOfPoints() == 0:
        return None
    return result


def create_pair_geodesics(
    surface: vtkPolyData,
    locator: vtkPointLocator,
    landmarks: dict[str, Sequence[float]],
    start_key: str,
    end_key: str,
    plane_keys: tuple[str, str, str],
    anterior_ref_key: str = "E",
    plane_origin_key: str = "A",
) -> tuple[str, GeodesicResult, str, GeodesicResult | None]:
    plane_a = landmarks[plane_keys[0]]
    plane_b = landmarks[plane_keys[1]]
    plane_c = landmarks[plane_keys[2]]
    plane_origin = landmarks[plane_origin_key]

    start_point = landmarks[start_key]
    end_point = landmarks[end_key]
    ref_point = landmarks[anterior_ref_key]

    normal = normalize(cross(sub(plane_b, plane_a), sub(plane_c, plane_a)))
    ref_side = plane_side(plane_origin, normal, ref_point)

    start_id = closest_point_id(locator, start_point)
    end_id = closest_point_id(locator, end_point)
    primary = compute_geodesic(surface, start_id, end_id)

    midpoint = polyline_midpoint(primary.polyline)
    mid_side = plane_side(plane_origin, normal, midpoint)

    if mid_side == ref_side:
        primary_key = f"{start_key}{end_key}_anterior"
        alternate_key = f"{start_key}{end_key}_posterior"
        opposite_side = -ref_side
    else:
        primary_key = f"{start_key}{end_key}_posterior"
        alternate_key = f"{start_key}{end_key}_anterior"
        opposite_side = ref_side

    alternate = compute_clipped_geodesic(
        surface,
        start_point,
        end_point,
        plane_origin,
        normal,
        opposite_side,
    )
    return primary_key, primary, alternate_key, alternate


def create_simple_geodesic(
    surface: vtkPolyData,
    locator: vtkPointLocator,
    landmarks: dict[str, Sequence[float]],
    start_key: str,
    end_key: str,
) -> GeodesicResult | None:
    start_point = landmarks[start_key]
    end_point = landmarks[end_key]
    start_id = closest_point_id(locator, start_point)
    end_id = closest_point_id(locator, end_point)
    result = compute_geodesic(surface, start_id, end_id)
    if result.polyline is None or result.polyline.GetNumberOfPoints() == 0:
        return None
    return result


def create_anisotropic_geodesic(
    surface: vtkPolyData,
    locator: vtkPointLocator,
    landmarks: dict[str, Sequence[float]],
    start_key: str,
    end_key: str,
    normal: Sequence[float],
    penalty_strength: float,
) -> GeodesicResult | None:
    start_point = landmarks[start_key]
    end_point = landmarks[end_key]
    start_id = closest_point_id(locator, start_point)
    end_id = closest_point_id(locator, end_point)
    result = compute_anisotropic_geodesic(surface, start_id, end_id, normal, penalty_strength)
    if result is None or result.polyline is None or result.polyline.GetNumberOfPoints() == 0:
        return None
    return result


def compute_ma_plane_normal(
    e: Sequence[float],
    f: Sequence[float],
    h: Sequence[float],
    i: Sequence[float],
) -> tuple[float, float, float] | None:
    plane = _best_fit_plane([e, f, h, i])
    if plane is None:
        return None
    _origin, normal = plane
    return normalize(normal)


def _best_fit_plane(points: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if len(points) < 3:
        return None
    vtk_pts = vtkPoints()
    for point in points:
        vtk_pts.InsertNextPoint(point)
    origin = [0.0, 0.0, 0.0]
    normal = [0.0, 0.0, 0.0]
    try:
        vtkPlane.ComputeBestFittingPlane(vtk_pts, origin, normal)
        best_normal = normalize(normal)
    except AttributeError:
        best_normal = normalize(cross(sub(points[1], points[0]), sub(points[2], points[0])))
        origin = list(centroid(points))
    if best_normal == (0.0, 0.0, 0.0):
        return None
    return (origin[0], origin[1], origin[2]), best_normal




