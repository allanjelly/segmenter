from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from vtkmodules.vtkCommonDataModel import vtkPointLocator, vtkPolyData
from vtkmodules.vtkCommonCore import vtkFloatArray
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


def bounds_diag(surface: vtkPolyData) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = surface.GetBounds()
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def distance_to_polyline(surface: vtkPolyData, polyline: vtkPolyData) -> list[float]:
    locator = vtkPointLocator()
    locator.SetDataSet(polyline)
    locator.BuildLocator()

    distances: list[float] = []
    for i in range(surface.GetNumberOfPoints()):
        point = surface.GetPoint(i)
        closest_id = locator.FindClosestPoint(point)
        closest_point = polyline.GetPoint(closest_id)
        dx = point[0] - closest_point[0]
        dy = point[1] - closest_point[1]
        dz = point[2] - closest_point[2]
        distances.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    return distances


def cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


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
