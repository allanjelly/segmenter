from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

from vtkmodules.vtkCommonCore import vtkIdList, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkPolyData

from geodesics import build_point_locator


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


def _find_non_boundary_seed(
    start_id: int,
    adjacency: list[set[int]],
    boundary_ids: set[int],
    blocked_ids: set[int] | None = None,
) -> int | None:
    if blocked_ids is None:
        blocked_ids = set()
    if start_id not in boundary_ids and start_id not in blocked_ids:
        return start_id

    visited = {start_id}
    queue: deque[int] = deque([start_id])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            if neighbor in boundary_ids or neighbor in blocked_ids:
                queue.append(neighbor)
                continue
            return neighbor
    return None


def _collect_component(
    seed_id: int,
    adjacency: list[set[int]],
    boundary_ids: set[int],
    blocked_ids: set[int] | None = None,
) -> set[int]:
    if blocked_ids is None:
        blocked_ids = set()

    visited: set[int] = set()
    queue: deque[int] = deque([seed_id])
    visited.add(seed_id)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            if neighbor in boundary_ids or neighbor in blocked_ids:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def _collect_boundary_ids(
    locator,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
    boundary_keys: Sequence[str],
) -> set[int] | None:
    boundary_ids: set[int] = set()
    for key in boundary_keys:
        polyline = geodesic_lines.get(key)
        if polyline is None:
            return None
        points = polyline.GetPoints()
        if points is None:
            return None
        lines = polyline.GetLines()
        if lines is None:
            return None
        lines.InitTraversal()
        id_list = vtkIdList()
        while lines.GetNextCell(id_list):
            if id_list.GetNumberOfIds() < 2:
                continue
            for i in range(1, id_list.GetNumberOfIds()):
                p0 = points.GetPoint(id_list.GetId(i - 1))
                p1 = points.GetPoint(id_list.GetId(i))
                boundary_ids.add(locator.FindClosestPoint(p0))
                boundary_ids.add(locator.FindClosestPoint(p1))
                mid1 = (
                    p0[0] + (p1[0] - p0[0]) * (1.0 / 3.0),
                    p0[1] + (p1[1] - p0[1]) * (1.0 / 3.0),
                    p0[2] + (p1[2] - p0[2]) * (1.0 / 3.0),
                )
                mid2 = (
                    p0[0] + (p1[0] - p0[0]) * (2.0 / 3.0),
                    p0[1] + (p1[1] - p0[1]) * (2.0 / 3.0),
                    p0[2] + (p1[2] - p0[2]) * (2.0 / 3.0),
                )
                boundary_ids.add(locator.FindClosestPoint(mid1))
                boundary_ids.add(locator.FindClosestPoint(mid2))

    return boundary_ids


def _collect_segment_component(
    surface: vtkPolyData,
    adjacency: list[set[int]],
    locator,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
    boundary_keys: Sequence[str],
    opposite_key: str,
    seed_key: str,
    blocked_ids: set[int] | None = None,
    seed_point: Sequence[float] | None = None,
) -> set[int] | None:
    boundary_ids = _collect_boundary_ids(locator, landmarks, geodesic_lines, boundary_keys)
    if boundary_ids is None:
        return None

    opposite_id = locator.FindClosestPoint(landmarks[opposite_key])
    seed_source = seed_point if seed_point is not None else landmarks[seed_key]
    seed_id = locator.FindClosestPoint(seed_source)

    opposite_seed = _find_non_boundary_seed(opposite_id, adjacency, boundary_ids)
    if opposite_seed is None:
        return None
    wrong_component = _collect_component(opposite_seed, adjacency, boundary_ids)

    seed = _find_non_boundary_seed(
        seed_id,
        adjacency,
        boundary_ids,
        blocked_ids=wrong_component if blocked_ids is None else blocked_ids | wrong_component,
    )
    if seed is None:
        return None

    blocked = wrong_component
    if blocked_ids:
        blocked = blocked | blocked_ids

    return _collect_component(seed, adjacency, boundary_ids, blocked_ids=blocked)


def _diagnose_segment_failure(
    segment_id: int,
    adjacency: list[set[int]],
    locator,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
    boundary_keys: Sequence[str],
    opposite_key: str,
    seed_key: str,
    blocked_ids: set[int] | None = None,
    seed_point: Sequence[float] | None = None,
) -> str:
    boundary_ids = _collect_boundary_ids(locator, landmarks, geodesic_lines, boundary_keys)
    if boundary_ids is None:
        return f"Segment {segment_id} failed: missing boundary polyline"

    opposite_id = locator.FindClosestPoint(landmarks[opposite_key])
    seed_source = seed_point if seed_point is not None else landmarks[seed_key]
    seed_id = locator.FindClosestPoint(seed_source)

    opposite_seed = _find_non_boundary_seed(opposite_id, adjacency, boundary_ids)
    if opposite_seed is None:
        reason = "opposite on boundary" if opposite_id in boundary_ids else "opposite enclosed"
        return f"Segment {segment_id} failed: {reason} (boundary={len(boundary_ids)})"

    wrong_component = _collect_component(opposite_seed, adjacency, boundary_ids)

    seed_blocked = wrong_component
    if blocked_ids:
        seed_blocked = seed_blocked | blocked_ids

    seed = _find_non_boundary_seed(
        seed_id,
        adjacency,
        boundary_ids,
        blocked_ids=seed_blocked,
    )
    if seed is None:
        if seed_id in boundary_ids:
            reason = "seed on boundary"
        elif seed_id in seed_blocked:
            reason = "seed blocked"
        else:
            reason = "seed enclosed"
        return (
            f"Segment {segment_id} failed: "
            f"{reason} (boundary={len(boundary_ids)}, blocked={len(seed_blocked)})"
        )

    return f"Segment {segment_id} failed: unknown"


def _collect_failure_debug(
    adjacency: list[set[int]],
    locator,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
    boundary_keys: Sequence[str],
    opposite_key: str,
    seed_key: str,
    blocked_ids: set[int] | None = None,
    seed_point: Sequence[float] | None = None,
) -> dict[str, object] | None:
    boundary_ids = _collect_boundary_ids(locator, landmarks, geodesic_lines, boundary_keys)
    if boundary_ids is None:
        return None

    opposite_id = int(locator.FindClosestPoint(landmarks[opposite_key]))
    seed_source = seed_point if seed_point is not None else landmarks[seed_key]
    seed_id = int(locator.FindClosestPoint(seed_source))

    opposite_seed = _find_non_boundary_seed(opposite_id, adjacency, boundary_ids)
    if opposite_seed is None:
        return {
            "boundary_ids": list(boundary_ids),
            "boundary_count": len(boundary_ids),
            "blocked_count": None,
            "total_points": len(adjacency),
            "seed_id": seed_id,
            "opposite_id": opposite_id,
            "opposite_seed_id": None,
            "seed_candidate_id": None,
        }

    wrong_component = _collect_component(opposite_seed, adjacency, boundary_ids)
    seed_blocked = wrong_component
    if blocked_ids:
        seed_blocked = seed_blocked | blocked_ids

    seed = _find_non_boundary_seed(
        seed_id,
        adjacency,
        boundary_ids,
        blocked_ids=seed_blocked,
    )

    return {
        "boundary_ids": list(boundary_ids),
        "boundary_count": len(boundary_ids),
        "blocked_count": len(seed_blocked),
        "total_points": len(adjacency),
        "seed_id": seed_id,
        "opposite_id": opposite_id,
        "opposite_seed_id": opposite_seed,
        "seed_candidate_id": seed,
    }


def _seed_fallback_candidates(
    landmarks: dict[str, Sequence[float]],
    seed_key: str,
    opposite_key: str,
) -> list[tuple[float, float, float]]:
    if seed_key not in landmarks or opposite_key not in landmarks:
        return []
    ax, ay, az = landmarks[seed_key]
    cx, cy, cz = landmarks[opposite_key]
    candidates = []
    for t in (0.25, 0.45, 0.65, 0.85):
        candidates.append(
            (
                ax + (cx - ax) * t,
                ay + (cy - ay) * t,
                az + (cz - az) * t,
            )
        )
    return candidates


def _polyline_midpoint_point(polyline: vtkPolyData) -> tuple[float, float, float] | None:
    points = polyline.GetPoints()
    if points is None:
        return None
    count = points.GetNumberOfPoints()
    if count == 0:
        return None
    return points.GetPoint(count // 2)


def _collect_available_boundary_ids(
    locator,
    geodesic_lines: dict[str, vtkPolyData],
    boundary_keys: Sequence[str],
) -> set[int]:
    boundary_ids: set[int] = set()
    for key in boundary_keys:
        polyline = geodesic_lines.get(key)
        if polyline is None:
            continue
        points = polyline.GetPoints()
        if points is None:
            continue
        lines = polyline.GetLines()
        if lines is None:
            continue
        lines.InitTraversal()
        id_list = vtkIdList()
        while lines.GetNextCell(id_list):
            if id_list.GetNumberOfIds() < 2:
                continue
            for i in range(1, id_list.GetNumberOfIds()):
                p0 = points.GetPoint(id_list.GetId(i - 1))
                p1 = points.GetPoint(id_list.GetId(i))
                boundary_ids.add(locator.FindClosestPoint(p0))
                boundary_ids.add(locator.FindClosestPoint(p1))
                mid1 = (
                    p0[0] + (p1[0] - p0[0]) * (1.0 / 3.0),
                    p0[1] + (p1[1] - p0[1]) * (1.0 / 3.0),
                    p0[2] + (p1[2] - p0[2]) * (1.0 / 3.0),
                )
                mid2 = (
                    p0[0] + (p1[0] - p0[0]) * (2.0 / 3.0),
                    p0[1] + (p1[1] - p0[1]) * (2.0 / 3.0),
                    p0[2] + (p1[2] - p0[2]) * (2.0 / 3.0),
                )
                boundary_ids.add(locator.FindClosestPoint(mid1))
                boundary_ids.add(locator.FindClosestPoint(mid2))
    return boundary_ids


def _assign_boundary_vertices(
    segment_ids: vtkIntArray,
    adjacency: list[set[int]],
    boundary_ids: set[int],
) -> None:
    for vertex_id in boundary_ids:
        if segment_ids.GetValue(vertex_id) != 0:
            continue
        neighbor_counts: dict[int, int] = {}
        for neighbor in adjacency[vertex_id]:
            seg_id = segment_ids.GetValue(neighbor)
            if seg_id == 0:
                continue
            neighbor_counts[seg_id] = neighbor_counts.get(seg_id, 0) + 1
        if neighbor_counts:
            best_seg = max(neighbor_counts.items(), key=lambda item: (item[1], -item[0]))[0]
            segment_ids.SetValue(vertex_id, best_seg)


def _build_segment_ids(
    surface: vtkPolyData,
    segments: dict[int, set[int] | None],
    adjacency: list[set[int]],
    locator,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
) -> vtkIntArray:
    segment_ids = vtkIntArray()
    segment_ids.SetName("SegmentId")
    segment_ids.SetNumberOfComponents(1)
    segment_ids.SetNumberOfTuples(surface.GetNumberOfPoints())
    for i in range(surface.GetNumberOfPoints()):
        segment_ids.SetValue(i, 0)

    for seg_id in range(1, 10):
        if segments.get(seg_id):
            for vertex_id in segments[seg_id] or []:
                if seg_id == 1 or segment_ids.GetValue(vertex_id) == 0:
                    segment_ids.SetValue(vertex_id, seg_id)

    boundary_ids = _collect_boundary_ids(
        locator,
        landmarks,
        geodesic_lines,
        tuple(geodesic_lines.keys()),
    )
    if boundary_ids:
        _assign_boundary_vertices(segment_ids, adjacency, boundary_ids)
    return segment_ids


def compute_segment_ids(
    surface: vtkPolyData,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
) -> tuple[vtkIntArray | None, str | None, dict | None]:
    if surface is None:
        return None, "No surface loaded", None
    if not {"A", "B", "C", "D", "E", "F"}.issubset(landmarks.keys()):
        return None, "Missing base landmarks", None
    if not {"AB_anterior", "AB_posterior"}.issubset(geodesic_lines.keys()):
        return None, "Missing AB geodesics", None
    if not {"CD_anterior", "CD_posterior"}.issubset(geodesic_lines.keys()):
        return None, "Missing CD geodesics", None

    adjacency = _build_point_adjacency(surface)
    locator = build_point_locator(surface)
    segments: dict[int, set[int] | None] = {}

    def compute_segment_with_fallback(
        seg_id: int,
        deps: Sequence[str],
        opposite_key: str,
        seed_key: str,
        blocked_ids: set[int] | None,
        required_landmarks: set[str] | None = None,
        allow_fallback: bool = True,
        report_missing_deps: bool = True,
    ) -> tuple[set[int] | None, str | None, dict | None]:
        if required_landmarks is not None and not required_landmarks.issubset(landmarks.keys()):
            return None, None, None

        present = all(key in geodesic_lines for key in deps)
        if not present:
            if report_missing_deps:
                missing = [key for key in deps if key not in geodesic_lines]
                if missing:
                    message = (
                        f"Segment {seg_id} skipped: missing boundary geodesics "
                        f"({', '.join(missing)})"
                    )
                else:
                    message = f"Segment {seg_id} skipped: missing boundary geodesics"
                debug_points = {
                    "boundary_ids": list(_collect_available_boundary_ids(locator, geodesic_lines, deps)),
                    "seed_id": None,
                    "opposite_id": None,
                    "seed_candidate_id": None,
                    "opposite_seed_id": None,
                }
                return None, message, debug_points
            return None, None, None

        seed_point = None
        if deps:
            boundary_key = deps[0]
            boundary_polyline = geodesic_lines.get(boundary_key)
            if boundary_polyline is not None:
                seed_point = _polyline_midpoint_point(boundary_polyline)

        segment = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            deps,
            opposite_key=opposite_key,
            seed_key=seed_key,
            blocked_ids=blocked_ids if blocked_ids else None,
            seed_point=seed_point,
        )
        if segment is None and allow_fallback:
            for candidate in _seed_fallback_candidates(landmarks, seed_key, opposite_key):
                segment = _collect_segment_component(
                    surface,
                    adjacency,
                    locator,
                    landmarks,
                    geodesic_lines,
                    deps,
                    opposite_key=opposite_key,
                    seed_key=seed_key,
                    blocked_ids=blocked_ids if blocked_ids else None,
                    seed_point=candidate,
                )
                if segment is not None:
                    break

        if segment is None:
            debug_points = _collect_failure_debug(
                adjacency,
                locator,
                landmarks,
                geodesic_lines,
                deps,
                opposite_key=opposite_key,
                seed_key=seed_key,
                blocked_ids=blocked_ids if blocked_ids else None,
                seed_point=seed_point,
            )
            if debug_points is None:
                debug_points = {
                    "boundary_ids": list(_collect_available_boundary_ids(locator, geodesic_lines, deps)),
                    "boundary_count": None,
                    "blocked_count": None,
                    "total_points": None,
                    "seed_id": None,
                    "opposite_id": None,
                    "seed_candidate_id": None,
                    "opposite_seed_id": None,
                }
            message = _diagnose_segment_failure(
                seg_id,
                adjacency,
                locator,
                landmarks,
                geodesic_lines,
                deps,
                opposite_key=opposite_key,
                seed_key=seed_key,
                blocked_ids=blocked_ids if blocked_ids else None,
                seed_point=seed_point,
            )
            seed_id = debug_points.get("seed_id")
            opposite_id = debug_points.get("opposite_id")
            opposite_seed_id = debug_points.get("opposite_seed_id")
            seed_candidate_id = debug_points.get("seed_candidate_id")
            boundary_count = debug_points.get("boundary_count")
            blocked_count = debug_points.get("blocked_count")
            total_points = debug_points.get("total_points")
            detail = (
                f" seed_id={seed_id}"
                f" opposite_id={opposite_id}"
                f" opposite_seed_id={opposite_seed_id}"
                f" seed_candidate_id={seed_candidate_id}"
                f" boundary_count={boundary_count}"
                f" blocked_count={blocked_count}"
                f" total_points={total_points}"
            )
            return (
                None,
                f"{message} |{detail}",
                debug_points,
            )

        return segment, None, None

    seg1_deps = ["AB_posterior", "AB_anterior"]
    if {"X1_X2_anterior", "X1_X2_posterior"}.issubset(geodesic_lines.keys()):
        seg1_deps.extend(["X1_X2_anterior", "X1_X2_posterior"])
    if {"A1_A2_anterior", "A1_A2_posterior"}.issubset(geodesic_lines.keys()):
        seg1_deps.extend(["A1_A2_anterior", "A1_A2_posterior"])      
    if {"B1_B2_anterior", "B1_B2_posterior"}.issubset(geodesic_lines.keys()):
        seg1_deps.extend(["B1_B2_anterior", "B1_B2_posterior"])            
    
    seg1, error_message, debug_points = compute_segment_with_fallback(
        1,
        seg1_deps,
        opposite_key="D",
        seed_key="A",
        blocked_ids=None,
        allow_fallback=False,
        report_missing_deps=True,
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg1:
        segments[1] = seg1

    seg2_deps = ["CD_posterior", "CD_anterior"]
    if {"C1_C2_anterior", "C1_C2_posterior"}.issubset(geodesic_lines.keys()):
        seg2_deps.extend(["C1_C2_anterior", "C1_C2_posterior"])     
    if {"D1_D2_anterior", "D1_D2_posterior"}.issubset(geodesic_lines.keys()):
        seg2_deps.extend(["D1_D2_anterior", "D1_D2_posterior"])         

    seg2, error_message, debug_points = compute_segment_with_fallback(
        2,
        seg2_deps,
        opposite_key="A",
        seed_key="C",
        blocked_ids=segments.get(1),
        allow_fallback=False,
        report_missing_deps=True,
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg2:
        segments[2] = seg2

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()

    seg3_deps = ("AB_posterior", "CD_posterior", "AC", "BD")
    seg3, error_message, debug_points = compute_segment_with_fallback(
        3,
        seg3_deps,
        opposite_key="E",
        seed_key="C",
        blocked_ids=blocked if blocked else None,
        allow_fallback=False,
        report_missing_deps=True,
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg3:
        segments[3] = seg3

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()

    seg4_deps = ("AC", "AF", "CE", "EF_aniso")
    seg4, error_message, debug_points = compute_segment_with_fallback(
        4,
        seg4_deps,
        opposite_key="H",
        seed_key="C",
        blocked_ids=blocked if blocked else None,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg4:
        segments[4] = seg4

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()
    if segments.get(4):
        blocked |= segments[4] or set()

    seg5_deps = ["LAA1_LAA2_anterior", "LAA1_LAA2_posterior"]
    if {"X1_X2_anterior", "X1_X2_posterior"}.issubset(geodesic_lines.keys()):
        seg5_deps.extend(["X1_X2_anterior", "X1_X2_posterior"])
    seg5, error_message, debug_points = compute_segment_with_fallback(
        5,
        seg5_deps,
        opposite_key="I",
        seed_key="LAA1",
        blocked_ids=blocked if blocked else None,
        allow_fallback=False,
        report_missing_deps=True,
        required_landmarks={"LAA1", "LAA2", "I", "F"},
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg5:
        segments[5] = seg5
        blocked |= seg5

    seg6_deps = ["AB_anterior", "AF", "BH", "FH_aniso"]
    if {"LAA1_LAA2_anterior", "LAA1_LAA2_posterior"}.issubset(geodesic_lines.keys()):
        seg6_deps.extend(["LAA1_LAA2_anterior", "LAA1_LAA2_posterior"])
    seg6, error_message, debug_points = compute_segment_with_fallback(
        6,
        seg6_deps,
        opposite_key="E",
        seed_key="B",
        blocked_ids=blocked if blocked else None,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg6:
        segments[6] = seg6

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()
    if segments.get(4):
        blocked |= segments[4] or set()
    if segments.get(6):
        blocked |= segments[6] or set()

    seg7_deps = ("BD", "DI", "BH", "HI_aniso")
    seg7, error_message, debug_points = compute_segment_with_fallback(
        7,
        seg7_deps,
        opposite_key="A",
        seed_key="D",
        blocked_ids=blocked if blocked else None,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg7:
        segments[7] = seg7

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()
    if segments.get(4):
        blocked |= segments[4] or set()
    if segments.get(6):
        blocked |= segments[6] or set()
    if segments.get(7):
        blocked |= segments[7] or set()

    seg8_deps = ("CD_anterior", "CE", "DI", "IE_aniso")
    seg8, error_message, debug_points = compute_segment_with_fallback(
        8,
        seg8_deps,
        opposite_key="B",
        seed_key="D",
        blocked_ids=blocked if blocked else None,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg8:
        segments[8] = seg8

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()
    if segments.get(4):
        blocked |= segments[4] or set()
    if segments.get(5):
        blocked |= segments[5] or set()
    if segments.get(6):
        blocked |= segments[6] or set()
    if segments.get(7):
        blocked |= segments[7] or set()
    if segments.get(8):
        blocked |= segments[8] or set()

    seg9_deps = ("EF_aniso", "FH_aniso", "HI_aniso", "IE_aniso")
    seg9, error_message, debug_points = compute_segment_with_fallback(
        9,
        seg9_deps,
        opposite_key="B",
        seed_key="E",
        blocked_ids=None,
        required_landmarks={"E", "F", "H", "I"},
    )
    if error_message:
        return (
            _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
            error_message,
            debug_points,
        )
    if seg9:
        segments[9] = seg9

    return (
        _build_segment_ids(surface, segments, adjacency, locator, landmarks, geodesic_lines),
        None,
        None,
    )
