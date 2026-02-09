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


def compute_segment_ids(
    surface: vtkPolyData,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
) -> vtkIntArray | None:
    segment_ids, _ = compute_segment_ids_cached(
        surface,
        landmarks,
        geodesic_lines,
        cache=None,
        changed_geodesics=None,
    )
    return segment_ids


def compute_segment_ids_cached(
    surface: vtkPolyData,
    landmarks: dict[str, Sequence[float]],
    geodesic_lines: dict[str, vtkPolyData],
    cache: dict | None,
    changed_geodesics: set[str] | None,
) -> tuple[vtkIntArray | None, dict | None]:
    if surface is None:
        return None, cache
    if not {"A", "B", "C", "D", "E", "F"}.issubset(landmarks.keys()):
        return None, cache
    if not {"AB_anterior", "AB_posterior"}.issubset(geodesic_lines.keys()):
        return None, cache
    if not {"CD_anterior", "CD_posterior"}.issubset(geodesic_lines.keys()):
        return None, cache

    if cache is None:
        cache = {}
    if cache.get("surface_points") != surface.GetNumberOfPoints():
        cache = {
            "surface_points": surface.GetNumberOfPoints(),
            "adjacency": _build_point_adjacency(surface),
            "locator": build_point_locator(surface),
            "segments": {},
            "deps_present": {},
            "segment_ids": None,
        }

    adjacency = cache["adjacency"]
    locator = cache["locator"]
    segments: dict[int, set[int] | None] = cache["segments"]
    deps_present_cache: dict[int, bool] = cache["deps_present"]

    if changed_geodesics is not None:
        changed_geodesics = set(changed_geodesics)

    segment_changed = False

    def deps_present(deps: Sequence[str]) -> bool:
        return all(key in geodesic_lines for key in deps)

    def should_recompute(seg_id: int, deps: Sequence[str], present: bool, upstream_changed: bool) -> bool:
        if not present:
            return False
        if seg_id not in segments:
            return True
        if deps_present_cache.get(seg_id) != present:
            return True
        if upstream_changed:
            return True
        if changed_geodesics is None:
            return True
        return any(key in changed_geodesics for key in deps)

    seg1_deps = ("AB_anterior", "AB_posterior")
    seg1_present = deps_present(seg1_deps)
    seg1_changed = False
    if seg1_present and should_recompute(1, seg1_deps, seg1_present, upstream_changed=False):
        segments[1] = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            seg1_deps,
            opposite_key="C",
            seed_key="A",
        )
        seg1_changed = True
    elif not seg1_present:
        segments[1] = None
        seg1_changed = deps_present_cache.get(1) != seg1_present

    deps_present_cache[1] = seg1_present

    seg2_deps = ("CD_anterior", "CD_posterior")
    seg2_present = deps_present(seg2_deps)
    seg2_changed = False
    if seg2_present and should_recompute(2, seg2_deps, seg2_present, upstream_changed=seg1_changed):
        segments[2] = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            seg2_deps,
            opposite_key="A",
            seed_key="C",
            blocked_ids=segments[1] if segments.get(1) else None,
        )
        seg2_changed = True
    elif not seg2_present:
        segments[2] = None
        seg2_changed = deps_present_cache.get(2) != seg2_present

    deps_present_cache[2] = seg2_present

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()

    seg3_deps = ("AB_posterior", "CD_posterior", "AC", "BD")
    seg3_present = deps_present(seg3_deps)
    seg3_changed = False
    if seg3_present and should_recompute(3, seg3_deps, seg3_present, upstream_changed=seg1_changed or seg2_changed):
        segments[3] = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            seg3_deps,
            opposite_key="E",
            seed_key="C",
            blocked_ids=blocked if blocked else None,
        )
        seg3_changed = True
    elif not seg3_present:
        segments[3] = None
        seg3_changed = deps_present_cache.get(3) != seg3_present

    deps_present_cache[3] = seg3_present

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()

    seg4_deps = ("AC", "AF", "CE", "EF_aniso")
    seg4_present = deps_present(seg4_deps)
    seg4_changed = False
    if seg4_present and should_recompute(4, seg4_deps, seg4_present, upstream_changed=seg1_changed or seg2_changed or seg3_changed):
        segments[4] = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            seg4_deps,
            opposite_key="H",
            seed_key="C",
            blocked_ids=blocked if blocked else None,
        )
        if segments[4] is None:
            for seed_point in _seed_fallback_candidates(landmarks, "C", "H"):
                segments[4] = _collect_segment_component(
                    surface,
                    adjacency,
                    locator,
                    landmarks,
                    geodesic_lines,
                    seg4_deps,
                    opposite_key="H",
                    seed_key="C",
                    blocked_ids=blocked if blocked else None,
                    seed_point=seed_point,
                )
                if segments[4] is not None:
                    break
        seg4_changed = True
        if segments[4] is None:
            cache["debug"] = _diagnose_segment_failure(
                4,
                adjacency,
                locator,
                landmarks,
                geodesic_lines,
                seg4_deps,
                opposite_key="H",
                seed_key="C",
                blocked_ids=blocked if blocked else None,
            )
        else:
            cache.pop("debug", None)
    elif not seg4_present:
        segments[4] = None
        seg4_changed = deps_present_cache.get(4) != seg4_present
        cache["debug"] = "Segment 4 skipped: missing boundary geodesics"

    deps_present_cache[4] = seg4_present

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()
    if segments.get(4):
        blocked |= segments[4] or set()

    seg6_deps = ("AB_anterior", "AF", "BH", "FH_aniso")
    seg6_present = deps_present(seg6_deps)
    seg6_changed = False
    if seg6_present and should_recompute(6, seg6_deps, seg6_present, upstream_changed=seg4_changed):
        segments[6] = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            seg6_deps,
            opposite_key="E",
            seed_key="B",
            blocked_ids=blocked if blocked else None,
        )
        if segments[6] is None:
            for seed_point in _seed_fallback_candidates(landmarks, "B", "E"):
                segments[6] = _collect_segment_component(
                    surface,
                    adjacency,
                    locator,
                    landmarks,
                    geodesic_lines,
                    seg6_deps,
                    opposite_key="E",
                    seed_key="B",
                    blocked_ids=blocked if blocked else None,
                    seed_point=seed_point,
                )
                if segments[6] is not None:
                    break
        seg6_changed = True
        if segments[6] is None:
            cache["debug"] = _diagnose_segment_failure(
                6,
                adjacency,
                locator,
                landmarks,
                geodesic_lines,
                seg6_deps,
                opposite_key="E",
                seed_key="B",
                blocked_ids=blocked if blocked else None,
            )
        else:
            cache.pop("debug", None)
    elif not seg6_present:
        segments[6] = None
        seg6_changed = deps_present_cache.get(6) != seg6_present

    deps_present_cache[6] = seg6_present

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

    seg7_deps = ("BH", "DI", "BD", "HI_aniso")
    seg7_present = deps_present(seg7_deps)
    seg7_changed = False
    if seg7_present and should_recompute(7, seg7_deps, seg7_present, upstream_changed=seg6_changed):
        segments[7] = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            seg7_deps,
            opposite_key="F",
            seed_key="D",
            blocked_ids=blocked if blocked else None,
        )
        if segments[7] is None:
            for seed_point in _seed_fallback_candidates(landmarks, "D", "F"):
                segments[7] = _collect_segment_component(
                    surface,
                    adjacency,
                    locator,
                    landmarks,
                    geodesic_lines,
                    seg7_deps,
                    opposite_key="F",
                    seed_key="D",
                    blocked_ids=blocked if blocked else None,
                    seed_point=seed_point,
                )
                if segments[7] is not None:
                    break
        seg7_changed = True
        if segments[7] is None:
            cache["debug"] = _diagnose_segment_failure(
                7,
                adjacency,
                locator,
                landmarks,
                geodesic_lines,
                seg7_deps,
                opposite_key="F",
                seed_key="D",
                blocked_ids=blocked if blocked else None,
            )
        else:
            cache.pop("debug", None)
    elif not seg7_present:
        segments[7] = None
        seg7_changed = deps_present_cache.get(7) != seg7_present

    deps_present_cache[7] = seg7_present

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
    seg8_present = deps_present(seg8_deps)
    seg8_changed = False
    if seg8_present and should_recompute(8, seg8_deps, seg8_present, upstream_changed=seg7_changed):
        segments[8] = _collect_segment_component(
            surface,
            adjacency,
            locator,
            landmarks,
            geodesic_lines,
            seg8_deps,
            opposite_key="F",
            seed_key="D",
            blocked_ids=blocked if blocked else None,
        )
        if segments[8] is None:
            for seed_point in _seed_fallback_candidates(landmarks, "D", "F"):
                segments[8] = _collect_segment_component(
                    surface,
                    adjacency,
                    locator,
                    landmarks,
                    geodesic_lines,
                    seg8_deps,
                    opposite_key="F",
                    seed_key="D",
                    blocked_ids=blocked if blocked else None,
                    seed_point=seed_point,
                )
                if segments[8] is not None:
                    break
        seg8_changed = True
        if segments[8] is None:
            cache["debug"] = _diagnose_segment_failure(
                8,
                adjacency,
                locator,
                landmarks,
                geodesic_lines,
                seg8_deps,
                opposite_key="F",
                seed_key="D",
                blocked_ids=blocked if blocked else None,
            )
        else:
            cache.pop("debug", None)
    elif not seg8_present:
        segments[8] = None
        seg8_changed = deps_present_cache.get(8) != seg8_present

    deps_present_cache[8] = seg8_present

    segment_changed = (
        seg1_changed
        or seg2_changed
        or seg3_changed
        or seg4_changed
        or seg6_changed
        or seg7_changed
        or seg8_changed
    )
    cached_ids = cache.get("segment_ids")
    if not segment_changed and cached_ids is not None:
        return cached_ids, cache

    segment_ids = vtkIntArray()
    segment_ids.SetName("SegmentId")
    segment_ids.SetNumberOfComponents(1)
    segment_ids.SetNumberOfTuples(surface.GetNumberOfPoints())
    for i in range(surface.GetNumberOfPoints()):
        segment_ids.SetValue(i, 0)

    if segments.get(1):
        for vertex_id in segments[1] or []:
            segment_ids.SetValue(vertex_id, 1)

    if segments.get(2):
        for vertex_id in segments[2] or []:
            if segment_ids.GetValue(vertex_id) == 0:
                segment_ids.SetValue(vertex_id, 2)

    if segments.get(3):
        for vertex_id in segments[3] or []:
            if segment_ids.GetValue(vertex_id) == 0:
                segment_ids.SetValue(vertex_id, 3)

    if segments.get(4):
        for vertex_id in segments[4] or []:
            if segment_ids.GetValue(vertex_id) == 0:
                segment_ids.SetValue(vertex_id, 4)

    if segments.get(6):
        for vertex_id in segments[6] or []:
            if segment_ids.GetValue(vertex_id) == 0:
                segment_ids.SetValue(vertex_id, 6)

    if segments.get(7):
        for vertex_id in segments[7] or []:
            if segment_ids.GetValue(vertex_id) == 0:
                segment_ids.SetValue(vertex_id, 7)

    if segments.get(8):
        for vertex_id in segments[8] or []:
            if segment_ids.GetValue(vertex_id) == 0:
                segment_ids.SetValue(vertex_id, 8)

    cache["segment_ids"] = segment_ids
    return segment_ids, cache
