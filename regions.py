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
            "debug_messages": {},
            "segment_updates": {},
        }

    adjacency = cache["adjacency"]
    locator = cache["locator"]
    segments: dict[int, set[int] | None] = cache["segments"]
    deps_present_cache: dict[int, bool] = cache["deps_present"]
    debug_messages: dict[int, str] = cache.setdefault("debug_messages", {})

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

    def update_debug_message(seg_id: int, message: str | None) -> None:
        if message is None:
            debug_messages.pop(seg_id, None)
        else:
            debug_messages[seg_id] = message

        if debug_messages:
            ordered = [debug_messages[key] for key in sorted(debug_messages.keys())]
            cache["debug"] = " | ".join(ordered)
        else:
            cache.pop("debug", None)

    def compute_segment_with_fallback(
        seg_id: int,
        deps: Sequence[str],
        opposite_key: str,
        seed_key: str,
        blocked_ids: set[int] | None,
        upstream_changed: bool,
        required_landmarks: set[str] | None = None,
        allow_fallback: bool = True,
        report_missing_deps: bool = True,
    ) -> tuple[bool, bool]:
        present = deps_present(deps)
        changed = False
        if required_landmarks is not None and not required_landmarks.issubset(landmarks.keys()):
            segments[seg_id] = None
            changed = deps_present_cache.get(seg_id) != present
            update_debug_message(seg_id, None)
            deps_present_cache[seg_id] = present
            return present, changed
        if present and should_recompute(seg_id, deps, present, upstream_changed=upstream_changed):
            segments[seg_id] = _collect_segment_component(
                surface,
                adjacency,
                locator,
                landmarks,
                geodesic_lines,
                deps,
                opposite_key=opposite_key,
                seed_key=seed_key,
                blocked_ids=blocked_ids if blocked_ids else None,
            )
            if segments[seg_id] is None and allow_fallback:
                for seed_point in _seed_fallback_candidates(landmarks, seed_key, opposite_key):
                    segments[seg_id] = _collect_segment_component(
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
                    if segments[seg_id] is not None:
                        break
            changed = True
            if segments[seg_id] is None:
                update_debug_message(
                    seg_id,
                    _diagnose_segment_failure(
                        seg_id,
                        adjacency,
                        locator,
                        landmarks,
                        geodesic_lines,
                        deps,
                        opposite_key=opposite_key,
                        seed_key=seed_key,
                        blocked_ids=blocked_ids if blocked_ids else None,
                    ),
                )
            else:
                update_debug_message(seg_id, None)
        elif not present:
            segments[seg_id] = None
            changed = deps_present_cache.get(seg_id) != present
            if report_missing_deps:
                update_debug_message(seg_id, f"Segment {seg_id} skipped: missing boundary geodesics")

        if present and segments.get(seg_id) is not None:
            update_debug_message(seg_id, None)

        deps_present_cache[seg_id] = present
        return present, changed

    seg1_deps = ("AB_anterior", "AB_posterior")
    seg1_present, seg1_changed = compute_segment_with_fallback(
        1,
        seg1_deps,
        opposite_key="C",
        seed_key="A",
        blocked_ids=None,
        upstream_changed=False,
        allow_fallback=False,
        report_missing_deps=True,
    )

    seg2_deps = ("CD_anterior", "CD_posterior")
    seg2_present, seg2_changed = compute_segment_with_fallback(
        2,
        seg2_deps,
        opposite_key="A",
        seed_key="C",
        blocked_ids=segments[1] if segments.get(1) else None,
        upstream_changed=seg1_changed,
        allow_fallback=False,
        report_missing_deps=True,
    )

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()

    seg3_deps = ("AB_posterior", "CD_posterior", "AC", "BD")
    seg3_present, seg3_changed = compute_segment_with_fallback(
        3,
        seg3_deps,
        opposite_key="E",
        seed_key="C",
        blocked_ids=blocked if blocked else None,
        upstream_changed=seg1_changed or seg2_changed,
        allow_fallback=False,
        report_missing_deps=True,
    )

    blocked = set()
    if segments.get(1):
        blocked |= segments[1] or set()
    if segments.get(2):
        blocked |= segments[2] or set()
    if segments.get(3):
        blocked |= segments[3] or set()

    seg4_deps = ("AC", "AF", "CE", "EF_aniso")
    seg4_present, seg4_changed = compute_segment_with_fallback(
        4,
        seg4_deps,
        opposite_key="H",
        seed_key="C",
        blocked_ids=blocked if blocked else None,
        upstream_changed=seg1_changed or seg2_changed or seg3_changed,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )

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
    seg6_present, seg6_changed = compute_segment_with_fallback(
        6,
        seg6_deps,
        opposite_key="E",
        seed_key="B",
        blocked_ids=blocked if blocked else None,
        upstream_changed=seg4_changed,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )

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
    seg7_present, seg7_changed = compute_segment_with_fallback(
        7,
        seg7_deps,
        opposite_key="F",
        seed_key="D",
        blocked_ids=blocked if blocked else None,
        upstream_changed=seg6_changed,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )

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
    seg8_present, seg8_changed = compute_segment_with_fallback(
        8,
        seg8_deps,
        opposite_key="F",
        seed_key="D",
        blocked_ids=blocked if blocked else None,
        upstream_changed=seg7_changed,
        required_landmarks={"A", "B", "C", "D", "E", "F", "H", "I"},
    )

    segment_updates: dict[int, bool] = {}
    if seg1_changed:
        segment_updates[1] = segments.get(1) is not None
    if seg2_changed:
        segment_updates[2] = segments.get(2) is not None
    if seg3_changed:
        segment_updates[3] = segments.get(3) is not None
    if seg4_changed:
        segment_updates[4] = segments.get(4) is not None
    if seg6_changed:
        segment_updates[6] = segments.get(6) is not None
    if seg7_changed:
        segment_updates[7] = segments.get(7) is not None
    if seg8_changed:
        segment_updates[8] = segments.get(8) is not None

    cache["segment_updates"] = segment_updates

    segment_changed = bool(segment_updates)
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
