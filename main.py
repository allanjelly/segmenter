import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPointLocator, vtkPolyData
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkRenderer,
)

from geodesics import (
    build_point_locator,
    compute_ma_plane_normal,
    create_anisotropic_geodesic,
    create_pair_geodesics,
    create_simple_geodesic,
)
from regions import compute_segment_ids


def detect_os() -> str:
    platform = sys.platform
    if platform.startswith("win"):
        return "windows"
    if platform == "darwin":
        return "mac"
    if platform.startswith("linux"):
        try:
            with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as handle:
                if "microsoft" in handle.read().lower():
                    return "wsl"
        except OSError:
            pass
        return "linux"
    return "unknown"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_file: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Segmenter")
        self.resize(1200, 800)
        self._mesh_actor = None
        self._mesh_mapper = None
        self._initial_file = initial_file
        self._pending_file = None
        self._polydata = None
        self._point_locator = None
        self._geo_locator = None
        self._renderer = vtkRenderer()
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.01)
        self._landmarks: dict[str, tuple[float, float, float]] = {}
        self._landmark_actors: dict[str, vtkActor] = {}
        self._geodesic_actors: dict[str, vtkActor] = {}
        self._geodesic_lines: dict[str, object] = {}
        self._aux_actors: dict[str, vtkActor] = {}
        self._segment_lut = self._build_segment_lut()
        self._current_step_index = 0
        self._steps = self._build_steps()
        self._message_box = None
        self._error_box = None
        self._updating_steps = False

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._left_panel = self._build_left_panel()
        self._vtk_widget = QVTKRenderWindowInteractor(central)
        self._vtk_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._vtk_widget.setFocus()

        layout.addWidget(self._left_panel)
        layout.addWidget(self._vtk_widget, stretch=1)
        self.setCentralWidget(central)

        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._setup_shortcuts()
        self.statusBar().showMessage("")

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        mesh_group = QtWidgets.QGroupBox("Mesh", panel)
        mesh_layout = QtWidgets.QVBoxLayout(mesh_group)
        self._mesh_info = QtWidgets.QLabel("No mesh loaded", mesh_group)
        self._mesh_info.setWordWrap(True)
        mesh_layout.addWidget(self._mesh_info)
        layout.addWidget(mesh_group)

        landmarks_group = QtWidgets.QGroupBox("Landmarks", panel)
        landmarks_layout = QtWidgets.QVBoxLayout(landmarks_group)
        self._step_label = QtWidgets.QLabel("Current: -", landmarks_group)
        landmarks_layout.addWidget(self._step_label)

        self._steps_list = QtWidgets.QListWidget(landmarks_group)
        self._steps_list.currentRowChanged.connect(self._on_step_changed)
        self._steps_list.itemChanged.connect(self._on_step_item_changed)
        self._steps_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        landmarks_layout.addWidget(self._steps_list)

        buttons_layout = QtWidgets.QHBoxLayout()
        self._prev_button = QtWidgets.QPushButton("Prev", landmarks_group)
        self._next_button = QtWidgets.QPushButton("Next", landmarks_group)
        self._prev_button.clicked.connect(self._go_prev_step)
        self._next_button.clicked.connect(self._go_next_step)
        buttons_layout.addWidget(self._prev_button)
        buttons_layout.addWidget(self._next_button)
        landmarks_layout.addLayout(buttons_layout)
        layout.addWidget(landmarks_group)

        controls_group = QtWidgets.QGroupBox("Controls", panel)
        controls_layout = QtWidgets.QVBoxLayout(controls_group)
        controls_label = QtWidgets.QLabel(
            "- Drag: rotate\n"
            "- Right drag: zoom\n"
            "- Middle drag: pan\n"
            "- Mouse wheel: zoom\n"
            "- Click mesh: set landmark\n"
            "Keyboard:\n"
            "- Space: next step\n"
            "- Esc: previous step",
            controls_group,
        )
        controls_label.setWordWrap(True)
        controls_layout.addWidget(controls_label)

        self._calculate_regions_button = QtWidgets.QPushButton("Calculate regions", controls_group)
        self._calculate_regions_button.clicked.connect(self._calculate_regions)
        controls_layout.addWidget(self._calculate_regions_button)
        layout.addWidget(controls_group)

        messages_group = QtWidgets.QGroupBox("Messages", panel)
        messages_layout = QtWidgets.QVBoxLayout(messages_group)
        self._message_box = QtWidgets.QPlainTextEdit(messages_group)
        self._message_box.setReadOnly(True)
        self._message_box.setMinimumHeight(120)
        messages_layout.addWidget(self._message_box)
        layout.addWidget(messages_group)

        errors_group = QtWidgets.QGroupBox("Errors", panel)
        errors_layout = QtWidgets.QVBoxLayout(errors_group)
        self._error_box = QtWidgets.QPlainTextEdit(errors_group)
        self._error_box.setReadOnly(True)
        self._error_box.setMinimumHeight(40)
        errors_layout.addWidget(self._error_box)
        layout.addWidget(errors_group)

        layout.addStretch(1)

        self._populate_steps()
        return panel

    def showEvent(self, event: QtCore.QEvent) -> None:
        super().showEvent(event)
        if self._vtk_widget is not None:
            QtCore.QTimer.singleShot(0, self._initialize_vtk)
        if self._initial_file:
            self._pending_file = self._initial_file

    def _initialize_vtk(self) -> None:
        if self._vtk_widget is not None:
            self._vtk_widget.Initialize()
            interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
            if interactor is not None:
                interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
                interactor.AddObserver(
                    "LeftButtonPressEvent",
                    self._on_left_button_press,
                )
            self._vtk_widget.GetRenderWindow().Render()
        if self._pending_file:
            file_path = self._pending_file
            self._pending_file = None
            self.load_mesh(file_path)

    def load_mesh(self, file_path: str) -> None:
        polydata = self._read_vtk_polydata(Path(file_path))
        if polydata is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Load Failed",
                "Failed to read VTK polydata.",
            )
            return

        self._polydata = polydata
        self._point_locator = vtkPointLocator()
        self._point_locator.SetDataSet(polydata)
        self._point_locator.BuildLocator()
        self._geo_locator = build_point_locator(polydata)

        self._display_polydata(polydata)
        self._update_mesh_info(polydata, file_path)
        self._append_message(f"Mesh loaded: {Path(file_path).name}")

    def _read_vtk_polydata(self, path: Path):
        reader = vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        polydata = reader.GetOutput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return None
        return polydata

    def _display_polydata(self, polydata) -> None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("SegmentId")
        mapper.SetLookupTable(self._segment_lut)
        mapper.SetScalarRange(0, 9)
        mapper.SetScalarVisibility(True)

        actor = vtkActor()
        actor.SetMapper(mapper)

        if self._mesh_actor is not None:
            self._renderer.RemoveActor(self._mesh_actor)

        self._mesh_actor = actor
        self._mesh_mapper = mapper
        self._renderer.AddActor(actor)
        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def _update_mesh_info(self, polydata, file_path: str) -> None:
        num_points = polydata.GetNumberOfPoints()
        num_cells = polydata.GetNumberOfCells()
        name = Path(file_path).name
        self._mesh_info.setText(
            f"{name}\nVerts: {num_points} Cells: {num_cells}"
        )

    def _build_steps(self) -> list[dict[str, str]]:
        return [
            {"key": "A", "label": "A: LSPV"},
            {"key": "B", "label": "B: LIPV"},
            {"key": "C", "label": "C: RSPV"},
            {"key": "D", "label": "D: RIPV"},
            {"key": "E", "label": "E: Mitral annulus 9 o'clock"},
            {"key": "F", "label": "F: Mitral annulus 1 o'clock"},
            {"key": "H", "label": "H: Mitral annulus 4 o'clock"},
            {"key": "I", "label": "I: Mitral annulus 7 o'clock"},
            {"key": "LAA1", "label": "LAA1: Appendage orifice point 1"},
            {"key": "LAA2", "label": "LAA2: Appendage orifice point 2"},
            {"key": "X1", "label": "X1: LAA/LSPV cut point 1"},
            {"key": "X2", "label": "X2: LAA/LSPV cut point 2"},
        ]

    def _populate_steps(self) -> None:
        self._steps_list.blockSignals(True)
        self._updating_steps = True
        self._steps_list.clear()
        for step in self._steps:
            item = QtWidgets.QListWidgetItem(step["label"])
            item.setData(QtCore.Qt.UserRole, step["key"])
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole + 1, False)
            self._steps_list.addItem(item)
        self._updating_steps = False
        self._steps_list.blockSignals(False)
        if self._steps:
            self._steps_list.setCurrentRow(0)
            self._update_step_label()

        row_height = self._steps_list.sizeHintForRow(0)
        if row_height <= 0:
            row_height = 24
        total_height = row_height * self._steps_list.count() + self._steps_list.frameWidth() * 2
        self._steps_list.setMinimumHeight(total_height)

    def _on_step_changed(self, row: int) -> None:
        if row < 0:
            return
        self._current_step_index = row
        self._update_step_label()

    def _on_step_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._updating_steps:
            return
        stored = bool(item.data(QtCore.Qt.UserRole + 1))
        desired = QtCore.Qt.Checked if stored else QtCore.Qt.Unchecked
        if item.checkState() != desired:
            self._updating_steps = True
            item.setCheckState(desired)
            self._updating_steps = False

    def _update_step_label(self) -> None:
        if not self._steps:
            self._step_label.setText("No steps")
            return
        step = self._steps[self._current_step_index]
        self._step_label.setText(f"Current: {step['label']}")

    def _go_next_step(self) -> None:
        if not self._steps:
            return
        next_index = min(self._current_step_index + 1, len(self._steps) - 1)
        self._steps_list.setCurrentRow(next_index)

    def _go_prev_step(self) -> None:
        if not self._steps:
            return
        prev_index = max(self._current_step_index - 1, 0)
        self._steps_list.setCurrentRow(prev_index)

    def _setup_shortcuts(self) -> None:
        QtGui.QShortcut(QtCore.Qt.Key_Space, self, self._go_next_step)
        QtGui.QShortcut(QtCore.Qt.Key_Escape, self, self._go_prev_step)

    def _append_message(self, message: str) -> None:
        if self._message_box is not None and message:
            self._message_box.appendPlainText(message)

    def _set_error_message(self, message: str) -> None:
        if self._error_box is not None:
            self._error_box.setPlainText(message)

    def _calculate_regions(self) -> None:
        if self._polydata is None or self._mesh_mapper is None:
            return
        segment_ids, error_message, debug_points = compute_segment_ids(
            self._polydata,
            self._landmarks,
            self._geodesic_lines,
        )
        if error_message:
            self._set_error_message(error_message)
        else:
            self._set_error_message("")
        self._show_failure_debug(debug_points)
        if segment_ids is None:
            return
        self._apply_segment_ids(segment_ids)
        self._append_message("Regions calculated")
        if self._vtk_widget is not None:
            self._vtk_widget.GetRenderWindow().Render()

    def _on_left_button_press(self, obj, _event) -> None:
        if self._polydata is None:
            return
        if self._vtk_widget is None or self._renderer is None:
            return

        interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
        if interactor is None:
            return

        x, y = interactor.GetEventPosition()
        if not self._picker.Pick(x, y, 0, self._renderer):
            interactor.GetInteractorStyle().OnLeftButtonDown()
            return

        pick_pos = self._picker.GetPickPosition()
        if self._point_locator is None:
            return

        point_id = self._point_locator.FindClosestPoint(pick_pos)
        point = self._polydata.GetPoint(point_id)
        self._set_landmark_point(point)

        interactor.GetInteractorStyle().OnLeftButtonDown()

    def _set_landmark_point(self, point: tuple[float, float, float]) -> None:
        if not self._steps:
            return
        step = self._steps[self._current_step_index]
        key = step["key"]
        previous = self._landmarks.get(key)
        moved = (
            previous is None
            or (previous[0] - point[0]) ** 2
            + (previous[1] - point[1]) ** 2
            + (previous[2] - point[2]) ** 2
            > 1.0e-10
        )
        if moved:
            self._landmarks[key] = point
            self._update_landmark_actor(key, point)
        self._mark_step_completed(self._current_step_index)
        self._update_geodesics({key} if moved else set())
        self._go_next_step()
        self._vtk_widget.GetRenderWindow().Render()

    def _mark_step_completed(self, index: int) -> None:
        item = self._steps_list.item(index)
        if item is None:
            return
        self._updating_steps = True
        item.setData(QtCore.Qt.UserRole + 1, True)
        item.setCheckState(QtCore.Qt.Checked)
        self._updating_steps = False

    def _update_landmark_actor(self, key: str, point: tuple[float, float, float]) -> None:
        if self._renderer is None:
            return
        actor = self._landmark_actors.get(key)
        if actor is None:
            sphere = vtkSphereSource()
            sphere.SetRadius(1.0)
            sphere.SetThetaResolution(16)
            sphere.SetPhiResolution(16)
            sphere.Update()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.4, 0.2)
            self._renderer.AddActor(actor)
            self._landmark_actors[key] = actor

        actor.SetPosition(point)

    def _update_geodesics(self, changed_landmarks: set[str] | None = None) -> None:
        if self._polydata is None or self._geo_locator is None or self._renderer is None:
            return
        if changed_landmarks is None:
            changed_landmarks = set(self._landmarks.keys())
        changed_geodesics: set[str] = set()
        required_pairs = {"A", "B", "C", "D", "E"}
        if required_pairs.issubset(self._landmarks.keys()):
            ab_missing = not {"AB_anterior", "AB_posterior"}.issubset(self._geodesic_lines.keys())
            cd_missing = not {"CD_anterior", "CD_posterior"}.issubset(self._geodesic_lines.keys())
            ab_changed = bool({"A", "B"} & changed_landmarks) or ab_missing
            cd_changed = bool({"C", "D"} & changed_landmarks) or cd_missing
            ab_ok = True
            cd_ok = True
            if ab_changed:
                self._remove_geodesic("AB_anterior")
                self._remove_geodesic("AB_posterior")
                ab_ok = self._create_pair_geodesics("A", "B", ("A", "B", "C"))
                changed_geodesics.update({"AB_anterior", "AB_posterior"})
            if cd_changed:
                self._remove_geodesic("CD_anterior")
                self._remove_geodesic("CD_posterior")
                cd_ok = self._create_pair_geodesics("C", "D", ("A", "C", "D"))
                changed_geodesics.update({"CD_anterior", "CD_posterior"})
            if ab_changed and cd_changed and ab_ok and cd_ok:
                self._append_message("AB/CD geodesics updated")

        if (
            "A" in self._landmarks
            and "C" in self._landmarks
            and ({"A", "C"} & changed_landmarks or "AC" not in self._geodesic_lines)
        ):
            self._remove_geodesic("AC")
            self._create_simple_geodesic("AC", "A", "C", (0.2, 0.8, 1.0), 6.0)
            changed_geodesics.add("AC")

        if (
            "B" in self._landmarks
            and "D" in self._landmarks
            and ({"B", "D"} & changed_landmarks or "BD" not in self._geodesic_lines)
        ):
            self._remove_geodesic("BD")
            self._create_simple_geodesic("BD", "B", "D", (0.2, 0.8, 1.0), 6.0)
            changed_geodesics.add("BD")

        if (
            "C" in self._landmarks
            and "E" in self._landmarks
            and ({"C", "E"} & changed_landmarks or "CE" not in self._geodesic_lines)
        ):
            self._remove_geodesic("CE")
            self._create_simple_geodesic("CE", "C", "E", (0.7, 0.9, 0.3), 6.0)
            changed_geodesics.add("CE")

        if (
            "A" in self._landmarks
            and "F" in self._landmarks
            and ({"A", "F"} & changed_landmarks or "AF" not in self._geodesic_lines)
        ):
            self._remove_geodesic("AF")
            self._create_simple_geodesic("AF", "A", "F", (0.7, 0.9, 0.3), 6.0)
            changed_geodesics.add("AF")

        if (
            "B" in self._landmarks
            and "H" in self._landmarks
            and ({"B", "H"} & changed_landmarks or "BH" not in self._geodesic_lines)
        ):
            self._remove_geodesic("BH")
            self._create_simple_geodesic("BH", "B", "H", (0.7, 0.9, 0.3), 6.0)
            changed_geodesics.add("BH")

        if (
            "D" in self._landmarks
            and "I" in self._landmarks
            and ({"D", "I"} & changed_landmarks or "DI" not in self._geodesic_lines)
        ):
            self._remove_geodesic("DI")
            self._create_simple_geodesic("DI", "D", "I", (0.7, 0.9, 0.3), 6.0)
            changed_geodesics.add("DI")

        if (
            "LAA1" in self._landmarks
            and "LAA2" in self._landmarks
            and "D" in self._landmarks
            and "F" in self._landmarks
            and (
                {"LAA1", "LAA2", "D", "F"} & changed_landmarks
                or not {"LAA1_LAA2_anterior", "LAA1_LAA2_posterior"}.issubset(
                    self._geodesic_lines.keys()
                )
            )
        ):
            self._remove_geodesic("LAA1_LAA2_anterior")
            self._remove_geodesic("LAA1_LAA2_posterior")
            primary_key, primary, alternate_key, alternate = create_pair_geodesics(
                self._polydata,
                self._geo_locator,
                self._landmarks,
                "LAA1",
                "LAA2",
                ("LAA1", "LAA2", "D"),
                anterior_ref_key="F",
                plane_origin_key="D",
            )
            if primary_key.endswith("_anterior"):
                resolved_primary = "LAA1_LAA2_anterior"
                resolved_alternate = "LAA1_LAA2_posterior"
            else:
                resolved_primary = "LAA1_LAA2_posterior"
                resolved_alternate = "LAA1_LAA2_anterior"
            self._store_geodesic_actor(resolved_primary, primary.polyline, (0.9, 0.6, 0.1), 6.0)
            self._append_message(f"Geodesic {resolved_primary} updated")
            changed_geodesics.add(resolved_primary)
            if alternate is None:
                self._set_error_message("Alternate LAA1_LAA2 geodesic not found")
            else:
                self._store_geodesic_actor(resolved_alternate, alternate.polyline, (0.2, 0.7, 0.2), 6.0)
                self._append_message(f"Geodesic {resolved_alternate} updated")
                changed_geodesics.add(resolved_alternate)

        if "X1" not in self._landmarks or "X2" not in self._landmarks:
            self._remove_geodesic("X1_X2_anterior")
            self._remove_geodesic("X1_X2_posterior")
        elif (
            {"X1", "X2", "D", "F"} & changed_landmarks
            or not {"X1_X2_anterior", "X1_X2_posterior"}.issubset(
                self._geodesic_lines.keys()
            )
        ):
            self._remove_geodesic("X1_X2_anterior")
            self._remove_geodesic("X1_X2_posterior")
            primary_key, primary, alternate_key, alternate = create_pair_geodesics(
                self._polydata,
                self._geo_locator,
                self._landmarks,
                "X1",
                "X2",
                ("X1", "X2", "D"),
                anterior_ref_key="F",
                plane_origin_key="D",
            )
            if primary_key.endswith("_anterior"):
                resolved_primary = "X1_X2_anterior"
                resolved_alternate = "X1_X2_posterior"
            else:
                resolved_primary = "X1_X2_posterior"
                resolved_alternate = "X1_X2_anterior"
            self._store_geodesic_actor(resolved_primary, primary.polyline, (0.8, 0.8, 0.2), 6.0)
            self._append_message(f"Geodesic {resolved_primary} updated")
            changed_geodesics.add(resolved_primary)
            if alternate is None:
                self._set_error_message("Alternate X1_X2 geodesic not found")
            else:
                self._store_geodesic_actor(resolved_alternate, alternate.polyline, (0.2, 0.8, 0.8), 6.0)
                self._append_message(f"Geodesic {resolved_alternate} updated")
                changed_geodesics.add(resolved_alternate)

        has_ma_points = {"E", "F", "H", "I"}.issubset(self._landmarks.keys())
        if not has_ma_points:
            for key in ("EF_aniso", "FH_aniso", "HI_aniso", "IE_aniso"):
                self._remove_geodesic(key)
        elif (
            {"E", "F", "H", "I"} & changed_landmarks
            or not {"EF_aniso", "FH_aniso", "HI_aniso", "IE_aniso"}.issubset(
                self._geodesic_lines.keys()
            )
        ):
            ma_normal = compute_ma_plane_normal(
                self._landmarks["E"],
                self._landmarks["F"],
                self._landmarks["H"],
                self._landmarks["I"],
            )
            if ma_normal is None:
                for key in ("EF_aniso", "FH_aniso", "HI_aniso", "IE_aniso"):
                    self._remove_geodesic(key)
            else:
                penalty_strength = 5.0
                aniso_specs = (
                    ("EF_aniso", "E", "F", (0.9, 0.2, 0.2)),
                    ("FH_aniso", "F", "H", (0.2, 0.9, 0.2)),
                    ("HI_aniso", "H", "I", (0.2, 0.2, 0.9)),
                    ("IE_aniso", "I", "E", (0.9, 0.7, 0.2)),
                )
                for key, start_key, end_key, color in aniso_specs:
                    self._remove_geodesic(key)
                    result = create_anisotropic_geodesic(
                        self._polydata,
                        self._geo_locator,
                        self._landmarks,
                        start_key,
                        end_key,
                        ma_normal,
                        penalty_strength,
                    )
                    if result is not None:
                        self._store_geodesic_actor(key, result.polyline, color, 4.0)
                        changed_geodesics.add(key)
                        self._append_message(f"Geodesic {key} updated")

        self._vtk_widget.GetRenderWindow().Render()


    def _create_pair_geodesics(
        self,
        start_key: str,
        end_key: str,
        plane_keys: tuple[str, str, str],
    ) -> bool:
        primary_key, primary, alternate_key, alternate = create_pair_geodesics(
            self._polydata,
            self._geo_locator,
            self._landmarks,
            start_key,
            end_key,
            plane_keys,
        )

        self._store_geodesic_actor(primary_key, primary.polyline, (0.9, 0.6, 0.1), 6.0)
        self._append_message(f"Geodesic {primary_key} updated")
        if alternate is None:
            self._set_error_message(f"Alternate {start_key}{end_key} geodesic not found")
            return False
        self._store_geodesic_actor(alternate_key, alternate.polyline, (0.2, 0.7, 0.2), 6.0)
        self._append_message(f"Geodesic {alternate_key} updated")
        return True

    def _store_geodesic_actor(
        self,
        key: str,
        polyline,
        color: tuple[float, float, float],
        line_width: float,
    ) -> None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polyline)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(line_width)
        self._renderer.AddActor(actor)
        self._geodesic_actors[key] = actor
        self._geodesic_lines[key] = polyline

    def _store_aux_actor(
        self,
        key: str,
        polyline,
        color: tuple[float, float, float],
        line_width: float,
    ) -> None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polyline)

        actor = self._aux_actors.get(key)
        if actor is None:
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(line_width)
            self._renderer.AddActor(actor)
            self._aux_actors[key] = actor
        else:
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(line_width)

    def _store_point_actor(
        self,
        key: str,
        point_ids: list[int],
        color: tuple[float, float, float],
        point_size: float,
    ) -> None:
        if self._polydata is None or self._renderer is None:
            return
        if not point_ids:
            self._remove_aux_actor(key)
            return

        points = vtkPoints()
        verts = vtkCellArray()
        for idx, point_id in enumerate(point_ids):
            points.InsertNextPoint(self._polydata.GetPoint(point_id))
            verts.InsertNextCell(1)
            verts.InsertCellPoint(idx)

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetVerts(verts)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = self._aux_actors.get(key)
        if actor is None:
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetRepresentationToPoints()
            actor.GetProperty().SetRenderPointsAsSpheres(True)
            actor.GetProperty().SetLighting(False)
            actor.GetProperty().SetPointSize(point_size)
            self._renderer.AddActor(actor)
            self._aux_actors[key] = actor
        else:
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetRepresentationToPoints()
            actor.GetProperty().SetRenderPointsAsSpheres(True)
            actor.GetProperty().SetLighting(False)
            actor.GetProperty().SetPointSize(point_size)

    def _show_failure_debug(self, debug_points: dict | None) -> None:
        if not debug_points:
            for key in (
                "debug_boundary",
                "debug_seed",
                "debug_opposite",
                "debug_seed_candidate",
                "debug_opposite_seed",
            ):
                self._remove_aux_actor(key)
            return

        boundary_ids = list(debug_points.get("boundary_ids", []))
        seed_id = debug_points.get("seed_id")
        opposite_id = debug_points.get("opposite_id")
        seed_candidate_id = debug_points.get("seed_candidate_id")
        opposite_seed_id = debug_points.get("opposite_seed_id")

        self._store_point_actor("debug_boundary", boundary_ids, (1.0, 0.85, 0.2), 6.0)
        self._store_point_actor(
            "debug_seed",
            [seed_id] if seed_id is not None else [],
            (1.0, 0.2, 0.2),
            12.0,
        )
        self._store_point_actor(
            "debug_opposite",
            [opposite_id] if opposite_id is not None else [],
            (0.2, 0.6, 1.0),
            12.0,
        )
        self._store_point_actor(
            "debug_seed_candidate",
            [seed_candidate_id] if seed_candidate_id is not None else [],
            (0.2, 1.0, 0.4),
            10.0,
        )
        self._store_point_actor(
            "debug_opposite_seed",
            [opposite_seed_id] if opposite_seed_id is not None else [],
            (0.6, 0.2, 1.0),
            10.0,
        )

    def _remove_aux_actor(self, key: str) -> None:
        actor = self._aux_actors.pop(key, None)
        if actor is not None:
            self._renderer.RemoveActor(actor)

    def _create_simple_geodesic(
        self,
        key: str,
        start_key: str,
        end_key: str,
        color: tuple[float, float, float],
        line_width: float,
    ) -> None:
        result = create_simple_geodesic(
            self._polydata,
            self._geo_locator,
            self._landmarks,
            start_key,
            end_key,
        )
        if result is None:
            self._set_error_message(f"{key} geodesic not found")
            return
        self._append_message(f"Geodesic {key} updated")
        self._store_geodesic_actor(key, result.polyline, color, line_width)

    def _remove_geodesic(self, key: str) -> None:
        actor = self._geodesic_actors.pop(key, None)
        if actor is not None:
            self._renderer.RemoveActor(actor)
        self._geodesic_lines.pop(key, None)

    def _build_segment_lut(self) -> vtkLookupTable:
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(10)
        lut.Build()
        lut.SetTableValue(0, 0.6, 0.6, 0.6, 1.0)
        lut.SetTableValue(1, 0.89, 0.10, 0.11, 1.0)
        lut.SetTableValue(2, 0.22, 0.49, 0.72, 1.0)
        lut.SetTableValue(3, 0.30, 0.69, 0.29, 1.0)
        lut.SetTableValue(4, 0.60, 0.31, 0.64, 1.0)
        lut.SetTableValue(5, 1.00, 0.50, 0.00, 1.0)
        lut.SetTableValue(6, 0.65, 0.34, 0.16, 1.0)
        lut.SetTableValue(7, 0.97, 0.51, 0.75, 1.0)
        lut.SetTableValue(8, 0.0, 1.0, 0.0, 1.0)
        lut.SetTableValue(9, 0.45, 0.45, 0.45, 1.0)
        return lut

    def _apply_segment_ids(self, segment_ids) -> None:
        point_data = self._polydata.GetPointData()
        point_data.AddArray(segment_ids)
        point_data.SetScalars(segment_ids)
        self._mesh_mapper.SetScalarRange(0, 9)

        required_keys = {"A", "B", "C", "D", "E", "F", "H", "I", "LAA1", "LAA2"}
        if required_keys.issubset(self._landmarks.keys()):
            unassigned = 0
            for i in range(segment_ids.GetNumberOfTuples()):
                if segment_ids.GetValue(i) == 0:
                    unassigned += 1
            self._segment_lut.SetTableValue(0, 1.0, 1.0, 1.0, 1.0)
            self.statusBar().showMessage(f"Unassigned vertices: {unassigned}")
        else:
            self._segment_lut.SetTableValue(0, 0.6, 0.6, 0.6, 1.0)
            self.statusBar().showMessage("")


def main() -> None:
    os_kind = detect_os()
    if os_kind == "wsl":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    input_file = None
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        input_file = arg
        break

    app = QtWidgets.QApplication(sys.argv)
    if input_file is None:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open VTK Mesh",
            "",
            "VTK Files (*.vtk)",
            options=options,
        )
        if not input_file:
            return

    window = MainWindow(initial_file=input_file)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()