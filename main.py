import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkCommonDataModel import vtkPointLocator
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkRenderer,
)

from geodesics import build_point_locator, create_pair_geodesics, create_simple_geodesic


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
        self._current_step_index = 0
        self._steps = self._build_steps()

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
        self.statusBar().showMessage("Load a mesh to begin")

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
            "- Mouse wheel: zoom\n\n"
            "- Click mesh: set landmark\n\n"
            "Keyboard:\n"
            "- Space: next step\n"
            "- Esc: previous step",
            controls_group,
        )
        controls_label.setWordWrap(True)
        controls_layout.addWidget(controls_label)
        layout.addWidget(controls_group)

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

        actor = vtkActor()
        actor.SetMapper(mapper)

        if self._mesh_actor is not None:
            self._renderer.RemoveActor(self._mesh_actor)

        self._mesh_actor = actor
        self._renderer.AddActor(actor)
        self._renderer.ResetCamera()
        self._vtk_widget.GetRenderWindow().Render()

    def _update_mesh_info(self, polydata, file_path: str) -> None:
        num_points = polydata.GetNumberOfPoints()
        num_cells = polydata.GetNumberOfCells()
        name = Path(file_path).name
        self._mesh_info.setText(
            f"{name}\nVertices: {num_points}\nCells: {num_cells}"
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
            {"key": "LAA", "label": "LAA: Appendage orifice"},
        ]

    def _populate_steps(self) -> None:
        self._steps_list.blockSignals(True)
        self._steps_list.clear()
        for step in self._steps:
            item = QtWidgets.QListWidgetItem(step["label"])
            item.setData(QtCore.Qt.UserRole, step["key"])
            item.setCheckState(QtCore.Qt.Unchecked)
            self._steps_list.addItem(item)
        self._steps_list.blockSignals(False)
        if self._steps:
            self._steps_list.setCurrentRow(0)
            self._update_step_label()

    def _on_step_changed(self, row: int) -> None:
        if row < 0:
            return
        self._current_step_index = row
        self._update_step_label()

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
        self._landmarks[key] = point
        self._update_landmark_actor(key, point)
        self._mark_step_completed(self._current_step_index)
        self._update_geodesics()
        self._go_next_step()
        self._vtk_widget.GetRenderWindow().Render()

    def _mark_step_completed(self, index: int) -> None:
        item = self._steps_list.item(index)
        if item is None:
            return
        item.setCheckState(QtCore.Qt.Checked)

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

    def _update_geodesics(self) -> None:
        if self._polydata is None or self._geo_locator is None or self._renderer is None:
            return
        required_pairs = {"A", "B", "C", "D", "E"}
        if required_pairs.issubset(self._landmarks.keys()):
            self._remove_geodesic("AB_anterior")
            self._remove_geodesic("AB_posterior")
            self._remove_geodesic("CD_anterior")
            self._remove_geodesic("CD_posterior")
            ab_ok = self._create_pair_geodesics("A", "B", ("A", "B", "C"))
            cd_ok = self._create_pair_geodesics("C", "D", ("A", "C", "D"))
            if ab_ok and cd_ok:
                self.statusBar().showMessage("AB/CD geodesics updated")

        if "A" in self._landmarks and "C" in self._landmarks:
            self._remove_geodesic("AC")
            self._create_simple_geodesic("AC", "A", "C", (0.2, 0.8, 1.0), 6.0)

        if "B" in self._landmarks and "D" in self._landmarks:
            self._remove_geodesic("BD")
            self._create_simple_geodesic("BD", "B", "D", (0.2, 0.8, 1.0), 6.0)

        if "C" in self._landmarks and "E" in self._landmarks:
            self._remove_geodesic("CE")
            self._create_simple_geodesic("CE", "C", "E", (0.7, 0.9, 0.3), 6.0)

        if "A" in self._landmarks and "F" in self._landmarks:
            self._remove_geodesic("AF")
            self._create_simple_geodesic("AF", "A", "F", (0.7, 0.9, 0.3), 6.0)

        if "B" in self._landmarks and "H" in self._landmarks:
            self._remove_geodesic("BH")
            self._create_simple_geodesic("BH", "B", "H", (0.7, 0.9, 0.3), 6.0)

        if "D" in self._landmarks and "I" in self._landmarks:
            self._remove_geodesic("DI")
            self._create_simple_geodesic("DI", "D", "I", (0.7, 0.9, 0.3), 6.0)
                     
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
        if alternate is None:
            self.statusBar().showMessage(f"Alternate {start_key}{end_key} geodesic not found")
            return False
        self._store_geodesic_actor(alternate_key, alternate.polyline, (0.2, 0.7, 0.2), 6.0)
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
            self.statusBar().showMessage(f"{key} geodesic not found")
            return
        self._store_geodesic_actor(key, result.polyline, color, line_width)

    def _remove_geodesic(self, key: str) -> None:
        actor = self._geodesic_actors.pop(key, None)
        if actor is not None:
            self._renderer.RemoveActor(actor)


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