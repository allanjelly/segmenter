import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer


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

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._left_panel = self._build_left_panel()
        self._vtk_widget = QVTKRenderWindowInteractor(central)
        self._vtk_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._vtk_widget.setFocus()
        viewport = self._vtk_widget

        layout.addWidget(self._left_panel)
        layout.addWidget(viewport, stretch=1)
        self.setCentralWidget(central)

        self._renderer = vtkRenderer()
        self._renderer.SetBackground(0.1, 0.1, 0.12)
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(self)
        panel.setFixedWidth(280)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        project_group = QtWidgets.QGroupBox("Project", panel)
        project_layout = QtWidgets.QVBoxLayout(project_group)

        self._mesh_info = QtWidgets.QLabel("No mesh loaded", project_group)
        self._mesh_info.setWordWrap(True)
        project_layout.addWidget(self._mesh_info)

        layout.addWidget(project_group)
        layout.addStretch(1)
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
