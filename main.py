import os
import sys
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPointLocator, vtkPolyData
from vtkmodules.vtkCommonCore import vtkIntArray, vtkPoints
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
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
        print(f"[DEBUG] MainWindow.__init__ started, platform={sys.platform}", flush=True)
        self.setWindowTitle("Segmenter")
        self.resize(1200, 800)
        self._mesh_actor = None
        self._mesh_mapper = None
        self._overlay_actor = None
        self._overlay_mapper = None
        self._overlay_polydata = None
        self._initial_file = initial_file
        self._pending_file = None
        self._polydata = None
        self._mesh_file_path = None
        self._last_segment_ids = None
        self._last_segment_error = None
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
        self._deferred_mesh_path = None  # For macOS deferred loading

        print(f"[DEBUG] Creating central widget", flush=True)
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        print(f"[DEBUG] Building left panel", flush=True)
        self._left_panel = self._build_left_panel()
        print(f"[DEBUG] Creating VTK widget", flush=True)
        self._vtk_widget = QVTKRenderWindowInteractor(central)
        print(f"[DEBUG] VTK widget created", flush=True)
        self._vtk_widget.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._vtk_widget.setFocus()

        layout.addWidget(self._left_panel)
        layout.addWidget(self._vtk_widget, stretch=1)
        self.setCentralWidget(central)

        # On macOS, defer all VTK operations until window is shown
        if sys.platform != "darwin":
            print(f"[DEBUG] Setting up VTK render window (non-macOS)", flush=True)
            render_window = self._vtk_widget.GetRenderWindow()
            render_window.AddRenderer(self._renderer)
        else:
            print(f"[DEBUG] Deferring VTK setup for macOS", flush=True)
        
        print(f"[DEBUG] Setting up shortcuts", flush=True)
        self._setup_shortcuts()
        self.statusBar().showMessage("")
        print(f"[DEBUG] MainWindow.__init__ completed", flush=True)

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
        self._overlay_button = QtWidgets.QPushButton("Load epicardium", mesh_group)
        self._overlay_button.clicked.connect(self._select_overlay_mesh)
        mesh_layout.addWidget(self._overlay_button)
        self._overlay_toggle = QtWidgets.QCheckBox("Show epicardium", mesh_group)
        self._overlay_toggle.setChecked(True)
        self._overlay_toggle.setEnabled(False)
        self._overlay_toggle.toggled.connect(self._toggle_overlay_visibility)
        mesh_layout.addWidget(self._overlay_toggle)
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
        self._delete_landmark_button = QtWidgets.QPushButton("Delete landmark", landmarks_group)
        self._delete_landmark_button.clicked.connect(self._delete_current_landmark)
        landmarks_layout.addWidget(self._delete_landmark_button)
        self._calculate_regions_button = QtWidgets.QPushButton("Calculate regions", landmarks_group)
        self._calculate_regions_button.clicked.connect(self._calculate_regions)
        landmarks_layout.addWidget(self._calculate_regions_button)

        layout.addWidget(landmarks_group)

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

        self._save_results_button = QtWidgets.QPushButton("Save results", panel)
        self._save_results_button.clicked.connect(self._save_results)
        layout.addWidget(self._save_results_button)

        layout.addStretch(1)

        self._populate_steps()
        return panel

    def showEvent(self, event: QtCore.QEvent) -> None:
        print(f"[DEBUG] showEvent called", flush=True)
        super().showEvent(event)
        if self._initial_file:
            print(f"[DEBUG] Initial file: {self._initial_file}", flush=True)
            self._pending_file = self._initial_file
        # Delay VTK initialization to ensure window is fully visible on macOS
        if self._vtk_widget is not None:
            if sys.platform == "darwin":
                print(f"[DEBUG] Scheduling VTK init for macOS (200ms delay)", flush=True)
                QtCore.QTimer.singleShot(200, self._initialize_vtk)
            else:
                print(f"[DEBUG] Scheduling VTK init immediately", flush=True)
                QtCore.QTimer.singleShot(0, self._initialize_vtk)

    def _initialize_vtk(self) -> None:
        print(f"[DEBUG] _initialize_vtk called", flush=True)
        try:
            if self._vtk_widget is None:
                print(f"[DEBUG] vtk_widget is None, returning", flush=True)
                return
            
            # On macOS, use a multi-step initialization to keep event loop alive
            if sys.platform == "darwin":
                print(f"[DEBUG] Starting multi-step macOS initialization", flush=True)
                self._macos_init_step_1()
                return
            
            # On macOS, ensure the widget and window are fully laid out
            if sys.platform == "darwin":
                print(f"[DEBUG] Ensuring widget is ready on macOS", flush=True)
                self._vtk_widget.setVisible(True)
                self._vtk_widget.setFocus()
                # Force geometry update
                self.centralWidget().updateGeometry()
                print(f"[DEBUG] Widget size: {self._vtk_widget.width()}x{self._vtk_widget.height()}", flush=True)
                
            self._append_message("Initializing VTK...")
            print(f"[DEBUG] Getting render window", flush=True)
            
            # Get render window before Initialize to set properties
            render_window = self._vtk_widget.GetRenderWindow()
            print(f"[DEBUG] Got render window", flush=True)
            
            # On macOS, set up renderer now (deferred from __init__)
            if sys.platform == "darwin":
                print(f"[DEBUG] Setting up macOS renderer...", flush=True)
                self._append_message("Setting up macOS renderer...")
                render_window.SetOffScreenRendering(0)
                print(f"[DEBUG] SetOffScreenRendering done", flush=True)
                render_window.SetMultiSamples(0)
                print(f"[DEBUG] SetMultiSamples done", flush=True)
            
            print(f"[DEBUG] Calling vtk_widget.Initialize()", flush=True)
            self._vtk_widget.Initialize()
            print(f"[DEBUG] Initialize() completed", flush=True)
            
            # On macOS, add renderer AFTER Initialize
            if sys.platform == "darwin":
                print(f"[DEBUG] Adding renderer after Initialize on macOS", flush=True)
                render_window.AddRenderer(self._renderer)
                print(f"[DEBUG] AddRenderer done", flush=True)
                # Ensure the render window is visible
                print(f"[DEBUG] Checking widget visibility: {self._vtk_widget.isVisible()}", flush=True)
                print(f"[DEBUG] Widget size: {self._vtk_widget.width()}x{self._vtk_widget.height()}", flush=True)
                if not self._vtk_widget.isVisible():
                    print(f"[DEBUG] WARNING: VTK widget not visible!", flush=True)
                    self._vtk_widget.show()
                    print(f"[DEBUG] Called show() on widget", flush=True)
            
            self._append_message("VTK initialized")
            
            print(f"[DEBUG] Getting interactor", flush=True)
            interactor = render_window.GetInteractor()
            print(f"[DEBUG] Got interactor: {interactor is not None}", flush=True)
            if interactor is not None:
                print(f"[DEBUG] Setting interactor style", flush=True)
                interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
                print(f"[DEBUG] Adding observer", flush=True)
                interactor.AddObserver(
                    "LeftButtonPressEvent",
                    self._on_left_button_press,
                )
                print(f"[DEBUG] Observer added", flush=True)
            
            print(f"[DEBUG] Calling render_window.Render()", flush=True)
            # On macOS, skip the initial render to avoid blocking the event loop
            # The first render will happen when the mesh is displayed
            if sys.platform != "darwin":
                render_window.Render()
                print(f"[DEBUG] Render() completed", flush=True)
            else:
                print(f"[DEBUG] Skipping initial Render() on macOS", flush=True)
            
            self._append_message("Renderer ready")
            
            print(f"[DEBUG] Checking pending file: {self._pending_file}", flush=True)
            if self._pending_file:
                file_path = self._pending_file
                self._pending_file = None
                print(f"[DEBUG] About to load mesh: {file_path}", flush=True)
                self._append_message(f"Loading mesh: {Path(file_path).name}")
                # On macOS, defer mesh loading to avoid blocking
                if sys.platform == "darwin":
                    print(f"[DEBUG] Scheduling deferred mesh load on macOS", flush=True)
                    self._deferred_mesh_path = file_path
                    QtCore.QTimer.singleShot(50, self._deferred_load_mesh_callback)
                else:
                    self.load_mesh(file_path)
                    print(f"[DEBUG] load_mesh() returned", flush=True)
            else:
                print(f"[DEBUG] No pending file to load", flush=True)
            print(f"[DEBUG] _initialize_vtk completed successfully", flush=True)
        except Exception as e:
            print(f"[DEBUG] Exception in _initialize_vtk: {str(e)}", flush=True)
            self._append_error(f"VTK initialization error: {str(e)}")
            self._append_error(traceback.format_exc())
        
        print(f"[DEBUG] About to return from _initialize_vtk", flush=True)
        
        # Critical test: Schedule a simple timer to see if Qt event loop works at all
        if sys.platform == "darwin":
            print(f"[DEBUG] Scheduling test timer immediately after return", flush=True)
            QtCore.QTimer.singleShot(10, self._test_event_loop)
            print(f"[DEBUG] Test timer scheduled", flush=True)
    
    def _test_event_loop(self) -> None:
        """Test if Qt event loop is running"""
        print(f"[DEBUG] *** TEST TIMER FIRED - EVENT LOOP IS WORKING! ***", flush=True)
        self._append_message("Event loop test passed")
    
    def _macos_init_step_1(self) -> None:
        """Step 1: Setup widget and get render window"""
        print(f"[DEBUG] macOS init step 1", flush=True)
        self._append_message("Initializing VTK (step 1)...")
        self._vtk_widget.setVisible(True)
        self._vtk_widget.setFocus()
        self.centralWidget().updateGeometry()
        print(f"[DEBUG] Widget size: {self._vtk_widget.width()}x{self._vtk_widget.height()}", flush=True)
        
        render_window = self._vtk_widget.GetRenderWindow()
        render_window.SetOffScreenRendering(0)
        render_window.SetMultiSamples(0)
        print(f"[DEBUG] Render window configured, scheduling step 2", flush=True)
        QtCore.QTimer.singleShot(50, self._macos_init_step_2)
    
    def _macos_init_step_2(self) -> None:
        """Step 2: Add renderer WITHOUT calling Initialize - let it auto-initialize"""
        print(f"[DEBUG] macOS init step 2 - SKIP manual Initialize(), auto-init instead", flush=True)
        self._append_message("Initializing VTK (step 2)...")
        
        # Don't call Initialize() manually on macOS - it breaks the event loop
        # Instead, just set up the renderer and let QVTKRenderWindowInteractor auto-initialize
        render_window = self._vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self._renderer)
        print(f"[DEBUG] Renderer added, scheduling step 3", flush=True)
        QtCore.QTimer.singleShot(50, self._macos_init_step_3)
    
    def _macos_init_step_3(self) -> None:
        """Step 3: Setup interactor"""
        print(f"[DEBUG] macOS init step 3 - setting up interactor", flush=True)
        self._append_message("Initializing VTK (step 3)...")
        render_window = self._vtk_widget.GetRenderWindow()
        
        # Get or create interactor
        interactor = render_window.GetInteractor()
        print(f"[DEBUG] Got interactor: {interactor is not None}", flush=True)
        if interactor is not None:
            # DON'T call interactor.Initialize() - it blocks the event loop on macOS!
            # QVTKRenderWindowInteractor handles initialization automatically
            print(f"[DEBUG] Setting interactor style", flush=True)
            interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            print(f"[DEBUG] Adding observer", flush=True)
            interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press)
            print(f"[DEBUG] Observer added", flush=True)
        
        print(f"[DEBUG] macOS init step 3 done, scheduling step 4", flush=True)
        QtCore.QTimer.singleShot(50, self._macos_init_step_4)
    
    def _macos_init_step_4(self) -> None:
        """Step 4: Load mesh if pending"""
        print(f"[DEBUG] macOS init step 4 - final step", flush=True)
        self._append_message("VTK initialized")
        self._append_message("Renderer ready")
        
        if self._pending_file:
            file_path = self._pending_file
            self._pending_file = None
            print(f"[DEBUG] Loading pending mesh: {file_path}", flush=True)
            self._append_message(f"Loading mesh: {Path(file_path).name}")
            QtCore.QTimer.singleShot(50, lambda: self._deferred_load_mesh(file_path))
        else:
            print(f"[DEBUG] No pending file", flush=True)
            self._append_message("Ready")
        
        print(f"[DEBUG] macOS initialization complete", flush=True)
    
    def _deferred_load_mesh_callback(self) -> None:
        """Timer callback for deferred mesh loading on macOS"""
        print(f"[DEBUG] _deferred_load_mesh_callback fired!", flush=True)
        if self._deferred_mesh_path:
            file_path = self._deferred_mesh_path
            self._deferred_mesh_path = None
            self._deferred_load_mesh(file_path)
        else:
            print(f"[DEBUG] No deferred mesh path found", flush=True)
    
    def _deferred_load_mesh(self, file_path: str) -> None:
        """Load mesh in a deferred callback to keep event loop responsive on macOS"""
        print(f"[DEBUG] _deferred_load_mesh called for: {file_path}", flush=True)
        self.load_mesh(file_path)
        print(f"[DEBUG] _deferred_load_mesh completed", flush=True)
        self._append_message("Application ready")
    
    def _deferred_render(self) -> None:
        """Deferred render call for macOS - called after window is fully ready"""
        print(f"[DEBUG] _deferred_render called (delayed init complete)", flush=True)
        if self._vtk_widget is not None:
            try:
                # Ensure widget is visible and has valid size
                if not self._vtk_widget.isVisible():
                    print(f"[DEBUG] WARNING: Widget not visible, calling show()", flush=True)
                    self._vtk_widget.show()
                
                print(f"[DEBUG] Widget geometry: {self._vtk_widget.width()}x{self._vtk_widget.height()}", flush=True)
                print(f"[DEBUG] Widget visible: {self._vtk_widget.isVisible()}", flush=True)
                
                # On macOS, DON'T call Render() - even with delays it breaks the event loop
                # Instead, just mark the widget as dirty and let Qt's paint system handle it
                print(f"[DEBUG] Scheduling repaint via update() - no direct Render() call", flush=True)
                self._vtk_widget.update()
                print(f"[DEBUG] Update scheduled", flush=True)
                
                # Schedule a test to verify event loop is still running after return
                QtCore.QTimer.singleShot(100, lambda: print("[DEBUG] *** EVENT LOOP STILL ALIVE AFTER RENDER ***", flush=True))
                
                self.activateWindow()
                self.raise_()
                print(f"[DEBUG] _deferred_render completed, returning to event loop", flush=True)
            except Exception as e:
                print(f"[DEBUG] Render exception: {e}", flush=True)
                self._append_error(f"Render error: {e}")

    def load_mesh(self, file_path: str) -> None:
        print(f"[DEBUG] load_mesh called: {file_path}", flush=True)
        try:
            self._append_message(f"Loading mesh from: {file_path}")
            print(f"[DEBUG] About to read VTK polydata", flush=True)
            polydata = self._read_vtk_polydata(Path(file_path))
            print(f"[DEBUG] read_vtk_polydata returned: {polydata is not None}", flush=True)
            if polydata is None:
                self._append_error("Failed to read VTK polydata - file may be corrupt or empty")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Load Failed",
                    "Failed to read VTK polydata.",
                )
                return

            print(f"[DEBUG] Setting polydata and building locators", flush=True)
            self._polydata = polydata
            self._mesh_file_path = file_path
            self._point_locator = vtkPointLocator()
            self._point_locator.SetDataSet(polydata)
            print(f"[DEBUG] BuildLocator...", flush=True)
            self._point_locator.BuildLocator()
            print(f"[DEBUG] build_point_locator...", flush=True)
            self._geo_locator = build_point_locator(polydata)
            print(f"[DEBUG] Locators built", flush=True)

            print(f"[DEBUG] Displaying polydata...", flush=True)
            self._display_polydata(polydata)
            print(f"[DEBUG] Polydata displayed", flush=True)
            print(f"[DEBUG] Updating mesh info...", flush=True)
            self._update_mesh_info(polydata, file_path)
            print(f"[DEBUG] Mesh info updated", flush=True)
            self._append_message(f"Mesh loaded: {Path(file_path).name}")
            print(f"[DEBUG] load_mesh completed successfully", flush=True)
        except Exception as e:
            self._append_error(f"Error loading mesh: {str(e)}")
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Error loading mesh: {str(e)}",
            )

    def _select_overlay_mesh(self) -> None:
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Reference Mesh",
            "",
            "VTK Files (*.vtk)",
            options=options,
        )
        if not file_path:
            return
        self._load_overlay_mesh(file_path)

    def _load_overlay_mesh(self, file_path: str) -> None:
        polydata = self._read_vtk_polydata(Path(file_path))
        if polydata is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Load Failed",
                "Failed to read VTK polydata.",
            )
            return
        self._display_overlay_polydata(polydata)
        self._overlay_polydata = polydata
        if self._overlay_toggle is not None:
            self._overlay_toggle.setEnabled(True)
            self._overlay_toggle.setChecked(True)
        self._append_message(f"Reference mesh loaded: {Path(file_path).name}")

    def _read_vtk_polydata(self, path: Path):
        reader = vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        polydata = reader.GetOutput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return None
        return polydata

    def _display_polydata(self, polydata) -> None:
        print(f"[DEBUG] _display_polydata called", flush=True)
        try:
            print(f"[DEBUG] Creating mapper and actor", flush=True)
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
                print(f"[DEBUG] Removing old mesh actor", flush=True)
                self._renderer.RemoveActor(self._mesh_actor)

            print(f"[DEBUG] Adding mesh actor to renderer", flush=True)
            self._mesh_actor = actor
            self._mesh_mapper = mapper
            self._renderer.AddActor(actor)
            print(f"[DEBUG] Resetting camera", flush=True)
            self._renderer.ResetCamera()
            print(f"[DEBUG] Camera reset", flush=True)
            
            if self._vtk_widget is not None:
                print(f"[DEBUG] About to render from _display_polydata", flush=True)
                # On macOS, call Render() with a delay to ensure window is ready
                if sys.platform == "darwin":
                    print(f"[DEBUG] Scheduling delayed render on macOS (500ms)", flush=True)
                    QtCore.QTimer.singleShot(500, self._deferred_render)
                else:
                    self._vtk_widget.GetRenderWindow().Render()
                    print(f"[DEBUG] Render() completed in _display_polydata", flush=True)
                self._append_message("Mesh displayed")
        except Exception as e:
            print(f"[DEBUG] Exception in _display_polydata: {str(e)}", flush=True)
            self._append_error(f"Display error: {str(e)}")
            self._append_error(traceback.format_exc())

    def _display_overlay_polydata(self, polydata) -> None:
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarVisibility(False)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        actor.GetProperty().SetOpacity(0.25)
        actor.SetPickable(False)

        if self._overlay_actor is not None:
            self._renderer.RemoveActor(self._overlay_actor)

        self._overlay_actor = actor
        self._overlay_mapper = mapper
        self._renderer.AddActor(actor)
        self._vtk_widget.GetRenderWindow().Render()

    def _toggle_overlay_visibility(self, visible: bool) -> None:
        if self._overlay_actor is None:
            return
        self._overlay_actor.SetVisibility(1 if visible else 0)
        if self._vtk_widget is not None:
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
            {"key": "A", "label": "A - LSPA"},
            {"key": "B", "label": "B - LIPA"},
            {"key": "C", "label": "C - RSPA"},
            {"key": "D", "label": "D - RIPA"},      
            {"key": "E", "label": "E - MA 9 o'clock"},
            {"key": "F", "label": "F - MA 1 o'clock"},
            {"key": "H", "label": "H - MA 4 o'clock"},
            {"key": "I", "label": "I - MA 7 o'clock"},                  
            {"key": "LAA1", "label": "LAA1 - point 1"},
            {"key": "LAA2", "label": "LAA2 - point 2"},            
            {"key": "A1", "label": "  A1 - LSPV sup dist"},
            {"key": "A2", "label": "  A2 - LSPV inf dist"},
            {"key": "B1", "label": "  B1 - LIPV sup dist"},
            {"key": "B2", "label": "  B2 - LIPV inf dist"},            
            {"key": "C1", "label": "  C1 - RSPV sup dist"},
            {"key": "C2", "label": "  C2 - RSPV inf dist"},
            {"key": "D1", "label": "  D1 - RIPV sup dist"},
            {"key": "D2", "label": "  D2 - RIPV inf dist"},                                    
            
            {"key": "X1", "label": "    X1: LAA/LSPV cut point 1"},
            {"key": "X2", "label": "    X2: LAA/LSPV cut point 2"},
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

    def _append_error(self, message: str) -> None:
        if self._error_box is not None and message:
            self._error_box.appendPlainText(message)

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
        self._last_segment_ids = segment_ids
        self._last_segment_error = error_message
        self._show_failure_debug(debug_points)
        if segment_ids is None:
            return
        self._apply_segment_ids(segment_ids)
        self._append_message("Regions calculated")
        if self._vtk_widget is not None:
            self._vtk_widget.GetRenderWindow().Render()

    def _save_results(self) -> None:
        if self._polydata is None or self._overlay_polydata is None:
            self._set_error_message("Load both endocardium and epicardium meshes before saving")
            return
        if not self._mesh_file_path:
            self._set_error_message("Missing endocardium file path; reload the mesh before saving")
            return

        segment_ids = self._last_segment_ids
        segment_error = self._last_segment_error
        all_segments_ok = segment_ids is not None and not segment_error

        output_dir = Path(self._mesh_file_path).parent
        base_name = Path(self._mesh_file_path).stem
        suffix = "_results.vtk" if all_segments_ok else "_error.vtk"
        output_path = output_dir / f"{base_name}{suffix}"

        try:
            endo_copy = vtkPolyData()
            endo_copy.DeepCopy(self._polydata)
            region_end = self._build_region_array(endo_copy, segment_ids)
            endo_copy.GetPointData().AddArray(region_end)
            endo_copy.GetPointData().SetScalars(region_end)

            epi_copy = vtkPolyData()
            epi_copy.DeepCopy(self._overlay_polydata)
            region_epi = vtkIntArray()
            region_epi.SetName("Regions")
            region_epi.SetNumberOfComponents(1)
            region_epi.SetNumberOfTuples(epi_copy.GetNumberOfPoints())
            for i in range(epi_copy.GetNumberOfPoints()):
                region_epi.SetValue(i, 0)
            epi_copy.GetPointData().AddArray(region_epi)
            epi_copy.GetPointData().SetScalars(region_epi)

            append = vtkAppendPolyData()
            append.AddInputData(endo_copy)
            append.AddInputData(epi_copy)
            append.Update()

            writer = vtkPolyDataWriter()
            writer.SetFileName(str(output_path))
            writer.SetInputData(append.GetOutput())
            if writer.Write() != 1:
                raise RuntimeError("VTK writer reported failure")
        except Exception as exc:
            self._set_error_message(f"Save failed: {exc}")
            return

        self._append_message(f"Saved results: {output_path}")

    def _build_region_array(self, polydata: vtkPolyData, segment_ids) -> vtkIntArray:
        region = vtkIntArray()
        region.SetName("Regions")
        region.SetNumberOfComponents(1)
        region.SetNumberOfTuples(polydata.GetNumberOfPoints())

        source = segment_ids
        if source is None:
            source = polydata.GetPointData().GetArray("SegmentId")

        if source is None:
            for i in range(polydata.GetNumberOfPoints()):
                region.SetValue(i, 0)
            return region

        for i in range(polydata.GetNumberOfPoints()):
            region.SetValue(i, int(source.GetValue(i)))
        return region

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

    def _mark_step_incomplete(self, index: int) -> None:
        item = self._steps_list.item(index)
        if item is None:
            return
        self._updating_steps = True
        item.setData(QtCore.Qt.UserRole + 1, False)
        item.setCheckState(QtCore.Qt.Unchecked)
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

    def _delete_current_landmark(self) -> None:
        if not self._steps:
            return
        step = self._steps[self._current_step_index]
        key = step["key"]
        if key not in self._landmarks:
            return
        self._landmarks.pop(key, None)

        actor = self._landmark_actors.pop(key, None)
        if actor is not None and self._renderer is not None:
            self._renderer.RemoveActor(actor)

        self._remove_dependent_geodesics(key)
        self._mark_step_incomplete(self._current_step_index)
        if self._vtk_widget is not None:
            self._vtk_widget.GetRenderWindow().Render()

    def _remove_dependent_geodesics(self, landmark_key: str) -> None:
        dependencies = (
            ("AB_anterior", {"A", "B", "C", "E"}),
            ("AB_posterior", {"A", "B", "C", "E"}),
            ("CD_anterior", {"A", "C", "D", "E"}),
            ("CD_posterior", {"A", "C", "D", "E"}),
            ("AC", {"A", "C"}),
            ("BD", {"B", "D"}),
            ("CE", {"C", "E"}),
            ("AF", {"A", "F"}),
            ("BH", {"B", "H"}),
            ("DI", {"D", "I"}),
            ("LAA1_LAA2_anterior", {"LAA1", "LAA2", "D", "F"}),
            ("LAA1_LAA2_posterior", {"LAA1", "LAA2", "D", "F"}),
            ("X1_X2_anterior", {"X1", "X2", "D", "F"}),
            ("X1_X2_posterior", {"X1", "X2", "D", "F"}),
            ("EF_aniso", {"E", "F", "H", "I"}),
            ("FH_aniso", {"E", "F", "H", "I"}),
            ("HI_aniso", {"E", "F", "H", "I"}),
            ("IE_aniso", {"E", "F", "H", "I"}),
        )
        for geodesic_key, required in dependencies:
            if landmark_key in required:
                self._remove_geodesic(geodesic_key)

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
                updated, ab_ok = self._update_landmark_pair_geodesics(
                    "A",
                    "B",
                    ("A", "B", "C"),
                    "AB",
                    primary_color=(0.9, 0.6, 0.1),
                    alternate_color=(0.2, 0.7, 0.2),
                    line_width=6.0,
                    anterior_ref_key="E",
                    plane_origin_key="A",
                )
                changed_geodesics.update(updated)
            if cd_changed:
                updated, cd_ok = self._update_landmark_pair_geodesics(
                    "C",
                    "D",
                    ("A", "C", "D"),
                    "CD",
                    primary_color=(0.9, 0.6, 0.1),
                    alternate_color=(0.2, 0.7, 0.2),
                    line_width=6.0,
                    anterior_ref_key="E",
                    plane_origin_key="A",
                )
                changed_geodesics.update(updated)
            if ab_changed and cd_changed and ab_ok and cd_ok:
                self._append_message("AB/CD geodesics updated")

        if (
            "A" in self._landmarks
            and "C" in self._landmarks
            and ({"A", "C"} & changed_landmarks or "AC" not in self._geodesic_lines)
        ):
            if self._update_simple_geodesic("AC", "A", "C", (0.2, 0.8, 1.0), 6.0):
                changed_geodesics.add("AC")

        if (
            "B" in self._landmarks
            and "D" in self._landmarks
            and ({"B", "D"} & changed_landmarks or "BD" not in self._geodesic_lines)
        ):
            if self._update_simple_geodesic("BD", "B", "D", (0.2, 0.8, 1.0), 6.0):
                changed_geodesics.add("BD")

        if (
            "C" in self._landmarks
            and "E" in self._landmarks
            and ({"C", "E"} & changed_landmarks or "CE" not in self._geodesic_lines)
        ):
            if self._update_simple_geodesic("CE", "C", "E", (0.7, 0.9, 0.3), 6.0):
                changed_geodesics.add("CE")

        if (
            "A" in self._landmarks
            and "F" in self._landmarks
            and ({"A", "F"} & changed_landmarks or "AF" not in self._geodesic_lines)
        ):
            if self._update_simple_geodesic("AF", "A", "F", (0.7, 0.9, 0.3), 6.0):
                changed_geodesics.add("AF")

        if (
            "B" in self._landmarks
            and "H" in self._landmarks
            and ({"B", "H"} & changed_landmarks or "BH" not in self._geodesic_lines)
        ):
            if self._update_simple_geodesic("BH", "B", "H", (0.7, 0.9, 0.3), 6.0):
                changed_geodesics.add("BH")

        if (
            "D" in self._landmarks
            and "I" in self._landmarks
            and ({"D", "I"} & changed_landmarks or "DI" not in self._geodesic_lines)
        ):
            if self._update_simple_geodesic("DI", "D", "I", (0.7, 0.9, 0.3), 6.0):
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
            updated, ok = self._update_landmark_pair_geodesics(
                "LAA1",
                "LAA2",
                ("LAA1", "LAA2", "D"),
                "LAA1_LAA2",
                primary_color=(0.9, 0.6, 0.1),
                alternate_color=(0.2, 0.7, 0.2),
                line_width=6.0,
                anterior_ref_key="F",
                plane_origin_key="D",
            )
            changed_geodesics.update(updated)

# A1=A2
        if (
            "A1" in self._landmarks
            and "A2" in self._landmarks
            and "D" in self._landmarks
            and "F" in self._landmarks
            and (
                {"A1", "A2", "D", "F"} & changed_landmarks
                or not {"A1_A2_anterior", "A1_A2_posterior"}.issubset(
                    self._geodesic_lines.keys()
                )
            )
        ):
            updated, ok = self._update_landmark_pair_geodesics(
                "A1",
                "A2",
                ("A1", "A2", "D"),
                "A1_A2",
                primary_color=(0.9, 0.6, 0.1),
                alternate_color=(0.2, 0.7, 0.2),
                line_width=6.0,
                anterior_ref_key="F",
                plane_origin_key="D",
            )
            changed_geodesics.update(updated)

# B1=B2
        if (
            "B1" in self._landmarks
            and "B2" in self._landmarks
            and "D" in self._landmarks
            and "F" in self._landmarks
            and (
                {"B1", "B2", "D", "F"} & changed_landmarks
                or not {"B1_B2_anterior", "B1_B2_posterior"}.issubset(
                    self._geodesic_lines.keys()
                )
            )
        ):
            updated, ok = self._update_landmark_pair_geodesics(
                "B1",
                "B2",
                ("B1", "B2", "D"),
                "B1_B2",
                primary_color=(0.9, 0.6, 0.1),
                alternate_color=(0.2, 0.7, 0.2),
                line_width=6.0,
                anterior_ref_key="F",
                plane_origin_key="D",
            )
            changed_geodesics.update(updated)            

# C1=C2
        if (
            "C1" in self._landmarks
            and "C2" in self._landmarks
            and "A" in self._landmarks
            and "E" in self._landmarks
            and (
                {"C1", "C2", "A", "E"} & changed_landmarks
                or not {"C1_C2_anterior", "C1_C2_posterior"}.issubset(
                    self._geodesic_lines.keys()
                )
            )
        ):
            updated, ok = self._update_landmark_pair_geodesics(
                "C1",
                "C2",
                ("C1", "C2", "A"),
                "C1_C2",
                primary_color=(0.9, 0.6, 0.1),
                alternate_color=(0.2, 0.7, 0.2),
                line_width=6.0,
                anterior_ref_key="E",
                plane_origin_key="A",
            )
            changed_geodesics.update(updated)

# D1=D2
        if (
            "D1" in self._landmarks
            and "D2" in self._landmarks
            and "A" in self._landmarks
            and "E" in self._landmarks
            and (
                {"D1", "D2", "A", "E"} & changed_landmarks
                or not {"D1_D2_anterior", "D1_D2_posterior"}.issubset(
                    self._geodesic_lines.keys()
                )
            )
        ):
            updated, ok = self._update_landmark_pair_geodesics(
                "D1",
                "D2",
                ("D1", "D2", "A"),
                "D1_D2",
                primary_color=(0.9, 0.6, 0.1),
                alternate_color=(0.2, 0.7, 0.2),
                line_width=6.0,
                anterior_ref_key="E",
                plane_origin_key="A",
            )
            changed_geodesics.update(updated)

        if "X1" not in self._landmarks or "X2" not in self._landmarks:
            self._remove_geodesic("X1_X2_anterior")
            self._remove_geodesic("X1_X2_posterior")
        elif (
            {"X1", "X2", "D", "F"} & changed_landmarks
            or not {"X1_X2_anterior", "X1_X2_posterior"}.issubset(
                self._geodesic_lines.keys()
            )
        ):
            updated, ok = self._update_landmark_pair_geodesics(
                "X1",
                "X2",
                ("X1", "X2", "D"),
                "X1_X2",
                primary_color=(0.8, 0.8, 0.2),
                alternate_color=(0.2, 0.8, 0.8),
                line_width=6.0,
                anterior_ref_key="F",
                plane_origin_key="D",
            )
            changed_geodesics.update(updated)

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
                penalty_strength = 2.0
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


    def _update_landmark_pair_geodesics(
        self,
        start_key: str,
        end_key: str,
        plane_keys: tuple[str, str, str],
        name_prefix: str,
        primary_color: tuple[float, float, float],
        alternate_color: tuple[float, float, float],
        line_width: float,
        anterior_ref_key: str,
        plane_origin_key: str,
    ) -> tuple[set[str], bool]:
        updated: set[str] = set()
        anterior_key = f"{name_prefix}_anterior"
        posterior_key = f"{name_prefix}_posterior"
        self._remove_geodesic(anterior_key)
        self._remove_geodesic(posterior_key)

        primary_key, primary, _alternate_key, alternate = create_pair_geodesics(
            self._polydata,
            self._geo_locator,
            self._landmarks,
            start_key,
            end_key,
            plane_keys,
            anterior_ref_key=anterior_ref_key,
            plane_origin_key=plane_origin_key,
        )
        if primary_key.endswith("_anterior"):
            resolved_primary = anterior_key
            resolved_alternate = posterior_key
        else:
            resolved_primary = posterior_key
            resolved_alternate = anterior_key

        self._store_geodesic_actor(resolved_primary, primary.polyline, primary_color, line_width)
        self._append_message(f"Geodesic {resolved_primary} updated")
        updated.add(resolved_primary)

        if alternate is None:
            self._set_error_message(f"Alternate {name_prefix} geodesic not found")
            return updated, False

        self._store_geodesic_actor(resolved_alternate, alternate.polyline, alternate_color, line_width)
        self._append_message(f"Geodesic {resolved_alternate} updated")
        updated.add(resolved_alternate)
        return updated, True

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

    def _update_simple_geodesic(
        self,
        key: str,
        start_key: str,
        end_key: str,
        color: tuple[float, float, float],
        line_width: float,
    ) -> bool:
        self._remove_geodesic(key)
        self._create_simple_geodesic(key, start_key, end_key, color, line_width)
        return key in self._geodesic_lines

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
    elif os_kind == "mac":
        # Explicitly set the Qt platform plugin for macOS
        os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
        # Critical macOS settings for VTK/Qt integration
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

    input_file = None
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        input_file = arg
        break

    app = QtWidgets.QApplication(sys.argv)
    
    # On macOS, we need to create the window first before showing file dialog
    # to ensure proper Qt event loop initialization
    if input_file is None:
        if os_kind == "mac":
            # Create a temporary window to initialize Qt properly on Mac
            temp_window = QtWidgets.QWidget()
            temp_window.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint)
            temp_window.resize(0, 0)
            temp_window.show()
            temp_window.hide()
            app.processEvents()
        
        options = QtWidgets.QFileDialog.Options()
        # Use native dialog on Mac for better integration
        if os_kind != "mac":
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