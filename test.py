import sys
import os

# On macOS, QVTKRenderWindowInteractor must use QOpenGLWidget as base class
# because WA_PaintOnScreen (used by default QWidget base) is unsupported on Cocoa
if sys.platform == "darwin":
    import vtkmodules.qt
    vtkmodules.qt.QVTKRWIBase = "QOpenGLWidget"

import vtk
from PySide6 import QtCore, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
 
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
 
    def __init__(self, parent = None):
        QtWidgets.QMainWindow.__init__(self, parent)
 
        self.frame = QtWidgets.QFrame()
 
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)
 
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
 
        # Create source
        source = vtk.vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)
 
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
 
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
 
        self.ren.AddActor(actor)
 
        self.ren.ResetCamera()
 
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
 
        self.show()
        self.iren.Initialize()
 
 
if __name__ == "__main__":
    os_kind = detect_os()
    if os_kind == "wsl":
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    elif os_kind == "mac":
        # Don't set QT_QPA_PLATFORM=cocoa - it's the default on macOS
        # and explicitly setting it can interfere with initialization
        # os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
        #os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")    
 
    app = QtWidgets.QApplication(sys.argv)
 
    window = MainWindow()
 
    sys.exit(app.exec())