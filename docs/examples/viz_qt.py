import os
import numpy as np

from fury.window import ShowManager, Scene, snapshot, QtWidgets
from fury.actor import sphere


app = QtWidgets.QApplication([])


class Main(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None)
        self.resize(800, 800)

        self._button = QtWidgets.QPushButton("Hide Sphere", self)
        self._button.clicked.connect(self._on_button_click)
        self.scene = Scene(background=(0, 0, 0, 1))
        self.show_manager = ShowManager(
            scene=self.scene,
            window_type="qt",
            size=None,
            qt_app=app,
            qt_parent=self,
        )

        self._sphere_actor = sphere(
            np.zeros((1, 3)),
            colors=(1, 0, 1, 1),
            radii=15.0,
            phi=48,
            theta=48,
        )
        self.scene.add(self._sphere_actor)

        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self._button)
        layout.addWidget(self.show_manager.window)

    def _on_button_click(self):
        if self._button.text() == "Show Sphere":
            self.scene.add(self._sphere_actor)
            self._button.setText("Hide Sphere")
        else:
            self.scene.remove(self._sphere_actor)
            self._button.setText("Show Sphere")
        self.show_manager.render()


m = Main()
m.setWindowTitle("FURY 2.0: Qt Example")

interactive = True

if "INTERACTIVE_TUTORIALS" in os.environ:
    interactive = os.environ["INTERACTIVE_TUTORIALS"].lower() == "true"


if __name__ == "__main__":
    if interactive:
        m.show()
        m.show_manager.start()
    else:
        snapshot(
            scene=m.scene,
            fname="qt.png",
        )
