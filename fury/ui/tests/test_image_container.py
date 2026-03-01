import numpy as np

from fury import io
from fury.ui import ImageContainer2D


def test_image_container_creation(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img_path = tmp_path / "test.png"
    io.save_image(img, str(img_path))
    container = ImageContainer2D(str(img_path))
    assert container is not None
    assert len(container.actors) == 1
