import os
import pytest
from fury.io import load_rectilinear_grid


def test_load_rectilinear_grid_file_not_found():
    """Should raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_rectilinear_grid("nonexistent.vtr")


def test_load_rectilinear_grid_wrong_extension(tmp_path):
    """Should raise FileNotFoundError for non-existent file."""
    fake_file = tmp_path / "test.txt"
    fake_file.write_text("not a vtr file")
    with pytest.raises(Exception):
        load_rectilinear_grid(str(fake_file))


def test_load_rectilinear_grid_returns_dict(tmp_path):
    """Should return dict with correct keys if vtk is available."""
    pytest.importorskip("vtk")
    # This test requires an actual .vtr file
    # Skipped if no sample file available
    sample = tmp_path / "sample.vtr"
    if not sample.exists():
        pytest.skip("No sample .vtr file available")
    result = load_rectilinear_grid(str(sample))
    assert isinstance(result, dict)
    assert "x_coords" in result
    assert "y_coords" in result
    assert "z_coords" in result
    assert "point_data" in result
    assert "cell_data" in result
