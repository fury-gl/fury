"""Example tests demonstrating the snapshot testing framework.

This module shows how to use all features of the snapshot testing tools
including color detection, shading analysis, opacity verification, and more.
"""

import numpy as np
import pytest

from fury import actor, window
from fury.testing import (
    analyze_snapshot,
    assert_colors_present,
    assert_object_count,
    assert_opacity_correct,
)


def test_basic_color_detection():
    """Test basic color detection in a simple scene."""
    # Create a scene with colored spheres
    scene = window.Scene()

    centers = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Red, Green, Blue
    radii = np.array([0.5, 0.5, 0.5])

    sphere_actor = actor.sphere(centers=centers, colors=colors, radii=radii)
    scene.add(sphere_actor)

    # Take snapshot
    arr = window.snapshot(scene, size=(400, 400))

    # Analyze colors
    report = analyze_snapshot(arr, colors=colors * 255)

    # Verify all colors are present
    assert all(report.colors_found), "Not all colors were detected"
    assert report.objects >= 3, "Expected at least 3 objects"

    print(f"[OK] Color detection: Found {len(report.colors_detected)} unique colors")
    print(f"[OK] Object count: {report.objects} objects detected")


def test_shading_detection():
    """Test detection of shading and lighting effects."""
    scene = window.Scene()

    # Create a sphere with lighting (should show shading)
    centers = np.array([[0, 0, 0]])
    colors = np.array([[0.8, 0.2, 0.2]])
    radii = np.array([1.0])

    sphere_actor = actor.sphere(centers=centers, colors=colors, radii=radii)
    scene.add(sphere_actor)

    # Take snapshot
    arr = window.snapshot(scene, size=(400, 400))

    # Analyze shading
    report = analyze_snapshot(arr, analyze_shading=True)

    assert report.has_shading, "Shading should be detected on sphere"
    assert report.gradient_magnitude > 0, "Gradient magnitude should be positive"

    print(f"[OK] Shading detected: quality={report.shading_quality:.2f}")
    print(f"[OK] Gradient magnitude: {report.gradient_magnitude:.2f}")


def test_opacity_detection():
    """Test detection of transparency and opacity levels."""
    # Create image with various opacity levels
    img = np.zeros((200, 200, 4), dtype=np.uint8)

    # Solid red square
    img[20:80, 20:80] = [255, 0, 0, 255]

    # Semi-transparent green square (50% opacity)
    img[50:110, 50:110] = [0, 255, 0, 128]

    # More transparent blue square (25% opacity)
    img[80:140, 80:140] = [0, 0, 255, 64]

    # Analyze opacity
    report = analyze_snapshot(img, analyze_opacity=True)

    assert report.opacity_detected, "Opacity should be detected"
    assert len(report.opacity_levels) >= 2, "Multiple opacity levels expected"

    # Check for specific opacity levels (with tolerance)
    opacity_50 = any(abs(level - 0.5) < 0.05 for level in report.opacity_levels)
    opacity_25 = any(abs(level - 0.25) < 0.05 for level in report.opacity_levels)

    assert opacity_50, "50% opacity not detected"
    assert opacity_25, "25% opacity not detected"

    print(f"[OK] Opacity detection: {len(report.opacity_levels)} levels found")
    opacity_str = [f"{level:.2f}" for level in sorted(report.opacity_levels)]
    print(f"  Opacity levels: {opacity_str}")


def test_object_counting():
    """Test counting distinct objects in a scene."""
    scene = window.Scene()

    # Create multiple separated spheres
    centers = np.array(
        [
            [-2, -2, 0],
            [2, -2, 0],
            [-2, 2, 0],
            [2, 2, 0],
            [0, 0, 0],
        ]
    )
    colors = np.random.rand(5, 3)
    radii = np.ones(5) * 0.5

    sphere_actor = actor.sphere(centers=centers, colors=colors, radii=radii)
    scene.add(sphere_actor)

    # Take snapshot
    arr = window.snapshot(scene, size=(400, 400))

    # Count objects
    report = analyze_snapshot(arr, find_objects=True)

    assert report.objects == 5, f"Expected 5 objects, found {report.objects}"
    assert len(report.component_stats) == 5, "Should have stats for 5 objects"

    print(f"[OK] Object counting: {report.objects} objects detected")
    for i, stat in enumerate(report.component_stats):
        print(
            f"  Object {i + 1}: size={stat['size']} pixels, centroid={stat['centroid']}"
        )


def test_brightness_contrast_analysis():
    """Test brightness and contrast measurements."""
    # Create gradient image
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Horizontal gradient from dark to bright
    for i in range(200):
        intensity = int(i / 200 * 255)
        img[:, i] = [intensity, intensity, intensity]

    report = analyze_snapshot(img)

    assert report.brightness_mean > 0, "Mean brightness should be positive"
    assert report.brightness_std > 0, "Brightness variation should exist"
    assert report.contrast > 0, "Contrast should be positive"

    print("[OK] Brightness analysis:")
    print(f"  Mean: {report.brightness_mean:.1f}")
    print(f"  Std Dev: {report.brightness_std:.1f}")
    print(f"  Contrast: {report.contrast:.1f}")


def test_edge_density():
    """Test edge detection and density calculation."""
    # Create image with shapes (high edge density)
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Draw rectangles (lots of edges)
    img[20:80, 20:80] = [255, 0, 0]
    img[40:60, 120:180] = [0, 255, 0]
    img[120:180, 60:120] = [0, 0, 255]

    report = analyze_snapshot(img)

    assert report.edge_density > 0, "Edge density should be positive"

    # Compare with smooth gradient (low edge density)
    smooth_img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        intensity = int(i / 200 * 255)
        smooth_img[:, i] = [intensity, 0, 0]

    smooth_report = analyze_snapshot(smooth_img)

    assert report.edge_density > smooth_report.edge_density, (
        "Sharp edges should have higher density than gradient"
    )

    print("[OK] Edge density:")
    print(f"  Sharp shapes: {report.edge_density:.4f}")
    print(f"  Smooth gradient: {smooth_report.edge_density:.4f}")


def test_color_coverage():
    """Test measuring how much of the image is covered by each color."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Red takes 25% (100x100 square in 200x200 image = 25%)
    img[0:100, 0:100] = [255, 0, 0]

    # Green takes 25%
    img[100:200, 0:100] = [0, 255, 0]

    # Blue takes 50%
    img[:, 100:200] = [0, 0, 255]

    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    report = analyze_snapshot(img, colors=colors)

    assert all(report.colors_found), "All colors should be found"

    # Check coverage (within reasonable tolerance)
    assert abs(report.color_coverage[0] - 25.0) < 2.0, "Red should cover ~25%"
    assert abs(report.color_coverage[1] - 25.0) < 2.0, "Green should cover ~25%"
    assert abs(report.color_coverage[2] - 50.0) < 2.0, "Blue should cover ~50%"

    print("[OK] Color coverage:")
    for i, color in enumerate(colors):
        print(f"  Color {color}: {report.color_coverage[i]:.1f}%")


def test_histogram_analysis():
    """Test color histogram generation."""
    # Create image with known color distribution
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Red region
    img[0:50, 0:50] = [200, 0, 0]

    # Green region
    img[50:100, 0:50] = [0, 150, 0]  # Blue region
    img[0:50, 50:100] = [0, 0, 100]

    # Mixed region
    img[50:100, 50:100] = [100, 100, 100]

    report = analyze_snapshot(img)

    assert "r" in report.histogram, "Red channel histogram missing"
    assert "g" in report.histogram, "Green channel histogram missing"
    assert "b" in report.histogram, "Blue channel histogram missing"

    # Check that histograms have data
    assert len(report.histogram["r"]["counts"]) > 0, "Red histogram empty"
    assert sum(report.histogram["r"]["counts"]) > 0, "Red histogram has no counts"

    print("[OK] Histogram analysis:")
    print(f"  R channel bins: {len(report.histogram['r']['counts'])}")
    print(f"  G channel bins: {len(report.histogram['g']['counts'])}")
    print(f"  B channel bins: {len(report.histogram['b']['counts'])}")


def test_convenience_assertions():
    """Test convenience assertion functions."""
    # Create simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [255, 0, 0]

    # Test assert_colors_present
    try:
        assert_colors_present(img, [[255, 0, 0]])
        print("[OK] assert_colors_present: Passed")
    except AssertionError as e:
        pytest.fail(f"assert_colors_present failed: {e}")

    # Test assert_object_count
    try:
        assert_object_count(img, expected_count=1)
        print("[OK] assert_object_count: Passed")
    except AssertionError as e:
        pytest.fail(f"assert_object_count failed: {e}")

    # Test with alpha channel for opacity
    img_alpha = np.zeros((100, 100, 4), dtype=np.uint8)
    img_alpha[20:80, 20:80] = [255, 0, 0, 128]  # 50% opacity

    try:
        assert_opacity_correct(img_alpha, expected_levels=[0.5], tolerance=0.05)
        print("[OK] assert_opacity_correct: Passed")
    except AssertionError as e:
        pytest.fail(f"assert_opacity_correct failed: {e}")


def test_compare_snapshots():
    """Test comparing two snapshots for consistency."""
    # Create two images with same structure but different colors
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[25:75, 25:75] = [200, 0, 0]

    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2[25:75, 25:75] = [0, 200, 0]

    report1 = analyze_snapshot(img1)
    report2 = analyze_snapshot(img2)

    # Both should have same object count
    assert report1.objects == report2.objects, "Object count should match"

    # Both should detect their respective colors
    r1 = analyze_snapshot(img1, colors=[[200, 0, 0]])
    r2 = analyze_snapshot(img2, colors=[[0, 200, 0]])

    assert r1.colors_found[0], "Should find red in img1"
    assert r2.colors_found[0], "Should find green in img2"

    print("[OK] Snapshot comparison: Both images analyzed correctly")


if __name__ == "__main__":
    """Run all example tests."""
    print("=" * 60)
    print("FURY Snapshot Testing Examples")
    print("=" * 60)
    print()

    try:
        test_basic_color_detection()
        print()
    except Exception as e:
        print(f"[FAIL] Color detection test failed: {e}\n")

    try:
        test_shading_detection()
        print()
    except Exception as e:
        print(f"[FAIL] Shading detection test failed: {e}\n")

    try:
        test_opacity_detection()
        print()
    except Exception as e:
        print(f"[FAIL] Opacity detection test failed: {e}\n")

    try:
        test_object_counting()
        print()
    except Exception as e:
        print(f"[FAIL] Object counting test failed: {e}\n")

    try:
        test_brightness_contrast_analysis()
        print()
    except Exception as e:
        print(f"[FAIL] Brightness/contrast test failed: {e}\n")

    try:
        test_edge_density()
        print()
    except Exception as e:
        print(f"[FAIL] Edge density test failed: {e}\n")

    try:
        test_color_coverage()
        print()
    except Exception as e:
        print(f"[FAIL] Color coverage test failed: {e}\n")

    try:
        test_histogram_analysis()
        print()
    except Exception as e:
        print(f"[FAIL] Histogram analysis test failed: {e}\n")

    try:
        test_convenience_assertions()
        print()
    except Exception as e:
        print(f"[FAIL] Convenience assertions test failed: {e}\n")

    try:
        test_compare_snapshots()
        print()
    except Exception as e:
        print(f"[FAIL] Snapshot comparison test failed: {e}\n")

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
