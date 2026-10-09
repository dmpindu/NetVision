"""Unit tests for court geometry, keypoint validation, homography and rendering."""
import cv2
import numpy as np
import pytest

from netvision.court import CourtCanvas, CourtGeometry, TacticalViewConverter, project_points, render_court

IMAGE_CORNERS = np.float32([[140, 250], [1140, 250], [1265, 690], [15, 690]])
COURT_CORNERS = np.float32([[0, 0], [28, 0], [28, 15], [0, 15]])
COURT_TO_IMAGE = cv2.getPerspectiveTransform(COURT_CORNERS, IMAGE_CORNERS)


def image_keypoints(geometry=CourtGeometry()):
    return project_points(COURT_TO_IMAGE, geometry.keypoints)


def test_geometry_keypoints():
    """TC-CRT-01: 18 keypoints, mirrored left/right, inside the court."""
    geometry = CourtGeometry()
    points = geometry.keypoints
    assert points.shape == (18, 2)
    assert all(geometry.contains(p) for p in points)
    assert points[8][0] == pytest.approx(28 - points[16][0])
    assert points[2][1] == pytest.approx(15 - points[3][1])
    assert geometry.lane_bottom - geometry.lane_top == pytest.approx(4.9)


def test_geometry_rims_and_distance():
    """TC-CRT-02: rim positions and distance to the nearest rim."""
    geometry = CourtGeometry()
    assert geometry.rims.tolist() == [[1.575, 7.5], [26.425, 7.5]]
    assert geometry.distance_to_nearest_rim((26.425, 7.5 + 6.75)) == pytest.approx(6.75)
    assert not geometry.contains((-3, 5), margin=1.0)
    assert geometry.contains((-0.5, 5), margin=1.0)
    assert len(geometry.line_polylines()) >= 10


def test_homography_recovers_court_positions():
    """TC-CRT-03: with exact keypoints the homography maps pixels back to meters within 1 cm."""
    converter = TacticalViewConverter()
    homography = converter.compute_homography(image_keypoints())
    assert homography is not None
    test_points_m = np.array([[3.0, 4.0], [14.0, 7.5], [25.0, 12.0]])
    pixels = project_points(COURT_TO_IMAGE, test_points_m)
    assert np.abs(project_points(homography, pixels) - test_points_m).max() < 0.01


def test_validation_removes_a_corrupted_keypoint():
    """TC-CRT-04: a keypoint far from where the court says it should be is dropped."""
    converter = TacticalViewConverter()
    points = image_keypoints()
    points[3] = (900, 100)
    validated = converter.validate_keypoints(points)
    assert np.isnan(validated[3]).all()
    assert np.isfinite(np.delete(validated, 3, axis=0)).all()
    assert converter.compute_homography(points) is not None  # RANSAC also copes without validation


def test_not_enough_keypoints():
    """TC-CRT-05: fewer than 4 keypoints, or none at all, gives no homography."""
    converter = TacticalViewConverter()
    points = np.full((18, 2), np.nan)
    points[:3] = image_keypoints()[:3]
    assert converter.compute_homography(points) is None
    assert converter.compute_homography(None) is None
    assert converter.validate_keypoints(None) is None


def test_homography_carry_forward():
    """TC-CRT-06: the last good homography covers up to N missing frames, then stops."""
    converter = TacticalViewConverter(carry_frames=3)
    good = image_keypoints()
    homographies = converter.build_homographies([good, None, None, None, None, good])
    assert [h is not None for h in homographies] == [True, True, True, True, False, True]


def test_players_to_court_drops_off_court_people():
    """TC-CRT-07: foot points become court meters; people outside the court (fans) are dropped."""
    converter = TacticalViewConverter()
    homography = converter.compute_homography(image_keypoints())
    foot_on_court = project_points(COURT_TO_IMAGE, [[10.0, 5.0]])[0]
    player = [foot_on_court[0] - 20, foot_on_court[1] - 90, foot_on_court[0] + 20, foot_on_court[1]]
    fan = [600, 60, 640, 150]  # in the stands, far behind the court
    positions = converter.players_to_court([{1: player, 2: fan}, {}], [homography, None])
    assert set(positions[0]) == {1}
    assert positions[0][1] == pytest.approx((10.0, 5.0), abs=0.01)
    assert positions[1] == {}
    assert project_points(homography, np.empty((0, 2))).shape == (0, 2)


def test_render_court():
    """TC-CRT-08: the top-down court image has the expected size and white lines."""
    canvas = CourtCanvas(CourtGeometry(), px_per_m=10, margin_px=5)
    image = render_court(canvas)
    assert image.shape == (160, 290, 3)
    assert (image == 255).all(axis=2).sum() > 500
    assert canvas.to_pixels([(0, 0)]).tolist() == [[5, 5]]
