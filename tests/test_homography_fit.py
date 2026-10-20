from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from soccer_homography.appState import ImageOptions
from soccer_homography.data import Homography, Point2D, SelectionPoint, ViewTransform
from soccer_homography.pitch import SoccerPitchConfiguration, SoccerPitchImage
from soccer_homography.ui.RadarCanvas import RadarCanvas
from soccer_homography.ui.config.imageoptions import ImageOptionsUI


def calibration_points():
  return [
      SelectionPoint( coords=Point2D( 0, 0 ) ),
      SelectionPoint( coords=Point2D( 10, 0 ) ),
      SelectionPoint( coords=Point2D( 0, 10 ) ),
      SelectionPoint( coords=Point2D( 10, 10 ) ),
  ]


def test_homography_compute_uses_requested_fit_method( monkeypatch ):
  find_homography = Mock( return_value=( np.eye( 3 ), None ) )
  monkeypatch.setattr( cv2, "findHomography", find_homography )
  homography = Homography(
      img_pts_4k=calibration_points(),
      world_pts=calibration_points(),
  )

  homography.compute( method=0 )

  assert homography.hom4k is not None
  assert find_homography.call_args.kwargs[ "method" ] == 0


def test_scaled_homography_uses_requested_fit_method( monkeypatch ):
  find_homography = Mock( return_value=( np.eye( 3 ), None ) )
  monkeypatch.setattr( cv2, "findHomography", find_homography )
  homography = Homography(
      img_pts_4k=calibration_points(),
      world_pts=calibration_points(),
  )

  result = homography.computeScaledHomography( ViewTransform(), method=0 )

  assert result is not None
  assert find_homography.call_args.kwargs[ "method" ] == 0


def test_image_options_switches_homography_fit_method():
  options = ImageOptionsUI.__new__( ImageOptionsUI )
  options.appState = SimpleNamespace( imgOpts=ImageOptions() )

  assert options.getHomographyMethod() == cv2.RANSAC
  options.appState.imgOpts.fitAllHomographyPoints = True
  assert options.getHomographyMethod() == 0


def test_image_options_fit_change_invokes_homography_refresh():
  options = ImageOptionsUI.__new__( ImageOptionsUI )
  options.appState = SimpleNamespace( imgOpts=ImageOptions() )
  option_values = {
      "closeEdges": False,
      "edgeEnhance": True,
      "edgeType": "Canny",
      "lineType": "LineSegmentDetector",
      "preBlur": True,
      "removeSky": True,
      "showHough": False,
      "showProjectedField": True,
      "fitAllHomographyPoints": True,
  }
  options.uiOpts = SimpleNamespace(
      **{
          key: SimpleNamespace( get=lambda key=key: option_values[ key ] )
          for key in option_values
      }
  )
  options.on_change = Mock()
  options.on_homography_fit_change = Mock()

  options.uiToState()

  assert options.appState.imgOpts.fitAllHomographyPoints is True
  options.on_change.assert_called_once_with()
  options.on_homography_fit_change.assert_called_once_with()


def make_radar_canvas():
  pitch = SoccerPitchImage( cfg=SoccerPitchConfiguration() )
  radar = RadarCanvas.__new__( RadarCanvas )
  radar.cfg = pitch.cfg
  radar.pitch = pitch
  return radar


def test_radar_canvas_allows_mapping_arbitrary_interior_field_points():
  radar = make_radar_canvas()
  world_point = ( 28.0, 20.0 )
  scale_w, scale_l = radar.pitch.get_pitch_scale
  screen_point = (
      world_point[ 0 ] * scale_w + radar.pitch.padding,
      world_point[ 1 ] * scale_l + radar.pitch.padding,
  )

  mapped, edge = radar.point_at_cursor( *screen_point )

  assert mapped.index is None
  assert np.allclose( tuple( mapped.coords ), world_point )
  assert edge is None


def test_radar_canvas_still_snaps_clicks_near_field_lines():
  radar = make_radar_canvas()
  scale_w, scale_l = radar.pitch.get_pitch_scale
  screen_point = ( 35.0 * scale_w + radar.pitch.padding, radar.pitch.padding + 2 )

  mapped, edge = radar.point_at_cursor( *screen_point )

  assert mapped.index is None
  assert np.isclose( mapped.coords.y, 0.0 )
  assert edge is not None


def test_radar_canvas_drag_moves_arbitrary_points_and_clamps_to_field():
  radar = make_radar_canvas()
  point = SelectionPoint( None, Point2D( 28.0, 20.0 ) )
  radar.dragged_selection = point
  radar.dragged_edge = None
  radar.drag_start = None
  radar.drag_was_fixed = False
  radar.selected = [ point ]
  radar.selection_items = {}
  radar.canvas = Mock()
  radar.on_selection_move = Mock()

  radar.handle_drag(
      SimpleNamespace(
          x=radar.pitch.padding + radar.pitch.width + 20,
          y=radar.pitch.padding + 12,
      )
  )

  assert np.isclose( point.coords.x, radar.cfg.length )
  assert np.isclose( point.coords.y, 3.0 )
  radar.on_selection_move.assert_called_once_with()
