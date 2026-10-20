from types import SimpleNamespace
from unittest.mock import Mock

from soccer_homography import App
from soccer_homography.data import Homography, Point2D, SelectionPoint
from soccer_homography.ui.MainCanvas import MainCanvasController
from soccer_homography.ui.RadarCanvas import RadarCanvas


def test_main_canvas_right_click_deletes_calibration_marker_instead_of_starting_pan():
  point = SelectionPoint( coords=Point2D( 20, 30 ) )
  controller = MainCanvasController.__new__( MainCanvasController )
  controller.canvas = Mock()
  controller.canvas.find_overlapping.return_value = [ 15, 27 ]
  controller.selection_items = { 27: point }
  controller.on_selection_delete = Mock()
  controller.startPan = Mock()

  result = controller.handleRightClick( SimpleNamespace( x=20, y=30 ) )

  controller.on_selection_delete.assert_called_once_with( point )
  controller.startPan.assert_not_called()
  assert result == "break"


def test_main_canvas_right_click_on_empty_space_starts_pan():
  controller = MainCanvasController.__new__( MainCanvasController )
  controller.canvas = Mock()
  controller.canvas.find_overlapping.return_value = [ 15 ]
  controller.selection_items = {}
  controller.startPan = Mock()
  event = SimpleNamespace( x=20, y=30 )

  controller.handleRightClick( event )

  controller.startPan.assert_called_once_with( event )


def test_radar_canvas_right_click_deletes_calibration_marker():
  point = SelectionPoint( coords=Point2D( 20, 30 ) )
  controller = RadarCanvas.__new__( RadarCanvas )
  controller.canvas = Mock()
  controller.canvas.find_overlapping.return_value = [ 18 ]
  controller.selection_items = { 18: ( point, None ) }
  controller.on_selection_delete = Mock()

  result = controller.handle_right_click( SimpleNamespace( x=20, y=30 ) )

  controller.on_selection_delete.assert_called_once_with( point )
  assert result == "break"


def make_app_with_points():
  app = App.__new__( App )
  image_points = [ SelectionPoint( coords=Point2D( i, i + 1 ) ) for i in range( 4 ) ]
  world_points = [ SelectionPoint( coords=Point2D( i + 10, i + 11 ) ) for i in range( 4 ) ]
  app.appState = SimpleNamespace( data=Homography( img_pts_4k=image_points, world_pts=world_points ) )
  app.radarMapController = Mock()
  app.mainImageController = Mock()
  app.redisplayHomographyData = Mock()
  return app, image_points, world_points


def test_deleting_image_point_removes_same_index_pair_and_refreshes_views():
  app, image_points, world_points = make_app_with_points()
  image_points_snapshot = image_points.copy()
  world_points_snapshot = world_points.copy()

  app.on_main_selection_delete( image_points[ 2 ] )

  assert app.appState.data.img_pts_4k == [ image_points_snapshot[ 0 ], image_points_snapshot[ 1 ], image_points_snapshot[ 3 ] ]
  assert app.appState.data.world_pts == [ world_points_snapshot[ 0 ], world_points_snapshot[ 1 ], world_points_snapshot[ 3 ] ]
  app.radarMapController.updateSelectionMarkers.assert_called_once_with( app.appState.data.world_pts )
  app.mainImageController.updateSelectionMarkers.assert_called_once_with( app.appState.data.img_pts_4k )
  app.redisplayHomographyData.assert_called_once_with()


def test_deleting_radar_point_removes_same_index_pair_and_recomputes():
  app, image_points, world_points = make_app_with_points()
  image_points_snapshot = image_points.copy()
  world_points_snapshot = world_points.copy()

  app.on_radar_selection_delete( world_points[ 1 ] )

  assert app.appState.data.img_pts_4k == [ image_points_snapshot[ 0 ], image_points_snapshot[ 2 ], image_points_snapshot[ 3 ] ]
  assert app.appState.data.world_pts == [ world_points_snapshot[ 0 ], world_points_snapshot[ 2 ], world_points_snapshot[ 3 ] ]
  app.redisplayHomographyData.assert_called_once_with()


def test_deletion_does_not_partially_remove_unpaired_homography_points():
  app, image_points, _ = make_app_with_points()
  app.appState.data.world_pts.pop()
  image_points_snapshot = image_points.copy()
  world_points_snapshot = app.appState.data.world_pts.copy()
  app.redisplayHomographyData = Mock()

  app.on_main_selection_delete( image_points[ 1 ] )

  assert app.appState.data.img_pts_4k == image_points_snapshot
  assert app.appState.data.world_pts == world_points_snapshot
  app.redisplayHomographyData.assert_not_called()
