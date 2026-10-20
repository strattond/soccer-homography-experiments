from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from soccer_homography.ui.MainCanvas import MainCanvasController


def test_draw_field_lines_projects_field_metres_and_applies_canvas_transform():
  controller = MainCanvasController.__new__( MainCanvasController )
  controller.canvas = Mock()
  controller.appState = SimpleNamespace(
      data=SimpleNamespace(
          hom4k=np.array(
              [
                      [ 0.1, 0.0, -10.0 ],
                      [ 0.0, 0.05, -2.5 ],
                  [ 0.0, 0.0, 1.0 ],
              ]
          )
      ),
      cfg=SimpleNamespace(
          edges=[ ( 1, 2 ) ],
          vertices=[ ( 1.0, 2.0 ), ( 3.0, 4.0 ) ],
          length=105.0,
          width=68.0,
          centre_circle_radius=0.0,
      ),
      imgOpts=SimpleNamespace( showProjectedField=True ),
  )
  controller.transform = SimpleNamespace(
      dimensions=SimpleNamespace( x=1280, y=720 ),
      toDisplay=lambda x, y: ( x * 0.5 + 5.0, y * 0.5 + 7.0 )
  )

  controller.drawFieldLines()

  controller.canvas.delete.assert_called_once_with( "projfield" )
  controller.canvas.create_line.assert_called_once_with(
      60.0,
      52.0,
      70.0,
      72.0,
      fill="cyan",
      width=2,
      tags=( "projfield", ),
  )


def test_draw_field_lines_clips_projected_segments_to_image_bounds():
  controller = MainCanvasController.__new__( MainCanvasController )
  controller.canvas = Mock()
  controller.appState = SimpleNamespace(
      data=SimpleNamespace(
          hom4k=np.array(
              [
                  [ 0.1, 0.0, -10.0 ],
                  [ 0.0, 0.05, -2.5 ],
                  [ 0.0, 0.0, 1.0 ],
              ]
          )
      ),
      cfg=SimpleNamespace(
          edges=[ ( 1, 2 ) ],
          vertices=[ ( 1.0, 2.0 ), ( 3.0, 4.0 ) ],
          length=105.0,
          width=68.0,
          centre_circle_radius=0.0,
      ),
      imgOpts=SimpleNamespace( showProjectedField=True ),
  )
  controller.transform = SimpleNamespace(
      dimensions=SimpleNamespace( x=120, y=100 ),
      toDisplay=lambda x, y: ( x * 0.5 + 5.0, y * 0.5 + 7.0 )
  )

  controller.drawFieldLines()

  controller.canvas.create_line.assert_called_once_with(
      60.0,
      52.0,
      62.25,
      56.5,
      fill="cyan",
      width=2,
      tags=( "projfield", ),
  )


def test_draw_field_lines_skips_segments_crossing_projective_horizon():
  inverse_homography = np.eye( 3 )
  inverse_homography[ 2 ] = ( 1.0, 0.0, -2.0 )
  controller = MainCanvasController.__new__( MainCanvasController )
  controller.canvas = Mock()
  controller.appState = SimpleNamespace(
      data=SimpleNamespace( hom4k=np.linalg.inv( inverse_homography ) ),
      cfg=SimpleNamespace(
          edges=[ ( 1, 2 ) ],
          vertices=[ ( 1.0, 2.0 ), ( 3.0, 4.0 ) ],
          length=105.0,
          width=68.0,
          centre_circle_radius=0.0,
      ),
      imgOpts=SimpleNamespace( showProjectedField=True ),
  )
  controller.transform = SimpleNamespace(
      dimensions=SimpleNamespace( x=1280, y=720 ),
      toDisplay=lambda x, y: ( x, y )
  )

  controller.drawFieldLines()

  controller.canvas.create_line.assert_not_called()


def test_draw_field_lines_includes_center_circle():
  controller = MainCanvasController.__new__( MainCanvasController )
  controller.canvas = Mock()
  controller.appState = SimpleNamespace(
      data=SimpleNamespace( hom4k=np.eye( 3 ) ),
      cfg=SimpleNamespace(
          edges=[],
          vertices=[],
          length=105.0,
          width=68.0,
          centre_circle_radius=9.14,
      ),
      imgOpts=SimpleNamespace( showProjectedField=True ),
  )
  controller.transform = SimpleNamespace(
      dimensions=SimpleNamespace( x=1280, y=720 ),
      toDisplay=lambda x, y: ( x, y )
  )

  controller.drawFieldLines()

  assert controller.canvas.create_line.call_count == 96
  first_segment = controller.canvas.create_line.call_args_list[ 0 ]
  angle = 2 * np.pi / 96
  np.testing.assert_allclose(
      first_segment.args,
      ( 52.5 + 9.14, 34.0, 52.5 + 9.14 * np.cos( angle ), 34.0 + 9.14 * np.sin( angle ) ),
  )
  assert first_segment.kwargs[ "tags" ] == ( "projfield", )


def test_draw_field_lines_toggle_clears_overlay():
  controller = MainCanvasController.__new__( MainCanvasController )
  controller.canvas = Mock()
  controller.appState = SimpleNamespace(
      data=SimpleNamespace( hom4k=np.eye( 3 ) ),
      cfg=SimpleNamespace( edges=[], vertices=[], length=105.0, width=68.0, centre_circle_radius=9.14 ),
      imgOpts=SimpleNamespace( showProjectedField=False ),
  )

  controller.drawFieldLines()

  controller.canvas.delete.assert_called_once_with( "projfield" )
  controller.canvas.create_line.assert_not_called()
