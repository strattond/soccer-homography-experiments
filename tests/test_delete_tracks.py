from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from soccer_homography import App
from soccer_homography.data import Track
from soccer_homography.SportsTracker import CommandType
from soccer_homography.ui.frameminimap import TrackingType


def make_app() -> App:
  app = cast( Any, App.__new__( App ) )
  app.root = Mock()
  app.appState = SimpleNamespace(
      curClipID=6,
      db=object(),
      tracks={ 9: Track( clip=6, id=9, boxes=[] ) },
      trackChunk=3,
  )
  app.tracking = None
  app.tracking_writer = Mock()
  app.pendingTrackingChunks = { 2 }
  app.tabData = SimpleNamespace( tabTracks=Mock() )
  app.mainImageController = Mock()
  app.mainImageController.frame_num = 42
  app.livePreviewController = Mock()
  app.minimap = Mock()
  app.checkButtonState = Mock()
  return app


def test_delete_tracks_requires_confirmation( monkeypatch ):
  app = make_app()
  delete_chunks = Mock()
  delete_clip_tracks = Mock()
  monkeypatch.setattr( "soccer_homography.messagebox.askyesno", Mock( return_value=False ) )
  monkeypatch.setattr( "soccer_homography.deleteTrackingChunks", delete_chunks )
  monkeypatch.setattr( "soccer_homography.deleteClipTracks", delete_clip_tracks )

  app.deleteTracksForCurrentClip()

  delete_chunks.assert_not_called()
  delete_clip_tracks.assert_not_called()
  app.tracking_writer.waitForPending.assert_not_called()
  assert app.appState.tracks
  assert app.pendingTrackingChunks == { 2 }


def test_delete_tracks_clears_clip_data_and_refreshes_ui( monkeypatch ):
  app = make_app()
  delete_chunks = Mock( return_value=2 )
  delete_clip_tracks = Mock()
  monkeypatch.setattr( "soccer_homography.messagebox.askyesno", Mock( return_value=True ) )
  monkeypatch.setattr( "soccer_homography.deleteTrackingChunks", delete_chunks )
  monkeypatch.setattr( "soccer_homography.deleteClipTracks", delete_clip_tracks )

  app.deleteTracksForCurrentClip()

  app.tracking_writer.waitForPending.assert_called_once_with()
  delete_chunks.assert_called_once_with( 6 )
  delete_clip_tracks.assert_called_once_with( app.appState.db, 6 )
  assert app.appState.tracks == {}
  assert app.appState.trackChunk == 0
  assert app.pendingTrackingChunks == set()
  app.tabData.tabTracks.refresh.assert_called_once_with()
  app.mainImageController.updateTracks.assert_called_once_with( {}, 42 )
  app.livePreviewController.updateMappings.assert_called_once_with( {}, 42 )
  app.minimap.clear.assert_called_once_with( TrackingType.CUR_TRACK )
  app.checkButtonState.assert_called_once_with()


def test_delete_tracks_is_refused_while_tracking_is_active( monkeypatch ):
  app = make_app()
  app.tracking = SimpleNamespace(
      thread=SimpleNamespace( is_alive=Mock( return_value=True ) ),
      stopped=False,
      curMode=CommandType.RUN_TRACK,
      out_queue=SimpleNamespace( empty=Mock( return_value=True ) ),
  )
  confirm = Mock()
  monkeypatch.setattr( "soccer_homography.messagebox.askyesno", confirm )
  monkeypatch.setattr( "soccer_homography.messagebox.showwarning", Mock() )

  app.deleteTracksForCurrentClip()

  confirm.assert_not_called()
  app.tracking_writer.waitForPending.assert_not_called()
  assert app.appState.tracks


def test_delete_tracks_waits_until_completed_tracker_outputs_are_drained( monkeypatch ):
  app = make_app()
  app.tracking = SimpleNamespace(
      thread=SimpleNamespace( is_alive=Mock( return_value=False ) ),
      stopped=False,
      curMode=CommandType.RUN_TRACK,
      out_queue=SimpleNamespace( empty=Mock( return_value=False ) ),
  )
  confirm = Mock()
  monkeypatch.setattr( "soccer_homography.messagebox.askyesno", confirm )
  monkeypatch.setattr( "soccer_homography.messagebox.showwarning", Mock() )

  app.deleteTracksForCurrentClip()

  confirm.assert_not_called()
  app.tracking_writer.waitForPending.assert_not_called()
  assert app.appState.tracks
