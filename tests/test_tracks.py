import queue
import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import cv2
import numpy as np

from soccer_homography import App
from soccer_homography.dataTypes import BoundingBox, Homography, Person, Track
from soccer_homography.ui.config.tracks import Tracks
from soccer_homography.pitch import SoccerPitchImage
from soccer_homography.ui.LivePreview import LivePreview


def test_track_export_preserves_person_and_participation_role():
  person = Person( id=12, name="Alex Smith" )
  track = Track(
      clip=4,
      id=9,
      person=person,
      boxes=[
          BoundingBox( 1, 2, 3, 4, 0.9, 0, 10 ),
          BoundingBox( 5, 6, 7, 8, 0.8, 0, 11 ),
      ],
      role="away_goalkeeper",
  )

  exported = track.forExport( 10, 11 )

  assert exported.person == person
  assert exported.role == "away_goalkeeper"
  assert [ box.frame for box in exported.boxes ] == [ 10 ]
  assert track.numId() == person.id
  assert track.to_dict()[ "role" ] == "away_goalkeeper"
  assert track.to_dict()[ "person" ] == person.id


def test_track_without_homography_does_not_create_partial_mappings():
  track = Track(
      clip=1,
      id=8,
      boxes=[ BoundingBox( 1, 2, 3, 4, 0.9, 0, 0 ) ],
  )

  track.refreshHomography( Homography() )

  assert track.homog == []
  assert track.homog_smooth == []


def test_live_preview_skips_tracks_without_mapped_points():
  preview = LivePreview.__new__( LivePreview )
  preview.canvas = Mock()
  preview.pitch = Mock( spec=SoccerPitchImage )
  preview.pitch.get_pitch_scale = ( 1.0, 1.0 )
  track = Track(
      clip=1,
      id=8,
      boxes=[ BoundingBox( 1, 2, 3, 4, 0.9, 0, 0 ) ],
  )

  preview.updateMappings( { track.id: track }, 0 )

  preview.canvas.delete.assert_called_once_with( "mapping" )


def test_initial_clip_load_preserves_preloaded_homography(monkeypatch):
  class FakeCapture:
    def isOpened( self ):
      return True

    def get( self, property_id ):
      return {
          cv2.CAP_PROP_FRAME_WIDTH: 100,
          cv2.CAP_PROP_FRAME_HEIGHT: 100,
          cv2.CAP_PROP_FPS: 30,
          cv2.CAP_PROP_FRAME_COUNT: 1,
      }.get( property_id, 0 )

    def release( self ):
      pass

  monkeypatch.setattr( "soccer_homography.cv2.VideoCapture", Mock( return_value=FakeCapture() ) )
  state = SimpleNamespace(
      cap=None,
      videoFile="",
      curClipID=-1,
      curHomographyID=None,
      boxes={},
      tracks={},
      framesProcessed=0,
      detectChunk=0,
      trackChunk=0,
      db=None,
      data=Homography(),
  )
  state.data.hom4k = np.eye( 3 )
  detections = { 0: [ BoundingBox( 1, 2, 3, 4, 0.9, 0, 0 ) ] }
  restored_tracks = { 5: Track( 1, 5, boxes=[ BoundingBox( 5, 6, 7, 8, 0.8, 0, 0 ) ] ) }
  monkeypatch.setattr( "soccer_homography.readDetectionChunks", lambda clip_id: detections if clip_id == 1 else {} )
  monkeypatch.setattr( "soccer_homography.readTrackingChunks", lambda clip_id: restored_tracks if clip_id == 1 else {} )
  app = cast( Any, App.__new__( App ) )
  app.appState = state
  app.pendingDetectionChunks = set()
  app.pendingTrackingChunks = set()
  app.root = Mock()
  app.tabData = SimpleNamespace(
      tabTracks=SimpleNamespace( onClipLoaded=Mock() ),
      tabClipParticipants=SimpleNamespace( refresh=Mock() ),
  )
  app.sldVideoFrame = Mock()
  app.minFrame = Mock()
  app.maxFrame = Mock()
  app.mainImageController = Mock()
  app.radarMapController = Mock()
  app.minimap = Mock()
  app.checkButtonState = Mock()
  app.refreshHomographyData = Mock()
  loaded_homography = state.data

  assert app.loadSourceVideo( "clip.mp4", 1 )
  assert state.data is loaded_homography
  assert state.data.hom4k is not None
  assert state.boxes == detections
  assert state.tracks == restored_tracks
  app.tabData.tabTracks.onClipLoaded.assert_called_once()
  app.refreshHomographyData.assert_called_once_with( 0 )

  assert app.loadSourceVideo( "next.mp4", 2 )
  assert state.data is not loaded_homography
  assert state.data.hom4k is None


def test_crop_coordinates_scale_to_source_frame_dimensions():
  class SourceCapture:
    def __init__( self ):
      self.frame = np.zeros( ( 400, 400, 3 ), dtype=np.uint8 )
      self.frame[ 10:20, 10:20 ] = ( 0, 0, 255 )

    def set( self, _property_id, _frame_number ):
      self.position = ( _property_id, _frame_number )

    def read( self ):
      return True, self.frame

  tracks = Tracks.__new__( Tracks )
  box = BoundingBox( 10, 10, 20, 20, 0.9, 0, 0 )

  crops = tracks.extractTrackCrops( cast( cv2.VideoCapture, SourceCapture() ), 8, [ box ] )

  assert len( crops ) == 1
  assert crops[ 0 ][ 1 ].shape[:2] == ( 10, 10 )
  assert np.all( crops[ 0 ][ 1 ][ :, :, 0 ] == 255 )


def test_crop_collection_reads_shared_frames_once_then_falls_back_for_unmatched_tracks(monkeypatch):
  class SourceCapture:
    def __init__( self ):
      self.frame_number = 0
      self.seeks: list[ int ] = []

    def isOpened( self ):
      return True

    def set( self, _property_id, frame_number ):
      self.position = ( _property_id, frame_number )
      self.frame_number = frame_number
      self.seeks.append( frame_number )

    def read( self ):
      frame = np.zeros( ( 40, 40, 3 ), dtype=np.uint8 )
      frame[ :, :, 0 ] = self.frame_number
      return True, frame

    def release( self ):
      pass

  capture = SourceCapture()
  monkeypatch.setattr( "soccer_homography.ui.config.tracks.cv2.VideoCapture", Mock( return_value=capture ) )
  tracks = Tracks.__new__( Tracks )
  tracks.cropResults = queue.Queue()
  box = lambda frame: BoundingBox( 2, 2, 12, 12, 0.9, 0, frame )
  unknown_tracks = [
      ( 1, [ box( frame ) for frame in range( 10 ) ] ),
      ( 2, [ box( frame ) for frame in range( 10 ) ] ),
      ( 3, [ box( 1 ) ] ),
      ( 4, [ box( frame ) for frame in range( 1, 9 ) ] ),
  ]

  tracks.collectCropsWorker( 7, "video.mp4", unknown_tracks, threading.Event() )

  messages = []
  while not tracks.cropResults.empty():
    messages.append( tracks.cropResults.get_nowait() )
  result = next( message for message in messages if message.kind == "done" )

  assert capture.seeks == [ 0, 2, 4, 5, 7, 9, 1, 1, 8 ]
  assert [ crop[ 0 ] for crop in result.crops[ 1 ] ] == [ 0, 2, 4, 5, 7, 9 ]
  assert [ crop[ 0 ] for crop in result.crops[ 2 ] ] == [ 0, 2, 4, 5, 7, 9 ]
  assert [ crop[ 0 ] for crop in result.crops[ 3 ] ] == [ 1 ]
  assert [ crop[ 0 ] for crop in result.crops[ 4 ] ] == [ 1, 2, 4, 5, 7, 8 ]


def test_view_change_redraws_detection_and_track_overlays():
  app = cast( Any, App.__new__( App ) )
  transform = object()
  app.mainImageController = Mock()
  app.mainImageController.transform = transform
  app.mainImageController.frame_num = 23
  app.tabData = SimpleNamespace( tabImagePreview=Mock() )
  app.appState = SimpleNamespace(
      boxes={ 23: [ BoundingBox( 1, 2, 3, 4, 0.9, 0, 23 ) ] },
      tracks={ 8: Track( 1, 8 ) },
      data=Homography(),
  )
  app.radarMapController = Mock()

  app.on_main_view_change()

  app.mainImageController.updateBoundingBoxes.assert_called_once_with( app.appState.boxes, 23 )
  app.mainImageController.updateTracks.assert_called_once_with( app.appState.tracks, 23 )
