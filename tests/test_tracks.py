from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import cv2
import numpy as np

from soccer_homography import App
from soccer_homography.dataTypes import BoundingBox, Homography, Person, Track
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
  app = cast( Any, App.__new__( App ) )
  app.appState = state
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
  loaded_homography = state.data

  assert app.loadSourceVideo( "clip.mp4", 1 )
  assert state.data is loaded_homography
  assert state.data.hom4k is not None

  assert app.loadSourceVideo( "next.mp4", 2 )
  assert state.data is not loaded_homography
  assert state.data.hom4k is None
