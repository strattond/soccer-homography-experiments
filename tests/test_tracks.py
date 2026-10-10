import importlib
import queue
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import cv2
import numpy as np

from soccer_homography import App
from soccer_homography.data import BoundingBox, Homography, Person, Point2D, Track, TrackSegment
from soccer_homography.db.persist import Match as DBMatch
from soccer_homography.db.persist import Person as DBPerson
from soccer_homography.db.persist import PersonParticipation
from soccer_homography.inference.abstractions import IdentificationImageResult, selectModelDevice
from soccer_homography.inference.clip_model import MIN_CLIP_GPU_MEMORY_BYTES, ClipImageResult, ClipResponse, ClipRoleClassifier
from soccer_homography.inference.crop_inference import (
  DEFAULT_IDENTIFICATION_PROMPT,
  CropInferenceWorker,
  loadIdentificationPrompt,
  mostLikelyRole,
  roleVoteCounts,
  saveIdentificationPrompt,
)
from soccer_homography.inference.vlm_model import MIN_VLM_GPU_MEMORY_BYTES, MoondreamVLM, VLMImageResult, VLMResponse, guessRole
from soccer_homography.pitch import SoccerPitchImage
from soccer_homography.SportsTracker import SportsTracker
from soccer_homography.ui.components import Slider
from soccer_homography.ui.config.crop_worker import CropExtractionWorker, cropFromFrame, planCropFrames
from soccer_homography.ui.config.tracks import Tracks
from soccer_homography.ui.config.modeloptions import ModelOptions
from soccer_homography.ui.Configuration import ClipParticipants
from soccer_homography.ui.frameminimap import FrameMinimap
from soccer_homography.ui.LivePreview import LivePreview

configuration_module = importlib.import_module( "soccer_homography.ui.Configuration" )
tracks_module = importlib.import_module( "soccer_homography.ui.config.tracks" )


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


def test_track_initial_segments_follow_consecutive_frame_runs_and_extend_live():
  track = Track(
      clip=1,
      id=8,
      boxes=[
          BoundingBox( 1, 2, 3, 4, 0.9, 0, frame )
          for frame in ( 0, 1, 2, 5, 6 )
      ],
  )

  assert [ ( segment.frame_start, segment.frame_end ) for segment in track.segments ] == [ ( 0, 2 ), ( 5, 6 ) ]

  track.addBox( BoundingBox( 1, 2, 3, 4, 0.9, 0, 7 ) )
  track.addBox( BoundingBox( 1, 2, 3, 4, 0.9, 0, 10 ) )

  assert [ ( segment.frame_start, segment.frame_end ) for segment in track.segments ] == [ ( 0, 2 ), ( 5, 7 ), ( 10, 10 ) ]


def test_track_segment_role_is_resolved_for_the_requested_frame():
  track = Track(
      clip=1,
      id=8,
      boxes=[
          BoundingBox( 1, 2, 3, 4, 0.9, 0, frame )
          for frame in ( 0, 1, 4, 5 )
      ],
      segments=[
          TrackSegment( 0, 1, 17 ),
          TrackSegment( 4, 5, None ),
      ],
  )

  assert track.roleAt( 0, { 17: "referee" } ) == "referee"
  assert track.roleAt( 4, { 17: "referee" } ) == "unknown"
  assert track.roleAt( 2, { 17: "referee" } ) == "unknown"


def test_live_preview_skips_tracks_without_mapped_points():
  preview = LivePreview.__new__( LivePreview )
  preview.canvas = Mock()
  preview.pitch = Mock( spec=SoccerPitchImage )
  preview.pitch.get_pitch_scale = ( 1.0, 1.0 )
  preview.pointerPosition = None
  track = Track(
      clip=1,
      id=8,
      boxes=[ BoundingBox( 1, 2, 3, 4, 0.9, 0, 0 ) ],
  )

  preview.updateMappings( { track.id: track }, 0 )

  preview.canvas.delete.assert_called_once_with( "mapping" )


def test_live_preview_colors_marker_from_segment_participant_role():
  preview = LivePreview.__new__( LivePreview )
  preview.canvas = Mock()
  preview.pitch = Mock( spec=SoccerPitchImage )
  preview.pitch.get_pitch_scale = ( 1.0, 1.0 )
  preview.pointerPosition = None
  preview.hoveredTrackID = None
  preview.draw = Mock()
  track = Track(
      clip=1,
      id=8,
      boxes=[ BoundingBox( 1, 2, 3, 4, 0.9, 0, 0 ) ],
      segments=[ TrackSegment( 0, 0, 17 ) ],
      homog=[ Point2D( 1, 2 ) ],
      homog_smooth=[ Point2D( 1, 2 ) ],
  )

  preview.updateMappings( { track.id: track }, 0, { 17: "referee" } )

  assert preview.draw.call_args_list[ 0 ].args[ 1 ] == "#ffffff"
  assert preview.draw.call_args_list[ 1 ].args[ 1 ] == "#ffffff"


def test_live_preview_hit_test_returns_nearest_marker_and_ignores_distant_markers():
  class MarkerCanvas:
    def find_overlapping( self, *_bounds ):
      return ( 1, 2 )

    def gettags( self, item_id ):
      return {
          1: ( "mapping", "homography_marker", "track:11" ),
          2: ( "mapping", "homography_marker", "track:12" ),
      }[ item_id ]

    def coords( self, item_id ):
      return {
          1: ( 8, 8, 20, 20 ),
          2: ( 18, 8, 30, 20 ),
      }[ item_id ]

  preview = LivePreview.__new__( LivePreview )
  preview.canvas = cast( Any, MarkerCanvas() )

  assert preview.markerTrackAt( 20, 14 ) == 12
  assert preview.markerTrackAt( 40, 40 ) is None


def test_live_preview_hover_highlights_marker_and_clears_when_pointer_moves_off():
  preview = LivePreview.__new__( LivePreview )
  preview.canvas = Mock()
  preview.hoveredTrackID = None
  preview.markerTrackAt = Mock( side_effect=( 11, None ) )

  preview.updateHoveredTrack( 12, 24 )
  preview.updateHoveredTrack( 40, 40 )

  assert preview.canvas.itemconfigure.call_args_list == [
      ( ( "track:11", ), { "outline": "#ffff00", "width": 3 } ),
      ( ( "track:11", ), { "outline": "black", "width": 1 } ),
  ]
  assert preview.hoveredTrackID is None
  preview.hoveredTrackID = 11

  preview.onCanvasLeave()

  assert preview.canvas.itemconfigure.call_args_list[ -1 ] == (
      ( "track:11", ),
      { "outline": "black", "width": 1 },
  )
  assert preview.pointerPosition is None


def test_live_preview_marker_click_selects_associated_track():
  preview = LivePreview.__new__( LivePreview )
  preview.markerTrackAt = Mock( return_value=11 )
  preview.on_track_select = Mock()

  preview.onCanvasClick( SimpleNamespace( x=12, y=24 ) )

  preview.on_track_select.assert_called_once_with( 11 )


def test_minimap_click_maps_vertical_position_to_nearest_frame():
  minimap = FrameMinimap.__new__( FrameMinimap )
  minimap.totalFrames = 101
  minimap.winfo_height = Mock( return_value=201 )
  minimap.on_frame_select = Mock()

  assert minimap.getFrameForY( 0 ) == 100
  assert minimap.getFrameForY( 100 ) == 50
  assert minimap.getFrameForY( 200 ) == 0
  assert minimap.getFrameForY( -5 ) == 100
  assert minimap.getFrameForY( 205 ) == 0
  minimap.onClick( SimpleNamespace( y=100 ) )
  minimap.on_frame_select.assert_called_once_with( 50 )


def test_minimap_draws_unassigned_and_assigned_track_segments_in_red_and_blue():
  minimap = FrameMinimap.__new__( FrameMinimap )
  minimap.trackSegments = [ TrackSegment( 0, 2 ), TrackSegment( 5, 8, 17 ) ]
  minimap.tracking = {}
  minimap.currentFrame = None
  minimap.colorTrack = "#404040"
  minimap.winfo_width = Mock( return_value=30 )
  minimap.winfo_height = Mock( return_value=101 )
  minimap.getYForFrame = lambda frame: 100 - frame * 10
  minimap.delete = Mock()
  minimap.create_rectangle = Mock()
  minimap.create_line = Mock()

  minimap.redraw()

  assert [ call.kwargs[ "fill" ] for call in minimap.create_rectangle.call_args_list ] == [
      "#404040",
      "#ff0000",
      "#0000ff",
  ]


def test_tracks_table_shows_all_tracks_without_a_role_column():
  tracks = Tracks.__new__( Tracks )
  tracks.appState = cast( Any, SimpleNamespace(
      tracks={
          1: Track( clip=1, id=1, boxes=[], role="unknown" ),
          2: Track( clip=1, id=2, boxes=[], role="home_player" ),
      }
  ) )
  tracks.refreshPeople = Mock()
  tracks.tblTrackData = Mock()
  tracks.tblTrackData.get_children.return_value = ()
  tracks.tblTrackData.selection.return_value = ()
  tracks.peopleByLabel = {}
  tracks.personLabels = {}
  tracks.current_frame = 0
  tracks.refreshing = False
  tracks.selectionChanged = Mock()

  tracks.refresh()

  assert [ call.kwargs[ "iid" ] for call in tracks.tblTrackData.insert.call_args_list ] == [ "1", "2" ]
  assert all( len( call.kwargs[ "values" ] ) == 3 for call in tracks.tblTrackData.insert.call_args_list )


def test_tracks_table_shows_person_for_segment_at_current_frame():
  track = Track(
      clip=1,
      id=1,
      boxes=[ BoundingBox( 1, 2, 3, 4, 0.9, 0, frame ) for frame in ( 0, 1, 5, 6 ) ],
      segments=[ TrackSegment( 0, 1, 17 ), TrackSegment( 5, 6, None ) ],
  )
  tracks = Tracks.__new__( Tracks )
  tracks.appState = cast( Any, SimpleNamespace( tracks={ 1: track } ) )
  tracks.refreshPeople = Mock()
  tracks.tblTrackData = Mock()
  tracks.tblTrackData.selection.return_value = ()
  tracks.tblTrackData.get_children.return_value = ()
  tracks.peopleByLabel = {}
  tracks.personLabels = { 17: "Jordan Example" }
  tracks.current_frame = 0
  tracks.refreshing = False
  tracks.selectionChanged = Mock()

  tracks.refresh()
  tracks.current_frame = 5
  tracks.refresh()

  assert tracks.tblTrackData.insert.call_args.kwargs[ "values" ][ 2 ] == "<Unknown>"


def test_person_assignment_updates_only_the_current_track_segment( monkeypatch ):
  participant = PersonParticipation(
      person_id=DBPerson( id=17, first_name="Sam", last_name="Player" ),
      match_id=DBMatch( id=23, date=None, home="A", away="B", division="D" ),
      shirt_number=None,
      role="away_player",
  )
  track = Track(
      clip=1,
      id=8,
      segments=[ TrackSegment( 0, 3 ), TrackSegment( 5, 8 ) ],
  )

  class FakeCombobox:
    def __init__( self, *_args, **_kwargs ):
      self.bindings = {}
      self.selected = ""

    def place( self, **_kwargs ):
      pass

    def set( self, value ):
      self.selected = value

    def get( self ):
      return self.selected

    def bind( self, event, callback ):
      self.bindings[ event ] = callback

    def focus_set( self ):
      pass

    def destroy( self ):
      pass

  monkeypatch.setattr( tracks_module.ttk, "Combobox", FakeCombobox )
  tracks = Tracks.__new__( Tracks )
  tracks.appState = cast( Any, SimpleNamespace( tracks={ 8: track }, db=Mock(), curClipID=1 ) )
  tracks.tab = Mock()
  tracks.tblTrackData = Mock()
  tracks.tblTrackData.identify_row.return_value = "8"
  tracks.tblTrackData.identify_column.return_value = "#3"
  tracks.tblTrackData.bbox.return_value = ( 0, 0, 100, 20 )
  tracks.tblTrackData.winfo_x.return_value = 0
  tracks.tblTrackData.winfo_y.return_value = 0
  tracks.peopleByLabel = { "Sam Player": participant }
  tracks.personLabels = { 17: "Sam Player" }
  tracks.person_options = ( "<Unknown>", "Sam Player" )
  tracks.current_frame = 6
  tracks.editor = None
  tracks.persistSegments = Mock( return_value=True )
  tracks.refresh = Mock()
  tracks.renderSelectedCrops = Mock()
  tracks.on_role_change = None

  tracks.editCell( SimpleNamespace( x=10, y=10 ) )
  tracks.editor.set( "Sam Player" )
  tracks.editor.bindings[ "<<ComboboxSelected>>" ]()

  assert [ segment.person_id for segment in track.segments ] == [ None, 17 ]
  tracks.persistSegments.assert_called_once_with( 8, track.segments )


def test_tracks_person_picker_filters_options_by_typed_text():
  tracks = Tracks.__new__( Tracks )
  tracks.editor = Mock()
  tracks.editor.get.return_value = "keeper"
  tracks.person_options = ( "<Unknown>", "Main Referee", "Opposition Keeper 1", "Player 1" )

  tracks.filterPersonOptions( SimpleNamespace( keysym="r" ) )

  tracks.editor.configure.assert_called_once_with( values=( "Opposition Keeper 1", ) )


def test_selecting_preview_track_refreshes_if_track_is_not_in_table():
  tracks = Tracks.__new__( Tracks )
  track = Track( clip=1, id=11, boxes=[], role="home_player" )
  tracks.appState = cast( Any, SimpleNamespace( tracks={ 11: track } ) )
  tracks.tblTrackData = Mock()
  tracks.tblTrackData.exists.side_effect = ( False, True )
  tracks.refresh = Mock()
  tracks.selectionChanged = Mock()

  tracks.selectTrack( 11 )

  tracks.refresh.assert_called_once_with()
  tracks.tblTrackData.selection_set.assert_called_once_with( "11" )
  tracks.tblTrackData.focus.assert_called_once_with( "11" )
  tracks.tblTrackData.see.assert_called_once_with( "11" )
  tracks.selectionChanged.assert_called_once_with( track )


def test_clip_participant_role_cell_updates_participation( monkeypatch ):
  participant = PersonParticipation(
      person_id=DBPerson( id=17, first_name="Sam", last_name="Player" ),
      match_id=DBMatch( id=23, date=None, home="A", away="B", division="D" ),
      shirt_number=8,
      role="unknown",
  )
  participants = ClipParticipants.__new__( ClipParticipants )
  participants.state = cast( Any, SimpleNamespace( db=Mock(), curClipID=3 ) )
  participants.tab = Mock()
  participants.participants = { 17: participant }
  participants.editor = None
  participants.participantTree = Mock()
  participants.participantTree.identify_row.return_value = "17"
  participants.participantTree.identify_column.return_value = "#4"
  participants.participantTree.bbox.return_value = ( 0, 0, 100, 20 )
  participants.participantTree.winfo_x.return_value = 0
  participants.participantTree.winfo_y.return_value = 0
  participants.refresh = Mock()
  participants.on_role_changed = Mock()

  class FakeCombobox:
    def __init__( self, *_args, **_kwargs ):
      self.bindings = {}
      self.selected = ""

    def place( self, **_kwargs ):
      pass

    def set( self, value ):
      self.selected = value

    def get( self ):
      return self.selected

    def bind( self, event, callback ):
      self.bindings[ event ] = callback

    def focus_set( self ):
      pass

    def destroy( self ):
      pass

  monkeypatch.setattr( configuration_module.ttk, "Combobox", FakeCombobox )
  upsert = Mock()
  monkeypatch.setattr( configuration_module, "upsertPersonParticipation", upsert )

  participants.editRole( SimpleNamespace( x=20, y=10 ) )
  assert isinstance( participants.editor, FakeCombobox )
  participants.editor.set( "referee" )
  participants.editor.bindings[ "<<ComboboxSelected>>" ]()

  upsert.assert_called_once()
  saved = upsert.call_args.args[ 1 ]
  assert saved.person_id == 17
  assert saved.match_id == 23
  assert saved.role == "referee"
  assert participant.role == "referee"
  participants.refresh.assert_called_once_with()
  participants.on_role_changed.assert_called_once_with()


def test_role_assignment_updates_participation_role(monkeypatch):
  participant = PersonParticipation(
      person_id=DBPerson( id=17, first_name="Unknown", last_name="Player 17" ),
      match_id=DBMatch( id=23, date=None, home="A", away="B", division="D" ),
      shirt_number=8,
      role="unknown",
      is_placeholder=True,
  )
  tracks = Tracks.__new__( Tracks )
  tracks.peopleByLabel = cast( Any, { "Player": participant } )
  tracks.appState = cast( Any, SimpleNamespace( db=Mock() ) )
  tracks.tab = Mock()
  upsert = Mock()
  monkeypatch.setattr( "soccer_homography.ui.config.tracks.upsertPersonParticipation", upsert )

  assert tracks.updateParticipantRole( participant, "home_player" )

  upsert.assert_called_once()
  participation = upsert.call_args.args[ 1 ]
  assert participation.match_id == 23
  assert participation.person_id == 17
  assert participation.shirt_number == 8
  assert participation.role == "home_player"
  assert participation.is_placeholder is True
  assert participant.role == "home_player"


def test_role_assignment_updates_known_participation_role(monkeypatch):
  participant = PersonParticipation(
      person_id=DBPerson( id=17, first_name="Sam", last_name="Player" ),
      match_id=DBMatch( id=23, date=None, home="A", away="B", division="D" ),
      shirt_number=8,
      role="away_player",
      is_placeholder=False,
  )
  tracks = Tracks.__new__( Tracks )
  tracks.peopleByLabel = cast( Any, { "Player": participant } )
  tracks.appState = cast( Any, SimpleNamespace( db=Mock() ) )
  tracks.tab = Mock()
  upsert = Mock()
  monkeypatch.setattr( "soccer_homography.ui.config.tracks.upsertPersonParticipation", upsert )

  assert tracks.updateParticipantRole( participant, "home_player" )

  upsert.assert_called_once()
  assert participant.role == "home_player"


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
  app.curTrackID = None
  app.participationRoles = {}
  app.appState = state
  app.pendingDetectionChunks = set()
  app.pendingTrackingChunks = set()
  app.root = Mock()
  app.tabData = SimpleNamespace(
      tabTracks=SimpleNamespace( onClipLoaded=Mock(), updateCurrentFrame=Mock(), refresh=Mock() ),
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


def test_track_segment_navigation_moves_to_current_boundary_then_adjacent_segment():
  track = Track(
      clip=1,
      id=8,
      segments=[ TrackSegment( 0, 10 ), TrackSegment( 20, 30 ), TrackSegment( 40, 45 ) ],
  )
  app = cast( Any, App.__new__( App ) )
  app.curTrackID = 8
  app.appState = SimpleNamespace( tracks={ 8: track } )
  app.mainImageController = SimpleNamespace( frame_num=4 )
  app.sldVideoFrame = Mock()

  app.navigateTrackSegment( 1 )
  app.sldVideoFrame.setValue.assert_called_once_with( 10 )

  app.sldVideoFrame.reset_mock()
  app.mainImageController.frame_num = 10
  app.navigateTrackSegment( 1 )
  app.sldVideoFrame.setValue.assert_called_once_with( 20 )

  app.sldVideoFrame.reset_mock()
  app.mainImageController.frame_num = 15
  app.navigateTrackSegment( -1 )
  app.sldVideoFrame.setValue.assert_called_once_with( 10 )

  app.sldVideoFrame.reset_mock()
  app.mainImageController.frame_num = 20
  app.navigateTrackSegment( -1 )
  app.sldVideoFrame.setValue.assert_called_once_with( 10 )


def test_splitting_track_segment_includes_current_frame_in_left_half():
  track = Track(
      clip=1,
      id=8,
      segments=[ TrackSegment( 0, 10, 17 ) ],
  )
  app = cast( Any, App.__new__( App ) )
  app.curTrackID = 8
  app.appState = SimpleNamespace( tracks={ 8: track } )
  app.mainImageController = SimpleNamespace( frame_num=5 )
  app.persistTrackSegments = Mock( return_value=True )
  app.tabData = SimpleNamespace( tabTracks=Mock() )
  app.minimap = Mock()
  app.updateLivePreviewMappings = Mock()

  app.splitTrackSegment()

  assert [ ( item.frame_start, item.frame_end, item.person_id ) for item in track.segments ] == [
      ( 0, 5, 17 ),
      ( 6, 10, 17 ),
  ]
  app.persistTrackSegments.assert_called_once_with( 8, track.segments )
  app.minimap.setTrackSegments.assert_called_once_with( track.segments )
  app.updateLivePreviewMappings.assert_called_once_with( 5 )


def test_crop_coordinates_scale_to_source_frame_dimensions():
  class SourceCapture:
    def __init__( self ):
      self.frame = np.zeros( ( 400, 400, 3 ), dtype=np.uint8 )
      self.frame[ 10:20, 10:20 ] = ( 0, 0, 255 )

    def set( self, _property_id, _frame_number ):
      self.position = ( _property_id, _frame_number )

    def read( self ):
      return True, self.frame

  box = BoundingBox( 10, 10, 20, 20, 0.9, 0, 0 )

  crops = cropFromFrame( SourceCapture().frame, box )

  assert crops is not None
  assert crops.shape[:2] == ( 10, 10 )
  assert np.all( crops[ :, :, 0 ] == 255 )


def test_crop_from_frame_converts_fractional_tracker_coordinates_to_slice_indices():
  frame = np.zeros( ( 40, 40, 3 ), dtype=np.uint8 )
  frame[ 10:20, 10:20 ] = ( 0, 0, 255 )
  box = BoundingBox( 10.5, 10.5, 20.5, 20.5, 0.9, 0, 0 )

  crop = cropFromFrame( frame, box )

  assert crop is not None
  assert crop.shape[:2] == ( 10, 10 )
  assert np.all( crop[ :, :, 0 ] == 255 )


def test_crop_frame_plan_prioritizes_frames_covering_more_tracks():
  box = lambda frame: BoundingBox( 2, 2, 12, 12, 0.9, 0, frame )
  tracks = [
      ( 1, [ box( frame ) for frame in ( 5, 10, 15 ) ] ),
      ( 2, [ box( frame ) for frame in ( 5, 11 ) ] ),
      ( 3, [ box( frame ) for frame in ( 5, 12 ) ] ),
      ( 4, [ box( frame ) for frame in ( 10, 11, 12 ) ] ),
  ]

  plan = planCropFrames( tracks )

  assert plan[ 0 ].frameNumber == 5
  assert { segment_key[ 0 ] for segment_key, _box in plan[ 0 ].trackBoxes } == { 1, 2, 3 }


def test_crop_frame_plan_spreads_samples_across_track_timeline():
  box = lambda frame: BoundingBox( 2, 2, 12, 12, 0.9, 0, frame )
  plan = planCropFrames( [ ( 1, [ box( frame ) for frame in range( 101 ) ] ) ] )

  sampled_frames = [ planned.frameNumber for planned in plan[ :6 ] ]

  assert min( abs( left - right ) for index, left in enumerate( sampled_frames ) for right in sampled_frames[ index + 1: ] ) >= 10
  assert min( sampled_frames ) == 0
  assert max( sampled_frames ) == 100


def test_crop_worker_plans_first_and_reads_each_selected_frame_once(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )
  operations: list[ str ] = []

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
      operations.append( "read" )
      frame = np.zeros( ( 40, 40, 3 ), dtype=np.uint8 )
      frame[ :, :, 0 ] = self.frame_number
      return True, frame

    def release( self ):
      pass

  capture = SourceCapture()
  def open_capture( _video_file ):
    operations.append( "open" )
    return capture

  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cv2.VideoCapture", open_capture )
  original_plan = planCropFrames

  def record_plan( tracks, max_crops=6, cancel_event=None ):
    operations.append( "plan" )
    return original_plan( tracks, max_crops, cancel_event )

  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.planCropFrames", record_plan )
  results = queue.Queue()
  box = lambda frame: BoundingBox( 2, 2, 12, 12, 0.9, 0, frame )
  unknown_tracks = [
      ( 1, [ box( frame ) for frame in range( 10 ) ] ),
      ( 2, [ box( frame ) for frame in range( 10 ) ] ),
      ( 3, [ box( 1 ) ] ),
      ( 4, [ box( frame ) for frame in range( 1, 9 ) ] ),
  ]

  worker = CropExtractionWorker( 7, "video.mp4", unknown_tracks, results, clip_id=12 )
  worker.run()

  messages = []
  while not results.empty():
    messages.append( results.get_nowait() )
  result = next( message for message in messages if message.kind == "done" )

  assert operations[ 0:2 ] == [ "plan", "open" ]
  assert len( capture.seeks ) == len( set( capture.seeks ) )
  assert len( capture.seeks ) <= 10
  assert len( result.crops[ ( 1, 0 ) ] ) == 6
  assert len( result.crops[ ( 2, 0 ) ] ) == 6
  assert [ crop[ 0 ] for crop in result.crops[ ( 3, 1 ) ] ] == [ 1 ]
  assert len( result.crops[ ( 4, 1 ) ] ) == 6


def test_crop_worker_enforces_crop_limit_independently_for_each_segment(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )

  class SourceCapture:
    def __init__( self ):
      self.frame_number = 0
      self.seeks = []

    def isOpened( self ):
      return True

    def set( self, _property_id, frame_number ):
      self.frame_number = frame_number
      self.seeks.append( frame_number )

    def read( self ):
      frame = np.zeros( ( 20, 20, 3 ), dtype=np.uint8 )
      return True, frame

    def release( self ):
      pass

  capture = SourceCapture()
  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cv2.VideoCapture", Mock( return_value=capture ) )
  box = lambda frame: BoundingBox( 2, 2, 12, 12, 0.9, 0, frame )
  segments = [
      ( 7, 0, [ box( frame ) for frame in range( 0, 10 ) ] ),
      ( 7, 20, [ box( frame ) for frame in range( 20, 30 ) ] ),
  ]
  results = queue.Queue()

  CropExtractionWorker( 12, "video.mp4", segments, results, clip_id=15, max_crops=3 ).run()

  message = next( item for item in list( results.queue ) if item.kind == "done" )
  assert len( message.crops[ ( 7, 0 ) ] ) == 3
  assert len( message.crops[ ( 7, 20 ) ] ) == 3
  cache_files = sorted( path.name for path in ( tmp_path / "crops" / "15" ).glob( "*.png" ) )
  assert len( cache_files ) == 6
  assert all( name.startswith( ( "7_0_", "7_20_" ) ) for name in cache_files )
  assert len( capture.seeks ) <= 6


def test_crop_worker_uses_planned_fallback_frames_after_read_failures(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )

  class SourceCapture:
    def __init__( self ):
      self.frame_number = 0
      self.read_count = 0

    def isOpened( self ):
      return True

    def set( self, _property_id, frame_number ):
      self.frame_number = frame_number

    def read( self ):
      self.read_count += 1
      if self.read_count <= 2:
        return False, None
      frame = np.zeros( ( 40, 40, 3 ), dtype=np.uint8 )
      return True, frame

    def release( self ):
      pass

  capture = SourceCapture()
  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cv2.VideoCapture", Mock( return_value=capture ) )
  results = queue.Queue()
  box = lambda frame: BoundingBox( 2, 2, 12, 12, 0.9, 0, frame )
  worker = CropExtractionWorker( 8, "video.mp4", [ ( 1, [ box( frame ) for frame in range( 8 ) ] ) ], results, clip_id=12 )

  worker.run()

  messages = []
  while not results.empty():
    messages.append( results.get_nowait() )
  result = next( message for message in messages if message.kind == "done" )
  assert capture.read_count == 8
  assert len( result.crops[ ( 1, 0 ) ] ) == 6


def test_crop_worker_cancellation_prevents_video_io(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )

  def unexpected_video_open( _video_file ):
    raise AssertionError( "Cancelled crop work must not open the video." )

  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cv2.VideoCapture", unexpected_video_open )
  box = BoundingBox( 2, 2, 12, 12, 0.9, 0, 0 )
  results = queue.Queue()
  worker = CropExtractionWorker( 9, "video.mp4", [ ( 1, [ box ] ) ], results, clip_id=12 )
  worker.cancel()

  worker.run()

  assert results.empty()


def test_crop_worker_uses_cached_png_without_opening_video(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )
  box = BoundingBox( 2, 2, 12, 12, 0.9, 0, 5 )
  cached_crop = np.full( ( 10, 10, 3 ), ( 11, 29, 47 ), dtype=np.uint8 )
  crop_directory = tmp_path / "crops" / "12"
  crop_directory.mkdir( parents=True )
  assert cv2.imwrite(
      str( crop_directory / "1_5_5.png" ),
      cv2.cvtColor( cached_crop, cv2.COLOR_RGB2BGR ),
  )

  def unexpected_video_open( _video_file ):
    raise AssertionError( "A fully cached crop must not open the video." )

  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cv2.VideoCapture", unexpected_video_open )
  results = queue.Queue()
  worker = CropExtractionWorker( 10, "video.mp4", [ ( 1, [ box ] ) ], results, clip_id=12 )

  worker.run()

  messages = []
  while not results.empty():
    messages.append( results.get_nowait() )
  result = next( message for message in messages if message.kind == "done" )
  assert len( result.crops[ ( 1, 5 ) ] ) == 1
  assert result.crops[ ( 1, 5 ) ][ 0 ][ 0 ] == 5
  assert np.array_equal( result.crops[ ( 1, 5 ) ][ 0 ][ 1 ], cached_crop )


def test_crop_worker_ignores_disk_crops_outside_updated_segment_bounds(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )
  crop_directory = tmp_path / "crops" / "12"
  crop_directory.mkdir( parents=True )
  stale = np.full( ( 10, 10, 3 ), 99, dtype=np.uint8 )
  cv2.imwrite( str( crop_directory / "1_5_8.png" ), cv2.cvtColor( stale, cv2.COLOR_RGB2BGR ) )

  class SourceCapture:
    def isOpened( self ):
      return True

    def set( self, _property_id, _frame_number ):
      pass

    def read( self ):
      frame = np.zeros( ( 20, 20, 3 ), dtype=np.uint8 )
      frame[ 2:12, 2:12 ] = ( 0, 0, 255 )
      return True, frame

    def release( self ):
      pass

  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cv2.VideoCapture", Mock( return_value=SourceCapture() ) )
  results = queue.Queue()
  boxes = [ BoundingBox( 2, 2, 12, 12, 0.9, 0, frame ) for frame in ( 5, 6 ) ]

  CropExtractionWorker( 13, "video.mp4", [ ( 1, 5, boxes ) ], results, clip_id=12, max_crops=2 ).run()

  message = next( item for item in list( results.queue ) if item.kind == "done" )
  assert [ frame for frame, _ in message.crops[ ( 1, 5 ) ] ] == [ 5, 6 ]
  assert not any( np.array_equal( crop, stale ) for _frame, crop in message.crops[ ( 1, 5 ) ] )


def test_crop_worker_seeks_for_only_tracks_missing_from_disk_cache(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )
  cached_crop = np.full( ( 10, 10, 3 ), ( 11, 29, 47 ), dtype=np.uint8 )
  crop_directory = tmp_path / "crops" / "12"
  crop_directory.mkdir( parents=True )
  assert cv2.imwrite(
      str( crop_directory / "1_5_5.png" ),
      cv2.cvtColor( cached_crop, cv2.COLOR_RGB2BGR ),
  )

  class SourceCapture:
    def __init__( self ):
      self.seeks: list[ int ] = []
      self.frame = np.zeros( ( 20, 20, 3 ), dtype=np.uint8 )
      self.frame[ 2:12, 2:12 ] = ( 0, 0, 255 )

    def isOpened( self ):
      return True

    def set( self, _property_id, frame_number ):
      self.seeks.append( frame_number )

    def read( self ):
      return True, self.frame

    def release( self ):
      pass

  capture = SourceCapture()
  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cv2.VideoCapture", Mock( return_value=capture ) )
  extracted_tracks = []
  original_crop_from_frame = cropFromFrame

  def record_extracted_crop( frame, box ):
    extracted_tracks.append( box.frame )
    return original_crop_from_frame( frame, box )

  monkeypatch.setattr( "soccer_homography.ui.config.crop_worker.cropFromFrame", record_extracted_crop )
  results = queue.Queue()
  boxes = [
      BoundingBox( 2, 2, 12, 12, 0.9, 0, 5 ),
      BoundingBox( 2, 2, 12, 12, 0.9, 0, 5 ),
  ]
  worker = CropExtractionWorker( 11, "video.mp4", [ ( 1, [ boxes[ 0 ] ] ), ( 2, [ boxes[ 1 ] ] ) ], results, clip_id=12 )

  worker.run()

  messages = []
  while not results.empty():
    messages.append( results.get_nowait() )
  result = next( message for message in messages if message.kind == "done" )
  assert capture.seeks == [ 5 ]
  assert extracted_tracks == [ 5 ]
  assert np.array_equal( result.crops[ ( 1, 5 ) ][ 0 ][ 1 ], cached_crop )
  assert np.array_equal( result.crops[ ( 2, 5 ) ][ 0 ][ 1 ], np.full( ( 10, 10, 3 ), ( 255, 0, 0 ), dtype=np.uint8 ) )
  saved_crop = cv2.imread( str( crop_directory / "2_5_5.png" ), cv2.IMREAD_COLOR )
  assert saved_crop is not None
  assert np.array_equal( cv2.cvtColor( saved_crop, cv2.COLOR_BGR2RGB ), result.crops[ ( 2, 5 ) ][ 0 ][ 1 ] )


def test_slider_set_value_updates_position_and_runs_frame_callback():
  slider = Slider.__new__( Slider )
  slider.min = 0
  slider.max = 100
  slider.root = Mock()
  slider._debounce_job = "pending"
  slider.interSnap = False
  slider.slider = Mock()
  slider.boundVar = Mock()
  slider.lblRadar = Mock()
  slider.command = Mock()

  slider.setValue( 42 )

  slider.root.after_cancel.assert_called_once_with( "pending" )
  slider.slider.set.assert_called_once_with( 42 )
  slider.boundVar.set.assert_called_once_with( 42 )
  slider.lblRadar.config.assert_called_once_with( text="42" )
  slider.command.assert_called_once_with( 42 )
  assert slider._debounce_job is None
  assert slider.interSnap is False


def test_clicking_crop_sends_its_frame_to_the_main_slider_callback():
  tracks = Tracks.__new__( Tracks )
  tracks.frameSelectCallback = Mock()

  tracks.selectCropFrame( 73 )

  tracks.frameSelectCallback.assert_called_once_with( 73 )


def test_crop_table_lists_each_frame_and_known_identification_guess():
  tracks = Tracks.__new__( Tracks )
  tracks.cropIdentificationResults = {
      9: { 73: IdentificationImageResult( frame_number=73, role="home_player" ) }
  }
  tracks.cropTable = Mock()
  crops = [
      ( 73, np.zeros( ( 8, 8, 3 ), dtype=np.uint8 ) ),
      ( 81, np.ones( ( 8, 8, 3 ), dtype=np.uint8 ) ),
  ]

  tracks.displayCrops( 9, crops )

  assert [ call.kwargs[ "iid" ] for call in tracks.cropTable.insert.call_args_list ] == [ "73", "81" ]
  assert [ call.kwargs[ "values" ] for call in tracks.cropTable.insert.call_args_list ] == [
      ( 73, "home player" ),
      ( 81, "Not analyzed" ),
  ]


def test_selecting_crop_displays_its_image_and_identification_guess( monkeypatch ):
  tracks = Tracks.__new__( Tracks )
  tracks.selTrackID = 9
  crop = np.full( ( 8, 8, 3 ), 17, dtype=np.uint8 )
  tracks.cropCache = { ( 9, 0 ): [ ( 73, crop ) ] }
  tracks.cropIdentificationResults = {
      9: { 73: IdentificationImageResult( frame_number=73, role="away_goalkeeper" ) }
  }
  tracks.cropTable = Mock()
  tracks.cropTable.selection.return_value = ( "73", )
  tracks.cropPreview = Mock()
  tracks.cropPreviewCaption = Mock()
  tracks.frameSelectCallback = Mock()
  photo_image = Mock( return_value="photo" )
  monkeypatch.setattr( "soccer_homography.ui.config.tracks.ImageTk.PhotoImage", photo_image )

  tracks.onCropSelected()

  assert np.array_equal( np.asarray( photo_image.call_args.args[ 0 ] ), crop )
  tracks.cropPreview.config.assert_called_once_with( image="photo", text="" )
  tracks.cropPreviewCaption.config.assert_called_once_with( text="Frame 73 - away goalkeeper" )
  tracks.frameSelectCallback.assert_called_once_with( 73 )


def test_vlm_button_requires_enabled_crop_action_and_selected_track_crops():
  tracks = Tracks.__new__( Tracks )
  cast( Any, tracks ).appState = SimpleNamespace(
      curClipID=4,
      tracks={ 9: Track( clip=1, id=9, segments=[ TrackSegment( 10, 20 ) ] ) },
  )
  tracks.current_frame = 12
  tracks.cropCacheClipID = 4
  tracks.cropCache = { ( 9, 10 ): [ ( 13, np.zeros( ( 8, 8, 3 ), dtype=np.uint8 ) ) ] }
  tracks.selTrackID = 9
  tracks.cropsButtonEnabled = False
  tracks.cropInferenceWorker = None
  tracks.vlmButton = Mock()

  tracks.updateVLMButtonState()
  tracks.vlmButton.config.assert_called_once_with( state="disabled" )

  tracks.updateVLMButtonState( crops_enabled=True )
  tracks.vlmButton.config.assert_called_with( state="normal" )


def test_crop_collection_sends_each_unassigned_segment_and_uses_model_option_limit( monkeypatch ):
  box = lambda frame: BoundingBox( 2, 2, 12, 12, 0.9, 0, frame )
  track = Track(
      clip=4,
      id=9,
      boxes=[ box( frame ) for frame in range( 10 ) ],
      segments=[ TrackSegment( 0, 4 ), TrackSegment( 5, 9, 17 ) ],
  )
  tracks = Tracks.__new__( Tracks )
  tracks.appState = SimpleNamespace(
      curClipID=4,
      videoFile="video.mp4",
      tracks={ 9: track },
  )
  tracks.tab = Mock()
  tracks.peopleByLabel = {}
  tracks.personLabels = {}
  tracks.person_options = ()
  tracks.refreshPeople = Mock()
  tracks.cancelCropJob = Mock()
  tracks.cancelVLMJob = Mock()
  tracks.cropCache = {}
  tracks.cropIdentificationResults = {}
  tracks.cropWorker = None
  tracks.updateVLMButtonState = Mock()
  tracks.clearCropImages = Mock()
  tracks.renderSelectedCrops = Mock()
  tracks.cropCacheClipID = None
  tracks.cropGeneration = 0
  tracks.cropResults = queue.Queue()
  tracks.cropProgress = Mock()
  tracks.cropStatus = Mock()
  tracks.cropCountProvider = Mock( return_value=3 )
  worker = Mock()
  monkeypatch.setattr( tracks_module, "CropExtractionWorker", worker )

  tracks.collectCrops()

  args = worker.call_args
  assert [ ( track_id, start, [ box.frame for box in boxes ] ) for track_id, start, boxes in args.args[ 2 ] ] == [
      ( 9, 0, [ 0, 1, 2, 3, 4 ] ),
  ]
  assert args.kwargs[ "max_crops" ] == 3
  worker.return_value.start.assert_called_once_with()


def test_model_options_provides_configured_crops_per_segment():
  options = ModelOptions.__new__( ModelOptions )
  options.modelOpts = SimpleNamespace( cropsPerSegment=Mock( get=Mock( return_value=12 ) ) )

  assert options.getCropsPerSegment() == 12


def test_sports_tracker_releases_inference_resources( monkeypatch ):
  cap = Mock()
  tracker = cast( Any, SportsTracker.__new__( SportsTracker ) )
  tracker.cap = cap
  tracker.model = object()
  tracker.tracker = object()
  tracker.inBoxes = { 10: [ object() ] }
  empty_cache = Mock()
  monkeypatch.setitem(
      sys.modules,
      "torch",
      SimpleNamespace( cuda=SimpleNamespace( is_available=lambda: True, empty_cache=empty_cache ) ),
  )

  tracker.releaseResources()

  cap.release.assert_called_once_with()
  assert tracker.model is None
  assert tracker.tracker is None
  assert tracker.inBoxes == {}
  empty_cache.assert_called_once_with()


def test_moondream_vlm_queries_rgb_crop_and_role_votes():
  class FakeModel:
    def __init__( self ):
      self.images = []
      self.prompts = []

    def query( self, image, prompt ):
      self.images.append( image )
      self.prompts.append( prompt )
      return { "answer": "home_player" }

  model = MoondreamVLM()
  fake_model = FakeModel()
  model.model = fake_model
  image = np.zeros( ( 10, 10, 3 ), dtype=np.uint8 )

  response = model.query( image, "Identify the role." )
  assert response == VLMResponse( role="home_player", confidence=None, answer="home_player" )
  assert fake_model.images[ 0 ].size == ( 10, 10 )
  worker = CropInferenceWorker(
      1,
      2,
      3,
      [ ( 1, image ) ],
      "Identify the role.",
      model,
      queue.Queue(),
  )
  assert "home_goalkeeper" in worker.prompt
  assert guessRole( [ "home_player", "This looks like a home_player." ] ) == "home_player"
  assert guessRole( [ "home_player", "away_player" ] ) is None


def test_moondream_selects_single_device_and_avoids_tight_gpu_memory():
  cpu_only = SimpleNamespace( cuda=SimpleNamespace( is_available=lambda: False ) )
  low_memory = SimpleNamespace(
      cuda=SimpleNamespace(
          is_available=lambda: True,
          mem_get_info=lambda _device: ( MIN_VLM_GPU_MEMORY_BYTES - 1, MIN_VLM_GPU_MEMORY_BYTES ),
      )
  )
  enough_memory = SimpleNamespace(
      cuda=SimpleNamespace(
          is_available=lambda: True,
          mem_get_info=lambda _device: ( MIN_VLM_GPU_MEMORY_BYTES, MIN_VLM_GPU_MEMORY_BYTES ),
      )
  )

  assert selectModelDevice( cpu_only, MIN_VLM_GPU_MEMORY_BYTES ) == "cpu"
  assert selectModelDevice( low_memory, MIN_VLM_GPU_MEMORY_BYTES ) == "cpu"
  assert selectModelDevice( enough_memory, MIN_VLM_GPU_MEMORY_BYTES ) == "cuda:0"


def test_moondream_load_uses_the_selected_single_device(monkeypatch):
  loader = Mock( return_value=object() )
  transformers_stub = SimpleNamespace(
      AutoModelForCausalLM=SimpleNamespace( from_pretrained=loader )
  )
  monkeypatch.setitem( sys.modules, "transformers", transformers_stub )
  monkeypatch.setattr( "soccer_homography.inference.clip_model.selectModelDevice", lambda _torch, _min_memory: "cpu" )
  model = MoondreamVLM()

  loaded = model.loadModel()

  assert loaded is loader.return_value
  loader.assert_called_once_with(
      "vikhyatk/moondream2",
      trust_remote_code=True,
      device_map="cpu",
  )


def test_identification_prompt_loads_default_or_saved_multiline_content(tmp_path):
  prompt_path = tmp_path / "identificationPrompt.txt"

  assert loadIdentificationPrompt( prompt_path ) == DEFAULT_IDENTIFICATION_PROMPT

  prompt = "Home team wears stripes.\nThe referee wears yellow."
  saveIdentificationPrompt( prompt, prompt_path )

  assert prompt_path.read_text( encoding="utf-8" ) == prompt + "\n"
  assert loadIdentificationPrompt( prompt_path ) == prompt


def test_moondream_response_ignores_confidence():
  class FakeModel:
    def query( self, _image, prompt ):
      self.prompt = prompt
      return { "role": None, "answer": "referee", "confidence": 0.94 }

  model = MoondreamVLM()
  model.model = FakeModel()
  image = np.zeros( ( 10, 10, 3 ), dtype=np.uint8 )

  assert model.query( image, "Identify the role." ) == VLMResponse( role="referee", confidence=None, answer="referee" )


def test_identification_guess_shows_confidence_only_for_clip():
  clip_result = ClipImageResult( 12, "home_player", 0.94 )
  vlm_result = VLMImageResult( 20, "referee", None )

  assert Tracks.formatIdentificationGuess( clip_result ) == "home player (94% confidence)"
  assert Tracks.formatIdentificationGuess( vlm_result ) == "referee"


def test_clip_role_classifier_returns_top_role_and_softmax_confidence():
  import torch

  class FakeProcessor:
    def __call__( self, *, text, images, return_tensors, padding ):
      assert len( text ) == 6
      assert images.size == ( 10, 10 )
      assert return_tensors == "pt"
      assert padding is True
      return { "pixel_values": torch.zeros( ( 1, 3, 10, 10 ) ) }

  class FakeModel:
    def __call__( self, **_inputs ):
      return SimpleNamespace( logits_per_image=torch.tensor( [ [ 0.0, 0.0, 0.0, 0.0, 5.0, 0.0 ] ] ) )

  model = ClipRoleClassifier()
  model.model = FakeModel()
  model.processor = FakeProcessor()
  image = np.zeros( ( 10, 10, 3 ), dtype=np.uint8 )

  result = model.query( image )

  assert result.role == "referee"
  assert result.confidence is not None
  assert 0.9 < result.confidence < 1.0


def test_clip_device_selection_falls_back_when_gpu_memory_is_low():
  cpu_only = SimpleNamespace( cuda=SimpleNamespace( is_available=lambda: False ) )
  low_memory = SimpleNamespace( cuda=SimpleNamespace( is_available=lambda: True, mem_get_info=lambda _device: ( 0, 0 ) ) )
  enough_memory = SimpleNamespace(
      cuda=SimpleNamespace( is_available=lambda: True, mem_get_info=lambda _device: ( 2 * 1024**3, 4 * 1024**3 ) )
  )

  assert selectModelDevice( cpu_only, MIN_CLIP_GPU_MEMORY_BYTES ) == "cpu"
  assert selectModelDevice( low_memory, MIN_CLIP_GPU_MEMORY_BYTES ) == "cpu"
  assert selectModelDevice( enough_memory, MIN_CLIP_GPU_MEMORY_BYTES ) == "cuda:0"


def test_clip_model_load_uses_one_selected_device( monkeypatch ):
  model_loader = Mock( return_value=object() )
  processor_loader = Mock( return_value=object() )
  transformers_stub = SimpleNamespace(
      CLIPModel=SimpleNamespace( from_pretrained=model_loader ),
      CLIPProcessor=SimpleNamespace( from_pretrained=processor_loader ),
  )
  monkeypatch.setitem( sys.modules, "transformers", transformers_stub )
  monkeypatch.setattr( "soccer_homography.inference.clip_model.selectModelDevice", lambda _torch, _min_memory: "cpu" )
  classifier = ClipRoleClassifier()

  model, processor, device = classifier.loadModel()

  assert model is model_loader.return_value
  assert processor is processor_loader.return_value
  assert device == "cpu"
  model_loader.assert_called_once_with( "openai/clip-vit-base-patch32", device_map={ "": "cpu" } )
  processor_loader.assert_called_once_with( "openai/clip-vit-base-patch32" )


def test_vlm_worker_returns_per_crop_responses_and_role_guess():
  class FakeVLM(MoondreamVLM):
    def query( self, image, prompt ):
      return VLMResponse( role="referee", confidence=None, answer="referee" )

  results = queue.Queue()
  crops = [
      ( 12, np.zeros( ( 8, 8, 3 ), dtype=np.uint8 ) ),
      ( 20, np.zeros( ( 8, 8, 3 ), dtype=np.uint8 ) ),
  ]
  worker = CropInferenceWorker( 3, 7, 8, crops, "Identify the role.", FakeVLM(), results )

  worker.run()

  messages = []
  while not results.empty():
    messages.append( results.get_nowait() )
  result = next( message for message in messages if message.kind == "done" )
  assert [ answer.answer for answer in result.answers ] == [ "referee", "referee" ]
  assert result.role == "referee"
  assert result.vote_counts[ "referee" ] == 2


def test_identification_worker_votes_across_clip_crops_and_continues():
  class FakeClip( ClipRoleClassifier ):
    def __init__( self ):
      super().__init__()
      self.index = 0

    def query( self, image, prompt="" ):
      self.index += 1
      return ClipResponse( "home_player" if self.index <= 2 else "referee", 0.95 )

  results = queue.Queue()
  crops = [ ( number, np.zeros( ( 8, 8, 3 ), dtype=np.uint8 ) ) for number in ( 12, 20, 30 ) ]
  worker = CropInferenceWorker( 5, 7, 8, crops, "", FakeClip(), results )

  worker.run()

  messages = []
  while not results.empty():
    messages.append( results.get_nowait() )
  done = next( message for message in messages if message.kind == "done" )
  assert done.kind == "done"
  assert len( done.answers ) == 3
  assert done.role == "home_player"
  assert done.vote_counts[ "home_player" ] == 2
  assert done.vote_counts[ "referee" ] == 1
  assert all( result.confidence == 0.95 for result in done.answers )


def test_role_vote_tie_has_no_unique_winner():
  counts = roleVoteCounts(
      [
          ClipImageResult( 12, "home_player", 0.8 ),
          ClipImageResult( 20, "referee", 0.7 ),
      ]
  )

  assert counts[ "home_player" ] == 1
  assert counts[ "referee" ] == 1
  assert mostLikelyRole( counts ) is None
  report = Tracks.formatIdentificationReport( None, counts, [] )
  assert "Most likely: tie" in report
  assert "home player: 1" in report
  assert "referee: 1" in report
  clip_report = Tracks.formatIdentificationReport(
      "home_player",
      counts,
      [ ClipImageResult( 12, "home_player", 0.8 ) ],
  )
  assert "Mean CLIP confidence for most likely role: 80%" in clip_report


def test_view_change_redraws_detection_and_track_overlays():
  app = cast( Any, App.__new__( App ) )
  app.curTrackID = None
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
  app.mainImageController.updateTracks.assert_called_once_with( app.appState.tracks, 23, None )
