import queue
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import cv2
import numpy as np

from soccer_homography import App
from soccer_homography.data import BoundingBox, Homography, Person, Track
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
  assert { track_id for track_id, _box in plan[ 0 ].trackBoxes } == { 1, 2, 3 }


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
  assert len( result.crops[ 1 ] ) == 6
  assert len( result.crops[ 2 ] ) == 6
  assert [ crop[ 0 ] for crop in result.crops[ 3 ] ] == [ 1 ]
  assert len( result.crops[ 4 ] ) == 6


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
  assert len( result.crops[ 1 ] ) == 6


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
      str( crop_directory / "5_1.png" ),
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
  assert len( result.crops[ 1 ] ) == 1
  assert result.crops[ 1 ][ 0 ][ 0 ] == 5
  assert np.array_equal( result.crops[ 1 ][ 0 ][ 1 ], cached_crop )


def test_crop_worker_seeks_for_only_tracks_missing_from_disk_cache(monkeypatch, tmp_path):
  monkeypatch.chdir( tmp_path )
  cached_crop = np.full( ( 10, 10, 3 ), ( 11, 29, 47 ), dtype=np.uint8 )
  crop_directory = tmp_path / "crops" / "12"
  crop_directory.mkdir( parents=True )
  assert cv2.imwrite(
      str( crop_directory / "5_1.png" ),
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
  assert np.array_equal( result.crops[ 1 ][ 0 ][ 1 ], cached_crop )
  assert np.array_equal( result.crops[ 2 ][ 0 ][ 1 ], np.full( ( 10, 10, 3 ), ( 255, 0, 0 ), dtype=np.uint8 ) )
  saved_crop = cv2.imread( str( crop_directory / "5_2.png" ), cv2.IMREAD_COLOR )
  assert saved_crop is not None
  assert np.array_equal( cv2.cvtColor( saved_crop, cv2.COLOR_BGR2RGB ), result.crops[ 2 ][ 0 ][ 1 ] )


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
  tracks.cropCache = { 9: [ ( 73, crop ) ] }
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
  cast( Any, tracks ).appState = SimpleNamespace( curClipID=4 )
  tracks.cropCacheClipID = 4
  tracks.cropCache = { 9: [ ( 13, np.zeros( ( 8, 8, 3 ), dtype=np.uint8 ) ) ] }
  tracks.selTrackID = 9
  tracks.cropsButtonEnabled = False
  tracks.cropInferenceWorker = None
  tracks.vlmButton = Mock()

  tracks.updateVLMButtonState()
  tracks.vlmButton.config.assert_called_once_with( state="disabled" )

  tracks.updateVLMButtonState( crops_enabled=True )
  tracks.vlmButton.config.assert_called_with( state="normal" )


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
