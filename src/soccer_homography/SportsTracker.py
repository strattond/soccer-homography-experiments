import gc
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

import cv2
import numpy as np
from boxmot import OccluBoost
from boxmot.trackers import Tracker
from ultralytics import YOLO

from soccer_homography.appState import ModelOptions
from soccer_homography.data import BoundingBox, Homography, TrackData
from soccer_homography.detectionadapter import DetectionAdapter
from soccer_homography.log import logger

PLAYER_CLASS_ID = 0


class CommandType( Enum ):
  RUN_BBOX = auto()
  FEED_BBOX = auto()
  RUN_TRACK = auto()
  SET_CLIP_ID = auto()
  PAUSE = auto()
  RESUME = auto()
  STOP = auto()
  SEEK = auto()


class OutputType( Enum ):
  BBOX = auto()
  TRACK = auto()
  NEW_FRAME = auto()
  FRAME_BATCH = auto()
  COMPLETED = auto()
  STOP = auto()


@dataclass
class Command:
  type: CommandType
  start: int | None = None
  end: int | None = None
  payload: Any = None


@dataclass
class FrameResult:
  frame: int
  boxes: list[BoundingBox] = field( default_factory=list )
  tracks: list[TrackData] = field( default_factory=list )
  mode: CommandType = CommandType.RUN_BBOX


@dataclass
class Output:
  type: OutputType
  data: BoundingBox | TrackData | int | list[FrameResult] | None = None


TRACK_OUTPUT_BATCH_SIZE = 8


# This will be responsible for loading the model, performing detections and tracking, and so on


@dataclass
class SportsTracker:
  # yapf: disable
  mdlOpts:          ModelOptions
  videoFile:        str
  range:            tuple[int, int]              = field( default_factory=tuple[int, int] )
  index:            int                          = 0
  model:            YOLO | None                  = field( init=False, default=None )
  activeMdlOpts:    ModelOptions | None          = field( init=False, default=None )
  tracker:          Tracker | None                = field( init=False, default=None )
  data:             Homography                   = field( init=False )
  cap:              cv2.VideoCapture             = field( init=False )

  # Threading + communication
  in_queue:         queue.Queue                  = field(default_factory=queue.Queue)
  out_queue:        queue.Queue                  = field(default_factory=queue.Queue)
  paused:           bool                         = True
  stopped:          bool                         = False
  thread:           threading.Thread             = field(init=False)

  # Operational data
  curMode:          CommandType                  = CommandType.PAUSE
  inBoxes:          dict[int, list[BoundingBox]] = field( default_factory=dict )
  curClipID:        int                          = -1
  # yapf: enable

  def __post_init__( self ) -> None:
    self.cap = cv2.VideoCapture( self.videoFile )
    self.thread = threading.Thread( target=self.run, daemon=True )

  def start( self ):
    logger.info( "Starting SportsTracker" )
    self.thread.start()

  def pause( self ):
    logger.info( "Pausing SportsTracker" )
    self.paused = True

  def resume( self ):
    logger.info( "Resuming SportsTracker" )
    self.paused = False

  def stop( self ):
    logger.info( "Stopping SportsTracker" )
    self.paused = False
    self.stopped = True

  def setImagePos( self, pos: int ):
    self.cap.set( cv2.CAP_PROP_POS_FRAMES, pos )

  def processCommands( self ):
    try:
      while True:
        cmd: Command = self.in_queue.get_nowait()

        print( f"Received command Command(type={cmd.type}, start={cmd.start}, end={cmd.end}, payload={'<Data>' if cmd.payload is not None else 'None'})" )
        if cmd.type == CommandType.PAUSE:
          self.pause()
        elif cmd.type == CommandType.RESUME:
          self.resume()
        elif cmd.type == CommandType.STOP:
          self.stop()
        elif cmd.type == CommandType.SEEK:
          if cmd.start is not None:
            self.setImagePos( cmd.start )

        elif cmd.type == CommandType.RUN_BBOX and cmd.start is not None and cmd.end is not None:
          if cmd.end >= int( self.cap.get( cv2.CAP_PROP_FRAME_COUNT ) ):
            cmd.end = int( self.cap.get( cv2.CAP_PROP_FRAME_COUNT ) ) - 1
          self.range = ( cmd.start, cmd.end )
          self.index = cmd.start
          self.activeMdlOpts = replace( self.mdlOpts )
          modelName = "yolo26" + self.activeMdlOpts.size + "." + self.activeMdlOpts.engine
          self.model = YOLO( modelName, verbose=False, task='detect' )
          self.curMode = CommandType.RUN_BBOX
          self.setImagePos( cmd.start )

        elif cmd.type == CommandType.FEED_BBOX and cmd.payload is not None:
          self.inBoxes = cmd.payload

        elif cmd.type == CommandType.RUN_TRACK and cmd.start is not None and cmd.end is not None:
          if cmd.end >= int( self.cap.get( cv2.CAP_PROP_FRAME_COUNT ) ):
            cmd.end = int( self.cap.get( cv2.CAP_PROP_FRAME_COUNT ) ) - 1
          self.range = ( cmd.start, cmd.end )
          self.index = cmd.start
          #self.tracker = create_tracker( "occluboost", reid=None )
          self.tracker = OccluBoost( use_embeddings=False, reid_model=None )
          print( vars( self.tracker ) )
          self.curMode = CommandType.RUN_TRACK
          self.setImagePos( cmd.start )

        elif cmd.type == CommandType.SET_CLIP_ID and cmd.payload is not None:
          self.curClipID = cmd.payload

    except queue.Empty:
      pass

  def processResults( self, results ):
    frame_results: list[ FrameResult ] = []
    for r in results:
      detections = DetectionAdapter( r )
      player_mask = detections.class_id == PLAYER_CLASS_ID
      player_boxes = detections.xyxy[ player_mask ]
      player_confidence = detections.confidence[ player_mask ]
      frame_boxes = [
          BoundingBox(
              int( coords[ 0 ] ),
              int( coords[ 1 ] ),
              int( coords[ 2 ] ),
              int( coords[ 3 ] ),
              float( confidence ),
              PLAYER_CLASS_ID,
              self.index,
          )
          for coords, confidence in zip( player_boxes, player_confidence, strict=True )
      ]
      frame_results.append( FrameResult( frame=self.index, boxes=frame_boxes, mode=CommandType.RUN_BBOX ) )
      self.index += 1
    if frame_results:
      self.out_queue.put( Output( type=OutputType.FRAME_BATCH, data=frame_results ) )
      self.checkCompletion()

  def checkCompletion( self ):
    if self.index > self.range[ 1 ]:
      logger.info( "Processing complete!" )
      self.out_queue.put( Output( type=OutputType.COMPLETED ) )
      self.stop()

  def run( self ):
    tracking_batch: list[ FrameResult ] = []
    try:
      while not self.stopped:
        self.processCommands()
        if self.paused or self.stopped:
          self.flushTrackingBatch( tracking_batch )
        while self.paused and not self.stopped:
          time.sleep( 0.5 )
          self.processCommands()

        if self.stopped:
          self.flushTrackingBatch( tracking_batch )
          break

        if self.curMode == CommandType.RUN_BBOX:
          if self.model is None or self.activeMdlOpts is None:
            raise RuntimeError( "YOLO model configuration is unavailable while detection is active." )
          active_mdl_opts = self.activeMdlOpts
          frames: list[ np.ndarray ] = []
          for _ in range( active_mdl_opts.batchSize ):
            if self.index + len( frames ) > self.range[ 1 ]:
              break
            ret, frame = self.cap.read()
            if not ret:
              logger.error( f"Failed reading cap {ret}" )
              self.stop()
              break
            frames.append( frame )
          if not frames:
            continue
          results = self.model.predict(
              source=frames,
              verbose=False,
              imgsz=active_mdl_opts.imgSz,
              batch=len( frames ),
          )
          self.processResults( results )
        elif self.curMode == CommandType.RUN_TRACK:
          ret, frame = self.cap.read()
          if not ret:
            logger.error( f"Failed reading cap {ret}" )
            self.stop()
            continue
          tracking_batch.append( self.processTracking( frame ) )
          if len( tracking_batch ) >= TRACK_OUTPUT_BATCH_SIZE or self.index > self.range[ 1 ]:
            self.flushTrackingBatch( tracking_batch )
          self.checkCompletion()
    finally:
      self.flushTrackingBatch( tracking_batch )
      self.releaseResources()
      logger.info( "Quitting thread" )

  def processTracking( self, frame ) -> FrameResult:
    if self.tracker is None:
      raise RuntimeError( "Tracker is unavailable while tracking is active." )
    currDets = self.inBoxes.get( self.index, [] )
    if currDets:
      dets = np.asarray( [ d.to_boxmot() for d in currDets ], dtype=np.float32 ).reshape( -1, 6 )
    else:
      dets = np.empty( ( 0, 6 ), dtype=np.float32 )
    tracks = self.tracker.update( dets, frame )
    frame_tracks = []

    # If we're in the tracks list, then we're active to some degree.  But we could still be predicted.
    #print( self.index )
    #print(type(self.tracker))
    #print(vars(self.tracker).keys())
    #
    #if 915 <= self.index <= 965:
    #  ids = [ 11, 15, 16, 24 ]
    #  active = []
    #  atrack = []
    #  for t in self.tracker.trackers:
    #    if hasattr( t, "id" ) and t.id in ids:  # t.xywha[ 0 ] < 1920:
    #      atrack.append( t.id )
    #
    #  #print( self.index, ids )
    #  for t in self.tracker.active_tracks:
    #    if t.id in ids:
    #      active.append( t.id )
    #      #dtls.append( f"id={t.id:3d} streak={t.hit_streak:3d} age={t.age:3d} conf={t.conf:.2f}" )
    #      #print( f"id={t.id:3d} "
    #      #       f"tsu={t.time_since_update:2d} "
    #      #       f"hits={t.hits:3d} "
    #      #       f"streak={t.hit_streak:3d} "
    #      #       f"age={t.age:3d} "
    #      #       f"conf={t.conf:.2f} "
    #      #       f"active={t.is_activated}" )
    #
    #  print( self.index, "Investigating", ids, "All Tracks", atrack, "Active", active )
    #  if 920 <= self.index <= 930:
    #    for t in self.tracker.trackers:
    #      if t.id in ids:
    #        print( self.index, t.id, t.time_since_update, t.kf.x.flatten()[ :4 ] )
    #        if self.index == 925:
    #          print( type( t.emb ) )
    #          print( t.emb is None )
    #        if self.index == 927 and t.id == 11:
    #          #print( self.index, len( t.history_observations ), t.history_observations[ -10: ] )
    #          print( type( t.history_observations ) )
    #          print( t.history_observations )
    #
    #  #t = self.tracker.active_tracks[0]
    #  #print(type(t))
    #  #print(vars(t).keys())
    #  #print(self.tracker.active_tracks)

    for track in tracks:
      x1, y1, x2, y2, track_id, score, cls, _ = track

      #if track.id in (4, 6, 40, 51):
      #  print( vars( track ) )

      bbox = BoundingBox( int( x1 ), int( y1 ), int( x2 ), int( y2 ), float( score ), int( cls ), self.index )
      frame_tracks.append( TrackData( self.curClipID, tid=int( track_id ), data=bbox ) )

    result = FrameResult( frame=self.index, tracks=frame_tracks, mode=CommandType.RUN_TRACK )
    self.index += 1
    return result

  def flushTrackingBatch( self, frame_results: list[ FrameResult ] ) -> None:
    if frame_results:
      self.out_queue.put( Output( type=OutputType.FRAME_BATCH, data=list( frame_results ) ) )
      frame_results.clear()

  def releaseResources( self ) -> None:
    self.cap.release()
    self.model = None
    self.activeMdlOpts = None
    self.tracker = None
    self.inBoxes.clear()
    gc.collect()

    import torch

    if torch.cuda.is_available():
      torch.cuda.empty_cache()
