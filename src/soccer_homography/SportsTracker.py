import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import cv2
import numpy as np
from boxmot.trackers.bbox import ByteTrack
from ultralytics import YOLO

from soccer_homography.appState import ModelOptions
from soccer_homography.dataTypes import BoundingBox, Homography, TrackData
from soccer_homography.detectionadapter import DetectionAdapter
from soccer_homography.log import logger

BALL_CLASS_ID = 32
PLAYER_CLASS_ID = 0


class CommandType( Enum ):
  RUN_BBOX = auto()
  FEED_BBOX = auto()
  RUN_TRACK = auto()
  PAUSE = auto()
  RESUME = auto()
  STOP = auto()
  SEEK = auto()


class OutputType( Enum ):
  BBOX = auto()
  TRACK = auto()
  NEW_FRAME = auto()
  COMPLETED = auto()


@dataclass
class Command:
  type: CommandType
  start: int | None = None
  end: int | None = None
  payload: Any = None


@dataclass
class Output:
  type: OutputType
  data: BoundingBox | TrackData | int | None = None


# This will be responsible for loading the model, performing detections and tracking, and so on


@dataclass
class SportsTracker:
  # yapf: disable
  mdlOpts:          ModelOptions
  videoFile:        str
  range:            tuple[int, int]              = field( default_factory=tuple[int, int] )
  index:            int                          = 0
  model:            YOLO                         = field( init=False )
  tracker:          ByteTrack                    = field( init=False )
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
    #logger.info( "Stopping SportsTracker" )
    print( "Stopping SportsTracker" )
    self.paused = False
    self.stopped = True

  def setImagePos( self, pos: int ):
    self.cap.set( cv2.CAP_PROP_POS_FRAMES, pos )

  def processCommands( self ):
    try:
      while True:
        cmd: Command = self.in_queue.get_nowait()

        print( "Received command", cmd )
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
          modelName = "yolo26" + self.mdlOpts.size + "." + self.mdlOpts.engine
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
          self.tracker = ByteTrack()
          self.curMode = CommandType.RUN_TRACK
          self.setImagePos( cmd.start )

    except queue.Empty:
      pass

  def processResults( self, results ):
    # Process results
    for r in results:
      self.out_queue.put( Output( type=OutputType.NEW_FRAME, data=self.index ) )
      detections = DetectionAdapter( r )
      keep_ids = { PLAYER_CLASS_ID, BALL_CLASS_ID }
      all_mask = [ cid in keep_ids for cid in detections.class_id ]
      detections = detections[ all_mask ]
      #balls = detections[ detections.class_id == BALL_CLASS_ID ]
      players = detections[ detections.class_id == PLAYER_CLASS_ID ]
      #ball_dets = np.hstack( ( balls.xyxy, balls.confidence[ :, None ], balls.class_id[ :, None ] ) )
      player_dets = np.hstack( ( players.xyxy, players.confidence[ :, None ], players.class_id[ :, None ], players.trackID[ :, None ] ) )
      for det in player_dets:
        x1f, y1f, x2f, y2f, conf, cidf, tidf = det

        # Team colour classifier
        x1, y1, x2, y2, cid, tid = map( int, ( x1f, y1f, x2f, y2f, cidf, tidf ) )
        logger.debug( f"Player box {x1:4d},{y1:4d} x {x2:4d},{y2:4d} Confidence {conf:8.4f} Class {cid} Track ID {tid}" )
        self.out_queue.put( Output( type=OutputType.BBOX, data=BoundingBox( x1, y1, x2, y2, conf, cid, self.index ) ) )

      self.index += 1
      self.checkCompletion()

  def checkCompletion( self ):
    if self.index > self.range[ 1 ]:
      logger.info( "Processing complete!" )
      self.out_queue.put( Output( type=OutputType.NEW_FRAME, data=self.index - 1 ) )
      self.out_queue.put( Output( type=OutputType.COMPLETED ) )
      self.stop()

  def run( self ):

    # Prebind these to make Python ignore it
    ret = True
    frame = None
    while not self.stopped:
      self.processCommands()
      while self.paused and not self.stopped:
        time.sleep( 0.5 )
        self.processCommands()

      # We might go from paused to stopped
      if self.stopped:
        break

      if self.curMode == CommandType.RUN_BBOX or self.curMode == CommandType.RUN_TRACK:
        ret, frame = self.cap.read()
        if not ret:
          logger.error( f"Failed reading cap {ret}" )
          self.stop()
          continue

      # Do nothing without a valid frame
      if frame is None:
        continue

      if self.curMode == CommandType.RUN_BBOX:
        #  Predicting
        results = self.model.predict( source=[ frame ], verbose=False, imgsz=self.mdlOpts.imgSz )
        self.processResults( results )
      elif self.curMode == CommandType.RUN_TRACK:
        currDets = self.inBoxes.get( self.index, [] )
        if currDets:
          dets = np.array( [ d.to_boxmot() for d in currDets ] )
        else:
          dets = np.empty( ( 0, 6 ) )
        tracks = self.tracker.update( dets, img=frame )
        self.out_queue.put( Output( type=OutputType.NEW_FRAME, data=self.index ) )
        for track in tracks:
          x1, y1, x2, y2, track_id, score, cls, frame_id = track

          bbox = BoundingBox( x1, y1, x2, y2, score, cls, self.index )
          self.out_queue.put( Output( type=OutputType.TRACK, data=TrackData( tid=track_id, data=bbox ) ) )

        self.index += 1
        self.checkCompletion()

    print( "Quitting thread" )
    #if hasattr( self.model, "predictor" ) and self.model.predictol0r is not None and hasattr( self.model.predictor, "trackers" ):
    #  print( "Tracker reset!" )
    #  for tracker in self.model.predictor.trackers:
    #    tracker.reset()
    self.cap.release()
