from typing import Iterable

from appState import ModelOptions
from dataclasses import dataclass, field
from dataTypes import Homography, RawTrackData
from detectionadapter import DetectionAdapter
from enum import Enum, auto
from log import logger
from ultralytics import YOLO

import cv2
import numpy as np
import queue
import threading
import time

BALL_CLASS_ID = 32
PLAYER_CLASS_ID = 0


class CommandType( Enum ):
  RUN_FRAMES = auto()
  PAUSE = auto()
  RESUME = auto()
  STOP = auto()
  SEEK = auto()


class OutputType( Enum ):
  BBOX = auto()
  NEW_FRAME = auto()
  COMPLETED = auto()


@dataclass
class Command:
  type: CommandType
  start: int | None = None
  end: int | None = None


@dataclass
class Output:
  type: OutputType
  data: RawTrackData | int | None = None


# This will be responsible for loading the model, performing detections and tracking, and so on


@dataclass
class SportsTracker:
  # yapf: disable
  mdlOpts:          ModelOptions
  videoFile:        str
  range:            tuple[int, int]          = field( default_factory=tuple[int, int] )
  index:            int                      = 0
  model:            YOLO                     = field( init=False )
  data:             Homography               = field( init=False )
  cap:              cv2.VideoCapture         = field( init=False )

  # Threading + communication
  in_queue:         queue.Queue              = field(default_factory=queue.Queue)
  out_queue:        queue.Queue              = field(default_factory=queue.Queue)
  paused:           bool                     = True
  stopped:          bool                     = False
  thread:           threading.Thread         = field(init=False)
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
            self.cap.set( cv2.CAP_PROP_POS_FRAMES, cmd.start )

        elif cmd.type == CommandType.RUN_FRAMES:
          if cmd.start is not None and cmd.end is not None:
            if cmd.end > int( self.cap.get( cv2.CAP_PROP_FRAME_COUNT ) ):
              cmd.end = int( self.cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
            self.range = ( cmd.start, cmd.end )
            self.index = cmd.start
            modelName = "yolo26" + self.mdlOpts.size + ".pt"
            if hasattr( self, "model" ):
              del self.model
            self.model = YOLO( modelName, verbose=False )

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
      balls = detections[ detections.class_id == BALL_CLASS_ID ]
      players = detections[ detections.class_id == PLAYER_CLASS_ID ]
      ball_dets = np.hstack( ( balls.xyxy, balls.confidence[ :, None ], balls.class_id[ :, None ] ) )
      player_dets = np.hstack( ( players.xyxy, players.confidence[ :, None ], players.class_id[ :, None ], players.trackID[ :, None ] ) )
      for det in player_dets:
        x1f, y1f, x2f, y2f, conf, cidf, tidf = det

        # Team colour classifier
        x1, y1, x2, y2, cid, tid = map( int, ( x1f, y1f, x2f, y2f, cidf, tidf ) )

        logger.info( f"Player box {x1:4d},{y1:4d} x {x2:4d},{y2:4d} Confidence {conf:8.4f} Class {cid} Track ID {tid}" )
        self.out_queue.put( Output( type=OutputType.BBOX, data=RawTrackData( tid, x1, y1, x2, y2, conf, self.index ) ) )

      self.index += 1
      if self.index > self.range[ 1 ]:
        logger.info( "Processing complete!" )
        self.out_queue.put( Output( type=OutputType.NEW_FRAME, data=self.index - 1 ) )
        self.out_queue.put( Output( type=OutputType.COMPLETED ) )
        self.pause()

  def run( self ):

    while not self.stopped:
      self.processCommands()
      while self.paused and not self.stopped:
        time.sleep( 0.5 )
        self.processCommands()

      # We might go from paused to stopped
      if self.stopped:
        break

      ret, frame = self.cap.read()
      if not ret:
        logger.error( f"Failed reading cap {ret}" )
        self.pause()
        continue

      #  Predicting
      results = self.model.track( source=[ frame ], verbose=False, tracker='track_custom.yaml', persist=True, imgsz=1280 )
      self.processResults( results )

    self.cap.release()
