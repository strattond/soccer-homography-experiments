from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from appState import ModelOptions
from dataTypes import Homography
from detectionadapter import DetectionAdapter

BALL_CLASS_ID = 32
PLAYER_CLASS_ID = 0

# This will be responsible for loading the model, performing detections and tracking, and so on


@dataclass
class Person:
  # yapf: disable
  id:    int = 0
  name:  str = ""
  pType: int = 0     # 0 - home, 1 - away, 2 - official
  # yapf: enable


@dataclass
class TrackData:
  # yapf: disable
  x1:    int
  y1:    int
  x2:    int
  y2:    int
  conf:  float
  index: int
  # yapf: enable

  def __init__( self, x1: int, y1: int, x2: int, y2: int, conf: float, index: int ) -> None:
    self.x1 = x1
    self.x2 = x2
    self.y1 = y1
    self.y2 = y2
    self.conf = conf
    self.index = index


@dataclass
class Track:
  # yapf: disable
  id:     int
  person: Person | None   = None
  boxes:  list[TrackData] = field( default_factory=list )
  # yapf: enable

  def getByIndex( self, index: int ) -> TrackData | None:
    for box in self.boxes:
      if box.index == index:
        return box

    return None


@dataclass
class SportsTracker:
  # yapf: disable
  mdlOpts:          ModelOptions
  model:            YOLO                     = field( init=False )
  data:             Homography               = field( init=False )
  cap:              cv2.VideoCapture         = field( init=False )
  tracks:           dict[int, Track]         = field( default_factory=dict )
  # yapf: enable

  def __post_init__( self ) -> None:
    modelName = "yolo26" + self.mdlOpts.size + ".pt"
    self.model = YOLO( modelName, verbose=True )

  def track( self, image, index ):
    #new_frame = cv2.resize( image, (new_w, new_h))

    new_frame = image
    #  Predicting
    results = self.model.track( new_frame, verbose=False, tracker='track_custom.yaml' )
    # Process results
    detections = DetectionAdapter( results )
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

      print( f"Player box {x1:4d},{y1:4d} x {x2:4d},{y2:4d} Confidence {conf:8.4f} Class {cid} Track ID {tid}" )
      if tid not in self.tracks:
        self.tracks[ tid ] = Track( tid )
      self.tracks[ tid ].boxes.append( TrackData( x1, y1, x2, y2, conf, index ) )
