from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import TRACKTRACK

from appState import ModelOptions
from dataTypes import Homography
from detectionadapter import DetectionAdapter

BALL_CLASS_ID = 32
PLAYER_CLASS_ID = 0

# This will be responsible for loading the model, performing detections and tracking, and so on


@dataclass
class Person:
  # yapf: disable
  id: int = 0
  name: str = ""
  pType: int = 0     # 0 - home, 1 - away, 2 - official

@dataclass
class Track:
  # yapf: disable
  id:     int = 0
  person: Person | None = None
  boxes = []

@dataclass
class SportsTracker:
  # yapf: disable
  mdlOpts:          ModelOptions             = field()
  model:            YOLO                     = field()
  tracker:          TRACKTRACK               = field()
  data:             Homography               = field( init=False )
  cap:              cv2.VideoCapture         = field( init=False )
  tracks                                     = {}

  # yapf: enable
  def __init__( self, mdlOpts: ModelOptions ) -> None:
    self.mdlOpts = mdlOpts
    modelName = "yolo26" + mdlOpts.size + ".pt"
    self.model = YOLO( modelName, verbose=False )
    self.tracker = TRACKTRACK( args={ 'with_reid': mdlOpts.withReID, 'reid_model': 'auto'} )

  def track( self, image, index ):
    #new_frame = cv2.resize( image, (new_w, new_h))

    new_frame = image
    #  Predicting
    results = self.model.track( new_frame, verbose=False )
    # Process results
    detections = DetectionAdapter( results )
    keep_ids = { PLAYER_CLASS_ID, BALL_CLASS_ID }
    all_mask = [ cid in keep_ids for cid in detections.class_id ]
    detections = detections[ all_mask ]
    balls = detections[ detections.class_id == BALL_CLASS_ID ]
    players = detections[ detections.class_id == PLAYER_CLASS_ID ]
    ball_dets = np.hstack( ( balls.xyxy, balls.confidence[ :, None ], balls.class_id[ :, None ] ) )
    player_dets = np.hstack( ( players.xyxy, players.confidence[ :, None ], players.class_id[ :, None ] ) )
    positions = []
    teams     = []
    for det in player_dets:
      x1f, y1f, x2f, y2f, conf, cid = det

      # Team colour classifier
      x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))
