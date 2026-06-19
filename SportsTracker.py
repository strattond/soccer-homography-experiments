from dataclasses import dataclass, field

import cv2
from ultralytics import YOLO

from appState import AppState
from dataTypes import Homography, SelectionPoint

# This will be responsible for loading the model, performing detections and tracking, and so on

@dataclass
class SportsTracker:
  # yapf: disable
  appState:         AppState                 = field( init=False )
  frameIndex:       int                      = 50
  data:             Homography               = field( default_factory=Homography )
  cap:              cv2.VideoCapture         = field( init=False )
  model:            YOLO | None              = None
