from dataclasses import dataclass, field

import cv2
from ultralytics import YOLO

from appState import AppState
from dataTypes import Homography, SelectionPoint
@dataclass
class SportsTracker:
  # yapf: disable
  appState:         AppState                 = field( init=False )
  frameIndex:       int                      = 50
  last_image_click: SelectionPoint | None    = None
  hover_point:      SelectionPoint | None    = None
  sel_world_point:  SelectionPoint | None    = None
  data:             Homography               = field( default_factory=Homography )
  cfg:              SoccerPitchConfiguration = field( default_factory=SoccerPitchConfiguration )
  colors:           SoccerPitchColors        = field( default_factory=SoccerPitchColors )
  pitch:            SoccerPitchImage         = field( init=False )
  cap:              cv2.VideoCapture | None  = None
  model:            YOLO | None              = None
  imgOpts:          ImageOptions             = field( default_factory=ImageOptions )

  def __post_init__( self ):
    self.pitch = SoccerPitchImage( cfg=self.cfg, colors=self.colors )

  def __init__( self, imgOpts: ImageOptions ) -> None:
    self.imgOpts = imgOpts

