from dataclasses import dataclass, field

import cv2

from dataTypes import Homography, SelectionPoint
from pitch import SoccerPitchColors, SoccerPitchConfiguration, SoccerPitchImage


@dataclass
class ImageOptions:
  # yapf: disable
  showHough:   bool = True                  # Checkbox - Show Hough layer
  preBlur:     bool = True                  # Checkbox - blur for edge detection
  removeSky:   bool = True                  # Checkbox - try to remove sky
  edgeEnhance: bool = True                  # Checkbox - apply CLAHE enhancement
  closeEdges:  bool = True                  # Checkbox - close edges
  edgeType:    str  = 'Canny'               # Combo box - edge type - Canny, Scharr
  lineType:    str  = 'LineSegmentDetector' # Combo box - line type - Hough, LineSegmentDetector


@dataclass
class ModelOptions:
  # yapf: disable
  withReID: bool = True
  size:     str  = 'm'
  imgSz:    int  = 640


@dataclass
class AppState:
  # yapf: disable
  input:            str                      = "test_homography_input.mp4"
  homogFile:        str                      = "H_image_to_pitch.json"
  frameIndex:       int                      = 50
  last_image_click: SelectionPoint | None    = None
  hover_point:      SelectionPoint | None    = None
  sel_world_point:  SelectionPoint | None    = None
  data:             Homography               = field( default_factory=Homography )
  cfg:              SoccerPitchConfiguration = field( default_factory=SoccerPitchConfiguration )
  colors:           SoccerPitchColors        = field( default_factory=SoccerPitchColors )
  pitch:            SoccerPitchImage         = field( init=False )
  cap:              cv2.VideoCapture | None  = None
  imgOpts:          ImageOptions             = field( default_factory=ImageOptions )
  mdlOpts:          ModelOptions             = field( default_factory=ModelOptions )

  def __post_init__( self ):
    self.pitch = SoccerPitchImage( cfg=self.cfg, colors=self.colors )
