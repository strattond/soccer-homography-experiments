import json
from dataclasses import asdict, dataclass, field

import cv2

from soccer_homography.dataTypes import Homography, Person, SelectionPoint, Track
from soccer_homography.pitch import SoccerPitchColors, SoccerPitchConfiguration, SoccerPitchImage


@dataclass
class ImageOptions:
  # yapf: disable
  showHough:   bool = False                 # Checkbox - Show Hough layer
  preBlur:     bool = True                  # Checkbox - blur for edge detection
  removeSky:   bool = True                  # Checkbox - try to remove sky
  edgeEnhance: bool = True                  # Checkbox - apply CLAHE enhancement
  closeEdges:  bool = True                  # Checkbox - close edges
  edgeType:    str  = 'Canny'               # Combo box - edge type - Canny, Scharr
  lineType:    str  = 'LineSegmentDetector' # Combo box - line type - Hough, LineSegmentDetector

  def to_dict(self):
    return asdict( self )
  def to_json(self) -> str:
    return json.dumps( self.to_dict() )


@dataclass
class ModelOptions:
  # yapf: disable
  withReID: bool = True
  size:     str  = 'x'
  imgSz:    int  = 1280
  engine:   str  = 'engine' # or 'pt'

  def to_dict(self):
    return asdict( self )
  def to_json(self) -> str:
    return json.dumps( self.to_dict() )


@dataclass
class AppState:
  # yapf: disable
  last_image_click: SelectionPoint | None    = None
  sel_world_point:  SelectionPoint | None    = None
  data:             Homography               = field( default_factory=Homography )
  cfg:              SoccerPitchConfiguration = field( default_factory=SoccerPitchConfiguration )
  colors:           SoccerPitchColors        = field( default_factory=SoccerPitchColors )
  pitch:            SoccerPitchImage         = field( init=False )
  cap:              cv2.VideoCapture | None  = None
  videoFile:        str                      = ""
  imgOpts:          ImageOptions             = field( default_factory=ImageOptions )
  mdlOpts:          ModelOptions             = field( default_factory=ModelOptions )
  tracks:           dict[int, Track]         = field( default_factory=dict )
  people:           list[Person]             = field( default_factory=list )
  framesProcessed:  int                      = 0
  chunk:            int                      = 0

  def __post_init__( self ):
    self.pitch = SoccerPitchImage( cfg=self.cfg, colors=self.colors )

  def save( self, path: str ):
    data = {
        "homography": self.data.to_dict(),
        "videoFile": self.videoFile,
        "imgOpts": self.imgOpts.to_dict(),
        "mdlOpts": self.mdlOpts.to_dict(),
        "tracks": { str(k): v.to_dict() for k, v in self.tracks.items() },
        "people": [ asdict( p ) for p in self.people ]
    }

    with open( path, "w" ) as f:
      json.dump( data, f, indent=2 )
