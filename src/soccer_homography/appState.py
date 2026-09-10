import json
from dataclasses import asdict, dataclass, field

import cv2
import duckdb

from soccer_homography.dataTypes import BoundingBox, Homography, SelectionPoint, Track
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
  imgSz:    tuple[int, int]  = (1280, 1280)
  engine:   str  = 'engine' # or 'pt'

  def to_dict(self):
    return asdict( self )
  def to_json(self) -> str:
    return json.dumps( self.to_dict() )


@dataclass
class AppState:
  # yapf: disable
  # Point tracking
  last_image_click: SelectionPoint | None            = None
  sel_world_point:  SelectionPoint | None            = None
  # Current homography
  data:             Homography                       = field( default_factory=Homography )
  # Soccer pitch controls
  cfg:              SoccerPitchConfiguration         = field( default_factory=SoccerPitchConfiguration )
  colors:           SoccerPitchColors                = field( default_factory=SoccerPitchColors )
  pitch:            SoccerPitchImage                 = field( init=False )

  # Video data
  cap:              cv2.VideoCapture | None          = None
  videoFile:        str                              = ""

  # Model/Image options
  imgOpts:          ImageOptions                     = field( default_factory=ImageOptions )
  mdlOpts:          ModelOptions                     = field( default_factory=ModelOptions )

  # Tracking/Box data
  tracks:           dict[int, Track]                 = field( default_factory=dict )
  boxes:            dict[int, list[BoundingBox]]     = field( default_factory=dict )
  framesProcessed:  int                              = 0
  detectChunk:      int                              = 0
  trackChunk:       int                              = 0

  # Current info being processed
  curClipID:        int                              = -1
  db:               duckdb.DuckDBPyConnection | None = None

  def __post_init__( self ):
    self.pitch = SoccerPitchImage( cfg=self.cfg, colors=self.colors )

  def save( self, path: str ):
    data = {
        "homography": self.data.to_dict(),
        "videoFile": self.videoFile,
        "imgOpts": self.imgOpts.to_dict(),
        "mdlOpts": self.mdlOpts.to_dict(),
        "tracks": { str(k): v.to_dict() for k, v in self.tracks.items() },
    }

    with open( path, "w" ) as f:
      json.dump( data, f, indent=2 )
