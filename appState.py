from dataclasses import dataclass, field

import cv2

from dataTypes import Homography, SelectionPoint, ViewTransform
from pitch import SoccerPitchColors, SoccerPitchConfiguration, SoccerPitchImage


@dataclass
class AppState:
  input: str = "test_homography_input.mp4"
  homogFile: str = "H_image_to_pitch.json"
  frameIndex: int = 50
  radar_bounds = []
  last_image_click: SelectionPoint | None = None
  hover_point: SelectionPoint | None = None
  sel_world_point: SelectionPoint | None = None
  view: ViewTransform | None = None
  data: Homography | None = None
  cfg: SoccerPitchConfiguration = field( default_factory=SoccerPitchConfiguration )
  colors: SoccerPitchColors = field( default_factory=SoccerPitchColors )
  pitch: SoccerPitchImage = field( init=False )
  cap: cv2.VideoCapture | None = None

  def __post_init__( self ):
    self.pitch = SoccerPitchImage( cfg=self.cfg, colors=self.colors )
