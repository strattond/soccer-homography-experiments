from dataclasses import dataclass

import cv2

from dataTypes import Homography, SelectionPoint, ViewTransform
from pitch import SoccerPitchColors, SoccerPitchConfiguration, SoccerPitchImage


@dataclass
class AppState:
  input: str = "test_homography_input.mp4"
  homogFile: str = "H_image_to_pitch.json"
  frameIndex: int = 50
  radar_bounds = []
  last_image_click: SelectionPoint = None
  hover_point: SelectionPoint = None
  sel_world_point: SelectionPoint = None
  view: ViewTransform = None
  data: Homography = None
  cfg: SoccerPitchConfiguration = None
  colors: SoccerPitchColors = None
  pitch: SoccerPitchImage = None
  cap: cv2.VideoCapture = None

  def __init__( self ):
    self.cfg = SoccerPitchConfiguration()
    self.colors = SoccerPitchColors()
    self.pitch = SoccerPitchImage( cfg=self.cfg, colors=self.colors )
