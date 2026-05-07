import cv2
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
from cv2.typing import MatLike


@dataclass
class SelectionPoint:
  index: int = None
  coords: Tuple[int, int] = field( default_factory=lambda: (None, None) )

@dataclass
class Homography:
  img_pts_disp:  List[SelectionPoint] = field( default_factory=lambda: [] )
  img_pts_4k:    List[SelectionPoint] = field( default_factory=lambda: [] )
  world_pts:     List[SelectionPoint] = field( default_factory=lambda: [] )
  displayWidth:  int                  = 1920
  displayHeight: int                  = 1080
  scaleUpX:      int                  = None
  scaleUpY:      int                  = None
  scaleDown:     float                = None
  sourceH:       int                  = None
  sourceW:       int                  = None
  homDisplay:    MatLike              = None
  hom4k:         MatLike              = None

  @property
  def displayDimensions( self ) -> Tuple[int, int]:
    return ( self.displayWidth, self.displayHeight )

  def setSourceDimensions( self, orig_h, orig_w ):
    self.sourceH = orig_h
    self.sourceW = orig_w
    self.recalcScale()

  def recalcScale( self ):
    self.scaleDown = min( self.displayWidth / self.sourceW, self.displayHeight / self.sourceH )
    self.scaleUpX = self.sourceW / self.displayWidth
    self.scaleUpY = self.sourceH / self.displayHeight


  def scaleUp( self, dimensions: Tuple[int, int] ) -> Tuple[int, int]:
    x, y = dimensions
    return ( x * self.scaleUpX, y * self.scaleUpY )

  def storeClickPair( self, image: SelectionPoint, world: SelectionPoint ):
      self.img_pts_disp.append( image )

      # Scale up and store in 4k space
      self.addScaledPair( image )

      self.world_pts.append( world )

      if len(self.img_pts_disp ) >= 4:
        self.compute()

  def addScaledPair( self, image: SelectionPoint ):
    (x, y) = image.coords
    x4k, y4k = self.scaleUp( (x, y) )
    self.img_pts_4k.append( SelectionPoint( image.index, [x4k, y4k] ) )

  def compute( self ):
    img_pts_4k_arr     = np.array( [ip.coords for ip in self.img_pts_4k], dtype=np.float32 )
    img_pts_arr        = np.array( [ip.coords for ip in self.img_pts_disp], dtype=np.float32 )
    world_pts_arr      = np.array( [wp.coords for wp in self.world_pts],  dtype=np.float32 )
    self.hom4k, _      = cv2.findHomography( img_pts_4k_arr, world_pts_arr, method=cv2.RANSAC )
    self.homDisplay, _ = cv2.findHomography( img_pts_arr, world_pts_arr, method=cv2.RANSAC )

  def save( self, path ):
    data = {
      "homography": {
        "display": self.homDisplay.tolist(),
        "maxRes": self.homDisplay.tolist()
      },
      "points": {
        "image": [asdict(p) for p in self.img_pts_disp],
        "world": [asdict(p) for p in self.world_pts]
      },
      "sizes": {
        "display": [ self.displayWidth, self.displayHeight ],
        "source":  [ self.sourceW,      self.sourceH       ]
      }
    }

    with open( path, "w") as f:
      json.dump( data, f, indent=2 )

  def load( self, path ):
    with open( path, "r" ) as f:
        data = json.load(f)

    # --- Homography ---
    self.homDisplay = np.array( data["homography"]["display"], dtype=np.float64 )
    self.homMaxRes  = np.array( data["homography"]["maxRes"],  dtype=np.float64 )

    # --- Points ---
    self.img_pts_disp = [SelectionPoint(**d) for d in data["points"]["image"]]
    self.world_pts    = [SelectionPoint(**d) for d in data["points"]["world"]]

    # --- Sizes ---
    self.displayWidth, self.displayHeight = data["sizes"]["display"]
    self.sourceW,      self.sourceH       = data["sizes"]["source"]

    self.recalcScale()

    for p in self.img_pts_disp:
      self.addScaledPair( p )

