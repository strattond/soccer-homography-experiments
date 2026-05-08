import cv2
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
from cv2.typing import MatLike

@dataclass
class Point2D:
  x: int = 0
  y: int = 0

  def __add__(self, other):
    if not isinstance(other, Point2D):
      return NotImplemented
    return Point2D( self.x + other.x, self.y + other.y )
    
  def __sub__(self, other):
    if not isinstance(other, Point2D):
      return NotImplemented
    return Point2D( self.x - other.x, self.y - other.y )
  
  def __iter__(self):
    yield self.x
    yield self.y

  def to_numpy(self) -> np.ndarray:
    return np.array( [int(self.x), int(self.y)], dtype=np.float32 )
  
@dataclass
class SelectionPoint:
  index: int = None
  coords: Point2D = field( default_factory=lambda: Point2D( None, None ) )

@dataclass
class Homography:
  img_pts_disp:  List[SelectionPoint] = field( default_factory=lambda: [] )
  img_pts_4k:    List[SelectionPoint] = field( default_factory=lambda: [] )
  world_pts:     List[SelectionPoint] = field( default_factory=lambda: [] )
  display:       Point2D              = field( default_factory=lambda: Point2D( 1920, 1080 ) )
  source:        Point2D              = field( default_factory=lambda: Point2D( None, None ) )
  scaleUpX:      int                  = None
  scaleUpY:      int                  = None
  scaleDown:     float                = None
  homDisplay:    MatLike              = None
  hom4k:         MatLike              = None

  def setSourceDimensions( self, orig: Point2D ):
    self.source = Point2D( orig.x, orig.y )
    self.recalcScale()

  def recalcScale( self ):
    self.scaleDown = min( self.display.x / self.source.x, self.display.y / self.source.y )
    self.scaleUpX = self.source.x / self.display.x
    self.scaleUpY = self.source.y / self.display.y


  def scaleUp( self, dimensions: Point2D ) -> Point2D:
    return Point2D( dimensions.x * self.scaleUpX, dimensions.y * self.scaleUpY )

  def storeClickPair( self, image: SelectionPoint, world: SelectionPoint ):
      self.img_pts_disp.append( image )

      # Scale up and store in 4k space
      self.addScaledPair( image )

      self.world_pts.append( world )

      if len(self.img_pts_disp ) >= 4:
        self.compute()

  def addScaledPair( self, image: SelectionPoint ):
    scaled = self.scaleUp( image.coords )
    self.img_pts_4k.append( SelectionPoint( image.index, scaled ) )

  def compute( self ):
    img_pts_4k_arr     = np.array( [ip.coords.to_numpy() for ip in self.img_pts_4k], dtype=np.float32 )
    img_pts_arr        = np.array( [ip.coords.to_numpy() for ip in self.img_pts_disp], dtype=np.float32 )
    world_pts_arr      = np.array( [wp.coords.to_numpy() for wp in self.world_pts],  dtype=np.float32 )
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
        "display": [ self.display.x, self.display.y ],
        "source":  [ self.source.x,  self.source.y  ]
      }
    }

    with open( path, "w") as f:
      json.dump( data, f, indent=2 )

  def load_point(self, d):
    c = d["coords"]
    if isinstance(c, dict):
        return Point2D(c["x"], c["y"])
    else:
        return Point2D(*c)
      
      
  def load( self, path ):
    with open( path, "r" ) as f:
        data = json.load(f)

    # --- Homography ---
    print( "Loading display homography")
    self.homDisplay = np.array( data["homography"]["display"], dtype=np.float64 )
    print( "Loading source  homography")
    self.homMaxRes  = np.array( data["homography"]["maxRes"],  dtype=np.float64 )

    # --- Points ---
    print( "Loading image points")
    self.img_pts_disp = [
      SelectionPoint(
        index=d["index"],
        coords=self.load_point(d)
      )
      for d in data["points"]["image"]
    ]
    print( "Loading world points homography" )
    #self.img_pts_disp = [SelectionPoint(**d) for d in data["points"]["image"]]
    self.world_pts = [
      SelectionPoint(
        index=d["index"],
        coords=self.load_point(d)
      )
      for d in data["points"]["world"]
    ]
    #self.world_pts    = [SelectionPoint(**d) for d in data["points"]["world"]]

    # --- Sizes ---
    print( "Loading display size")
    self.display = Point2D( *data["sizes"]["display"] )
    print( "Loading source size")
    self.source = Point2D( *data["sizes"]["source"] )

    self.recalcScale()

    for p in self.img_pts_disp:
      self.addScaledPair( p )

@dataclass
class VideoData:
  width: int 
  height: int
  fourcc: int
  fps: int   

  def __init__( self, cap: cv2.VideoCapture ):
    self.width  = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    self.height = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    self.fourcc = int( cap.get( cv2.CAP_PROP_FOURCC ) )
    self.fps    = int( cap.get( cv2.CAP_PROP_FPS ) )

@dataclass
class ViewTransform:
  dimensions: Point2D
  scale: float
  offset: Point2D
  
  def __init( self, cap: VideoData ):
    self.dimensions = Point2D( cap.width, cap.height )
    self.scale      = 1.0
    self.offset     = Point2D( 0, 0 )
    
  def toImage( self, x, y ) -> Tuple[float, float]:
    ix = (x - self.offset.x) / self.scale
    iy = (y - self.offset.y) / self.scale
    return (ix, iy)
  
  def toDisplay( self, x, y ) -> Tuple[float, float]:
    ix = x * self.scale + self.offset.x
    iy = y * self.scale + self.offset.y
    return (ix, iy)
  
