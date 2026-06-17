import cv2
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from cv2.typing import MatLike


@dataclass
class Point2D:
  x: int = 0
  y: int = 0

  def __add__( self, other ):
    if not isinstance( other, Point2D ):
      return NotImplemented
    return Point2D( self.x + other.x, self.y + other.y )

  def __sub__( self, other ):
    if not isinstance( other, Point2D ):
      return NotImplemented
    return Point2D( self.x - other.x, self.y - other.y )

  def __iter__( self ):
    yield self.x
    yield self.y

  def to_numpy( self ) -> np.ndarray:
    return np.array( [ int( self.x ), int( self.y ) ], dtype=np.float32 )


@dataclass
class SelectionPoint:
  index: int | None = None
  coords: Point2D = field( default_factory=lambda: Point2D() )


@dataclass
class VideoData:
  width: int
  height: int
  fourcc: int
  fps: int
  frames: int

  def __init__( self, cap: cv2.VideoCapture ):
    self.width = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    self.height = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    self.fourcc = int( cap.get( cv2.CAP_PROP_FOURCC ) )
    self.fps = int( cap.get( cv2.CAP_PROP_FPS ) )
    self.frames = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )


@dataclass
class ViewTransform:
  dimensions: Point2D = field( default_factory=lambda: Point2D() )
  scale: float = 1.0
  offset: Point2D = field( default_factory=lambda: Point2D() )

  def __init__( self, cap: VideoData | None = None ):
    if cap is not None:
      self.dimensions = Point2D( cap.width, cap.height )
    else:
      self.dimensions = Point2D()
    self.scale = 1.0
    self.offset = Point2D()

  def toImage( self, x, y ) -> tuple[ float, float ]:
    ix = ( x - self.offset.x ) / self.scale
    iy = ( y - self.offset.y ) / self.scale
    return ( ix, iy )

  def toDisplay( self, x, y ) -> tuple[ float, float ]:
    ix = x * self.scale + self.offset.x
    iy = y * self.scale + self.offset.y
    return ( ix, iy )

  def scaledDimensions( self ) -> tuple[ int, int ]:
    iwdth = int( self.dimensions.x * self.scale )
    ihght = int( self.dimensions.y * self.scale )
    return ( iwdth, ihght )

  def getScaledPoints( self, points: list[ SelectionPoint ] ) -> list[ SelectionPoint ]:
    img_pts_scaled: list[ SelectionPoint ] = []
    for ip in points:
      img_pts_scaled.append( SelectionPoint( ip.index, Point2D( int(ip.coords.x * self.scale), int(ip.coords.y * self.scale) ) ) )
    return img_pts_scaled


@dataclass
class Homography:
  img_pts_4k: list[ SelectionPoint ] = field( default_factory=lambda: [] )
  world_pts: list[ SelectionPoint ] = field( default_factory=lambda: [] )
  display: Point2D = field( default_factory=lambda: Point2D( 1920, 1080 ) )
  source: Point2D = field( default_factory=lambda: Point2D() )
  scaleUpX: float = 0
  scaleUpY: float = 0
  scaleDown: float = 0
  hom4k: MatLike | None = None

  def setSourceDimensions( self, orig: Point2D ):
    self.source = Point2D( orig.x, orig.y )
    self.recalcScale()

  def recalcScale( self ):
    self.scaleDown = min( self.display.x / self.source.x, self.display.y / self.source.y )
    self.scaleUpX = self.source.x / self.display.x
    self.scaleUpY = self.source.y / self.display.y

  def scaleUp( self, dimensions: Point2D ) -> Point2D:
    print( f"Scaling {dimensions.x},{dimensions.y} to {dimensions.x * self.scaleUpX}, {dimensions.y * self.scaleUpY}" )
    return Point2D( int(dimensions.x * self.scaleUpX), int(dimensions.y * self.scaleUpY) )

  def computeScaledHomography( self, transform: ViewTransform ) -> MatLike:
    img_pts_scaled = transform.getScaledPoints( self.img_pts_4k )
    img_pts_arr = np.array( [ ip.coords.to_numpy() for ip in img_pts_scaled ], dtype=np.float32 )
    world_pts_arr = np.array( [ wp.coords.to_numpy() for wp in self.world_pts ], dtype=np.float32 )
    homScaled, _ = cv2.findHomography( img_pts_arr, world_pts_arr, method=cv2.RANSAC )
    return homScaled

  def storeClickPair( self, image: SelectionPoint, world: SelectionPoint ):
    # Scale up and store in 4k space
    self.addScaledPair( image )

    self.world_pts.append( world )

    if len( self.img_pts_4k ) >= 4:
      self.compute()

  def addScaledPair( self, image: SelectionPoint ):
    scaled = self.scaleUp( image.coords )
    self.img_pts_4k.append( SelectionPoint( image.index, scaled ) )

  def compute( self ):
    img_pts_4k_arr = np.array( [ ip.coords.to_numpy() for ip in self.img_pts_4k ], dtype=np.float32 )
    world_pts_arr = np.array( [ wp.coords.to_numpy() for wp in self.world_pts ], dtype=np.float32 )
    self.hom4k, _ = cv2.findHomography( img_pts_4k_arr, world_pts_arr, method=cv2.RANSAC )

  def save( self, path ):
    data = {
        "homography": self.hom4k.tolist() if self.hom4k else None,
        "points": {
            "image": [ asdict( p ) for p in self.img_pts_4k ],
            "world": [ asdict( p ) for p in self.world_pts ]
        },
        "sizes": {
            "display": [ self.display.x, self.display.y ],
            "source": [ self.source.x, self.source.y ]
        }
    }

    with open( path, "w" ) as f:
      json.dump( data, f, indent=2 )

  def load_point( self, d ):
    c = d[ "coords" ]
    if isinstance( c, dict ):
      return Point2D( c[ "x" ], c[ "y" ] )
    else:
      return Point2D( *c )

  def load( self, path ):
    with open( path, "r" ) as f:
      data = json.load( f )

    # --- Homography ---
    print( "Loading source  homography" )
    self.hom4k = np.array( data[ "homography" ], dtype=np.float64 )

    # --- Points ---
    print( "Loading image points" )
    self.img_pts_4k = [ SelectionPoint( index=d[ "index" ], coords=self.load_point( d ) ) for d in data[ "points" ][ "image" ] ]
    print( "Loading world points homography" )
    self.world_pts = [ SelectionPoint( index=d[ "index" ], coords=self.load_point( d ) ) for d in data[ "points" ][ "world" ] ]

    # --- Sizes ---
    print( "Loading display size" )
    self.display = Point2D( *data[ "sizes" ][ "display" ] )
    print( "Loading source size" )
    self.source = Point2D( *data[ "sizes" ][ "source" ] )

    self.recalcScale()
