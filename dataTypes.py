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
      img_pts_scaled.append( SelectionPoint( ip.index, Point2D( int( ip.coords.x * self.scale ), int( ip.coords.y * self.scale ) ) ) )
    return img_pts_scaled


@dataclass
class Homography:
  img_pts_4k: list[ SelectionPoint ] = field( default_factory=lambda: [] )
  world_pts: list[ SelectionPoint ] = field( default_factory=lambda: [] )
  display: Point2D = field( default_factory=lambda: Point2D( 1920, 1080 ) )
  source: Point2D = field( default_factory=lambda: Point2D() )
  hom4k: MatLike | None = None

  def setSourceDimensions( self, orig: Point2D ):
    self.source = Point2D( orig.x, orig.y )

  def computeScaledHomography( self, transform: ViewTransform ) -> MatLike:
    img_pts_scaled = transform.getScaledPoints( self.img_pts_4k )
    img_pts_arr = np.array( [ ip.coords.to_numpy() for ip in img_pts_scaled ], dtype=np.float32 )
    world_pts_arr = np.array( [ wp.coords.to_numpy() for wp in self.world_pts ], dtype=np.float32 )
    homScaled, _ = cv2.findHomography( img_pts_arr, world_pts_arr, method=cv2.RANSAC )
    return homScaled

  def compute( self ):
    img_pts_4k_arr = np.array( [ ip.coords.to_numpy() for ip in self.img_pts_4k ], dtype=np.float32 )
    world_pts_arr = np.array( [ wp.coords.to_numpy() for wp in self.world_pts ], dtype=np.float32 )
    self.hom4k, _ = cv2.findHomography( img_pts_4k_arr, world_pts_arr, method=cv2.RANSAC )

  def to_dict( self ):
    return {
        "homography": self.hom4k.tolist() if self.hom4k is not None else None,
        "points": {
            "image": [ asdict( p ) for p in self.img_pts_4k ],
            "world": [ asdict( p ) for p in self.world_pts ]
        },
        "sizes": {
            "display": [ self.display.x, self.display.y ],
            "source": [ self.source.x, self.source.y ]
        }
    }

  def to_json( self ) -> str:
    return json.dumps( self.to_dict() )

  def save( self, path ):
    data = self.to_dict()

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
    self.hom4k = np.array( data[ "homography" ], dtype=np.float64 )

    # --- Points ---
    self.img_pts_4k = [ SelectionPoint( index=d[ "index" ], coords=self.load_point( d ) ) for d in data[ "points" ][ "image" ] ]
    self.world_pts = [ SelectionPoint( index=d[ "index" ], coords=self.load_point( d ) ) for d in data[ "points" ][ "world" ] ]

    # --- Sizes ---
    self.display = Point2D( *data[ "sizes" ][ "display" ] )
    self.source = Point2D( *data[ "sizes" ][ "source" ] )

  def shazam( self, positions: list[ list[ float ] ] ) -> MatLike | None:
    # Reshape it for our perspective transform
    if self.hom4k is not None:
      pts = np.array( positions, dtype=np.float32 ).reshape( -1, 1, 2 )
      return cv2.perspectiveTransform( pts, self.hom4k ).reshape( -1, 2 )


@dataclass
class Person:
  # yapf: disable
  id:     int       = 0
  name:   str       = ""
  pType:  int       = 0     # 0 - home, 1 - away, 2 - official
  tracks: list[int] = field( default_factory=list )
  # yapf: enable


@dataclass
class TrackData:
  # yapf: disable
  x1:    int
  y1:    int
  x2:    int
  y2:    int
  conf:  float
  index: int
  # yapf: enable

  def to_dict( self ):
    return asdict( self )


@dataclass
class RawTrackData:
  # yapf: disable
  tid:   int
  data:  TrackData
  # yapf: enable

  def __init__( self, tid: int, x1: int, y1: int, x2: int, y2: int, conf: float, index: int ) -> None:
    self.tid = tid
    self.data = TrackData( x1, y1, x2, y2, conf, index )


@dataclass
class Track:
  # yapf: disable
  id:            int
  person:        Person | int | None = None
  boxes:         list[TrackData]     = field( default_factory=list )
  homog:         list[Point2D]       = field( default_factory=list )
  homog_smooth:  list[Point2D]       = field( default_factory=list )
  smooth_pos:    np.ndarray | None   = None
  # yapf: enable

  def forExport( self, lo: int, hi: int ):
    nBoxes: list[TrackData] = []
    for box in self.boxes:
      if box.index >= lo and box.index < hi:
        nBoxes.append( box )

    return Track( self.id, self.person, nBoxes )

  def numId( self ) -> int | None:
    actId = None
    if isinstance( self.id, Person ):
      actId = self.id.id
    elif isinstance( self.id, int ):
      actId = self.id
    return actId          

  def to_dict( self ):
    return { "id": self.id, "person": self.numId(), "boxes": [ [ box.to_dict() for box in self.boxes ] ]}

  def getByIndex( self, index: int ) -> TrackData | None:
    for box in self.boxes:
      if box.index == index:
        return box
    return None

  def getListIndex( self, index: int ) -> int | None:
    for ( i, box ) in enumerate( self.boxes ):
      if box.index == index:
        return i
    return None

  def clearFrame( self, index: int ) -> None:
    for box in self.boxes:
      if box.index == index:
        self.boxes.remove( box )
        return

  def clearHomography( self ):
    self.homog.clear()
    self.homog_smooth.clear()

  def refreshHomography( self, transformer: Homography ):
    boxLen = len( self.boxes )
    hmgLen = len( self.homog )
    positions: list[ list[ float ] ] = []
    print( f"Refreshing from {hmgLen} to {boxLen - 1} homography" )
    for i in range( hmgLen, boxLen ):
      b = self.boxes[ i ]
      # Get the middle bottom of the bounding box aka Da Feet
      positions.append( [ 0.5 * ( b.x1 + b.x2 ), b.y2 ] )
    # Cool, now we have positions ...
    if len( positions ) == 0:
      return
    # Something to homography
    xf = transformer.shazam( positions )
    if xf is not None:
      for ( x, y ) in xf:
        # Raw positions
        self.homog.append( Point2D( x, y ) )
        # Smooth it
        xs, ys = self.smooth( x, y )

        self.homog_smooth.append( Point2D( xs, ys ) )

    if len( self.boxes ) != len( self.homog ):
      print( f"Track {self.id} - {len(self.boxes)} vs {len(self.homog)}" )

  def smooth( self, x, y ):
    alpha = 0.2
    dead_zone = 0.15

    curr = np.array( [ x, y ], dtype=float )

    if self.smooth_pos is None:
      self.smooth_pos = curr
      return Point2D( curr[ 0 ], curr[ 1 ] )

    prev = self.smooth_pos
    delta = curr - prev

    if np.linalg.norm( delta ) < dead_zone:
      return prev

    new = alpha*curr + ( 1-alpha ) * prev
    self.smooth_pos = new
    return Point2D( new[ 0 ], new[ 1 ] )
