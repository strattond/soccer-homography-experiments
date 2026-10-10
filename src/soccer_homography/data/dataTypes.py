import json
from dataclasses import asdict, dataclass, field
from typing import Literal

import cv2
import numpy as np
from cv2.typing import MatLike


@dataclass( slots=True )
class Point2D:
  x: float = 0.0
  y: float = 0.0

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
    return np.array( [ self.x, self.y ], dtype=np.float32 )


@dataclass( slots=True )
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
  img_pts_4k: list[ SelectionPoint ] = field( default_factory=list )
  world_pts: list[ SelectionPoint ] = field( default_factory=list )
  display: Point2D = field( default_factory=lambda: Point2D( 1920, 1080 ) )
  source: Point2D = field( default_factory=lambda: Point2D() )
  hom4k: MatLike | None = None

  def setSourceDimensions( self, orig: Point2D ):
    self.source = Point2D( orig.x, orig.y )

  def computeScaledHomography( self, transform: ViewTransform ) -> MatLike:
    if len( self.img_pts_4k ) < 4 or len( self.img_pts_4k ) != len( self.world_pts ):
      raise ValueError( "At least four matching image and world points are required" )
    img_pts_scaled = transform.getScaledPoints( self.img_pts_4k )
    img_pts_arr = np.array( [ ip.coords.to_numpy() for ip in img_pts_scaled ], dtype=np.float32 )
    world_pts_arr = np.array( [ wp.coords.to_numpy() for wp in self.world_pts ], dtype=np.float32 )
    homScaled, _ = cv2.findHomography( img_pts_arr, world_pts_arr, method=cv2.RANSAC )
    return homScaled

  def compute( self ):
    self.hom4k = None
    if len( self.img_pts_4k ) < 4 or len( self.img_pts_4k ) != len( self.world_pts ):
      return
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

    self.load_dict( data )

  def load_dict( self, data: dict ):
    # --- Homography ---
    self.hom4k = np.array( data[ "homography" ], dtype=np.float64 ) if data[ "homography" ] is not None else None

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
  # yapf: enable


ParticipationRole = Literal[ "home_player", "home_goalkeeper", "away_player", "away_goalkeeper", "referee", "unknown" ]

roles: tuple[ ParticipationRole, ...] = (
    "home_player",
    "home_goalkeeper",
    "away_player",
    "away_goalkeeper",
    "referee",
    "unknown",
)



@dataclass( slots=True )
class BoundingBox:
  # yapf: disable
  x1:        int
  y1:        int
  x2:        int
  y2:        int
  conf:      float
  cls:       int
  frame:     int
  # yapf: enable

  def to_dict( self ):
    return asdict( self )

  def to_boxmot( self ):
    return [ self.x1, self.y1, self.x2, self.y2, self.conf, self.cls ]


@dataclass( slots=True )
class TrackData:
  # yapf: disable
  clip:      int
  tid:       int
  data:      BoundingBox
  # yapf: enable


@dataclass( slots=True )
class TrackSegment:
  frame_start: int
  frame_end: int
  person_id: int | None = None


@dataclass( slots=True )
class Track:
  # yapf: disable
  clip:          int
  id:            int
  person:        Person | int | None = None
  boxes:         list[BoundingBox]   = field( default_factory=list )
  role:          ParticipationRole   = "unknown"
  homog:         list[Point2D]       = field( default_factory=list )
  homog_smooth:  list[Point2D]       = field( default_factory=list )
  smooth_pos:    np.ndarray | None   = None
  segments:      list[TrackSegment]  = field( default_factory=list )
  # yapf: enable

  def __post_init__( self ) -> None:
    if not self.segments:
      self.refreshSegments()

  def refreshSegments( self ) -> None:
    if not self.boxes:
      self.segments = []
      return

    frames = sorted( { box.frame for box in self.boxes } )
    segments: list[ TrackSegment ] = []
    start = end = frames[ 0 ]
    for frame in frames[ 1: ]:
      if frame != end + 1:
        segments.append( TrackSegment( start, end ) )
        start = frame
      end = frame
    segments.append( TrackSegment( start, end ) )
    self.segments = segments

  def addBox( self, box: BoundingBox ) -> None:
    self.boxes.append( box )
    if any( segment.frame_start <= box.frame <= segment.frame_end for segment in self.segments ):
      return

    previous = next(
        ( segment for segment in reversed( self.segments ) if segment.frame_end < box.frame ),
        None,
    )
    following = next(
        ( segment for segment in self.segments if segment.frame_start > box.frame ),
        None,
    )
    if previous is not None and previous.frame_end + 1 == box.frame:
      previous.frame_end = box.frame
    elif following is not None and box.frame + 1 == following.frame_start:
      following.frame_start = box.frame
    else:
      self.segments.append( TrackSegment( box.frame, box.frame ) )
      self.segments.sort( key=lambda segment: segment.frame_start )

  def segmentAt( self, frame: int ) -> TrackSegment | None:
    return next(
        ( segment for segment in self.segments if segment.frame_start <= frame <= segment.frame_end ),
        None,
    )

  def roleAt( self, frame: int, roles_by_person: dict[ int, ParticipationRole ] ) -> ParticipationRole:
    segment = self.segmentAt( frame )
    if segment is None or segment.person_id is None:
      return "unknown"
    return roles_by_person.get( segment.person_id, "unknown" )

  def forExport( self, lo: int, hi: int ):
    nBoxes = [ box for box in self.boxes if box.frame >= lo and box.frame < hi ]

    exported = Track( self.clip, self.id, self.person, nBoxes, self.role )
    exported.segments = [
        TrackSegment(
            max( segment.frame_start, lo ),
            min( segment.frame_end, hi - 1 ),
            segment.person_id,
        )
        for segment in self.segments
        if segment.frame_start < hi and segment.frame_end >= lo
    ]
    return exported

  def numId( self ) -> int | None:
    if isinstance( self.person, Person ):
      return self.person.id
    if isinstance( self.person, int ):
      return self.person
    return None

  def to_dict( self ):
    return { "id": self.id, "person": self.numId(), "role": self.role, "boxes": [ [ box.to_dict() for box in self.boxes ] ]}

  def getByIndex( self, index: int ) -> BoundingBox | None:
    return next( ( box for box in self.boxes if box.frame == index ), None )

  def getListIndex( self, index: int ) -> int | None:
    return next( ( i for i, box in enumerate( self.boxes ) if box.frame == index ), None )

  def clearFrame( self, index: int ) -> None:
    for box in self.boxes:
      if box.frame == index:
        self.boxes.remove( box )
        return

  def clearHomography( self ):
    self.homog.clear()
    self.homog_smooth.clear()

  def refreshHomography( self, transformer: Homography ):
    if transformer.hom4k is None:
      return
    boxLen = len( self.boxes )
    hmgLen = len( self.homog )
    positions: list[ list[ float ] ] = []
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
