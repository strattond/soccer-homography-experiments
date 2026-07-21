from dataclasses import dataclass, field
import cv2
import numpy as np
import supervision as sv

from dataTypes import Point2D, SelectionPoint

# From https://raw.githubusercontent.com/roboflow/sports/refs/heads/main/sports/configs/soccer.py
# But with modifications for size accuracy


# Standard FIFA dimensions https://publications.fifa.com/de/football-stadiums-guidelines/technical-guideline/stadium-guidelines/pitch-dimensions-and-surrounding-areas/
@dataclass
class SoccerPitchConfiguration:
  # yapf: disable
  width:                  float = 68.00  # [cm]
  length:                 float = 105.00  # [cm]
  penalty_box_width:      float = 40.23  # [cm] - From 44 yards (4023.36)
  penalty_box_length:     float = 16.46  # [cm] - From 18 yards (1645.92)
  goal_box_width:         float = 18.29  # [cm] - From 20 yards (8 + 6 + 6) - 1828.8
  goal_box_length:        float = 5.48  # [cm] - From 6 yards - 548.64
  centre_circle_radius:   float = 9.14  # [cm] - From 10 yards - 9.144
  penalty_spot_distance:  float = 10.97  # [cm] - From 12 yards - 10.9728

  @property
  def vertices( self ) -> list[ tuple[ float, float ] ]:
    pbw = self.penalty_box_width
    pbl = self.penalty_box_length
    gbw = self.goal_box_width
    gbl = self.goal_box_length
    ccr = self.centre_circle_radius
    psd = self.penalty_spot_distance
    return [ ( 0,                         0                        ),  # 1
             ( 0,                         ( self.width - pbw ) / 2 ),  # 2
             ( 0,                         ( self.width - gbw ) / 2 ),  # 3
             ( 0,                         ( self.width + gbw ) / 2 ),  # 4
             ( 0,                         ( self.width + pbw ) / 2 ),  # 5
             ( 0,                         self.width               ),  # 6
             ( gbl,                       ( self.width - gbw ) / 2 ),  # 7
             ( gbl,                       ( self.width + gbw ) / 2 ),  # 8
             ( psd,                       self.width / 2           ),  # 9
             ( pbl,                       ( self.width - pbw ) / 2 ),  # 10
             ( pbl,                       ( self.width - gbw ) / 2 ),  # 11
             ( pbl,                       ( self.width + gbw ) / 2 ),  # 12
             ( pbl,                       ( self.width + pbw ) / 2 ),  # 13
             ( self.length / 2,           0                        ),  # 14
             ( self.length / 2,           self.width / 2 - ccr     ),  # 15
             ( self.length / 2,           self.width / 2 + ccr     ),  # 16
             ( self.length / 2,           self.width               ),  # 17
             ( self.length - pbl,         ( self.width - pbw ) / 2 ),  # 18
             ( self.length - pbl,         ( self.width - gbw ) / 2 ),  # 19
             ( self.length - pbl,         ( self.width + gbw ) / 2 ),  # 20
             ( self.length - pbl,         ( self.width + pbw ) / 2 ),  # 21
             ( self.length - psd,         self.width / 2           ),  # 22
             ( self.length - gbl,         ( self.width - gbw ) / 2 ),  # 23
             ( self.length - gbl,         ( self.width + gbw ) / 2 ),  # 24
             ( self.length,               0                        ),  # 25
             ( self.length,               ( self.width - pbw ) / 2 ),  # 26
             ( self.length,               ( self.width - gbw ) / 2 ),  # 27
             ( self.length,               ( self.width + gbw ) / 2 ),  # 28
             ( self.length,               ( self.width + pbw ) / 2 ),  # 29
             ( self.length,               self.width               ),  # 30
             ( self.length / 2 - ccr,     self.width / 2           ),  # 31
             ( self.length / 2 + ccr,     self.width / 2           ),  # 32
             ( psd + ccr,                 self.width / 2           ),  # 33
             ( self.length - (psd + ccr), self.width / 2           ),  # 34
            ]

  edges: list[ tuple[ int, int ] ] = field(
      default_factory=lambda: [ (  1,  2 ), (  2,  3 ), (  3,  4 ), (  4,  5 ), (  5,  6 ), (  7,  8 ),
                                ( 10, 11 ), ( 11, 12 ), ( 12, 13 ), ( 14, 15 ), ( 15, 16 ), ( 16, 17 ),
                                ( 18, 19 ), ( 19, 20 ), ( 20, 21 ), ( 23, 24 ), ( 25, 26 ), ( 26, 27 ),
                                ( 27, 28 ), ( 28, 29 ), ( 29, 30 ), (  1, 14 ), (  2, 10 ), (  3,  7 ),
                                (  4,  8 ), (  5, 13 ), (  6, 17 ), ( 14, 25 ), ( 18, 26 ), ( 23, 27 ),
                                ( 24, 28 ), ( 21, 29 ), ( 17, 30 )
                              ]
  )

  labels: list[ str ] = field(
      default_factory=lambda: [
          "01", "02", "03", "04", "05", "06",
          "07", "08", "09", "10", "11", "12",
          "13", "15", "16", "17", "18", "20",
          "21", "22", "23", "24", "25", "26",
          "27", "28", "29", "30", "31", "32",
          "14", "19", "33", "34"
      ]
  )

  colors: list[ str ] = field(
      default_factory=lambda: [
          "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
          "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
          "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
          "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
          "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
          "#00BFFF", "#00BFFF"
      ]
  )
  # yapf: enable

  @property
  def STD_PITCH_LENGTH( self ) -> float:
    return self.length

  @property
  def STD_PITCH_WIDTH( self ) -> float:
    return self.width


@dataclass
class SoccerPitchColors:
  # yapf: disable
  background_color: sv.Color = sv.Color( 34, 139, 34 )
  line_color:       sv.Color = sv.Color.WHITE
  point_color:      sv.Color = sv.Color.YELLOW
  highlight_color:  sv.Color = sv.Color.GREEN
  hover_color:      sv.Color = sv.Color.RED
  sel_color:        sv.Color = sv.Color.from_hex( "#FF00FF" )
  # yapf: enable


@dataclass
class SoccerPitchImage:
  # Our pitch is on it's side
  # yapf: disable
  width:          int = 420  # [px]
  height:         int = 272  # [px]
  padding:        int =  50  # [px]
  line_thickness: int =   4
  point_radius:   int =   1
  cfg:            SoccerPitchConfiguration = field( default_factory=SoccerPitchConfiguration )
  colors:         SoccerPitchColors        = field( default_factory=SoccerPitchColors )
  empty:          np.ndarray               = field( init = False )  # The empty pitch once constructed
  # yapf: enable

  @property
  def get_pitch_scale( self ):
    return [ ( self.width / self.cfg.STD_PITCH_LENGTH ), ( self.height / self.cfg.STD_PITCH_WIDTH ) ]

  @property
  def get_pitch_centre( self ):
    return [ int( self.width // 2 + self.padding ), int( self.height // 2 + self.padding ) ]

  def makeEmptyPitch( self ) -> np.ndarray:

    scaleW, scaleL = self.get_pitch_scale
    centreX, centreY = self.get_pitch_centre
    yard10 = int( self.cfg.centre_circle_radius * scaleW )
    scaled_penalty = int( self.cfg.penalty_spot_distance * scaleL )

    # Blank out the image
    self.empty = np.ones( ( self.height + 2 * self.padding, self.width + 2 * self.padding, 3 ), dtype=np.uint8 ) * np.array( self.colors.background_color.as_bgr(), dtype=np.uint8 )

    for start, end in self.cfg.edges:
      point1 = ( int( self.cfg.vertices[ start - 1 ][ 0 ] * scaleW ) + self.padding, int( self.cfg.vertices[ start - 1 ][ 1 ] * scaleL ) + self.padding )
      point2 = ( int( self.cfg.vertices[ end - 1 ][ 0 ] * scaleW ) + self.padding, int( self.cfg.vertices[ end - 1 ][ 1 ] * scaleL ) + self.padding )
      cv2.line( img=self.empty, pt1=point1, pt2=point2, color=self.colors.line_color.as_bgr(), thickness=self.line_thickness )

    centre_circle_center = ( centreX, centreY )
    cv2.circle( img=self.empty, center=centre_circle_center, radius=yard10, color=self.colors.line_color.as_bgr(), thickness=self.line_thickness )

    penalty_spots = [ ( int( scaled_penalty + self.padding ), centreY ), ( int( self.width - scaled_penalty + self.padding ), centreY ) ]
    arc_angles = [ ( -57, 57 ), ( 123, 237 ) ]
    for spot, ( start, end ) in zip( penalty_spots, arc_angles ):
      cv2.circle( img=self.empty, center=spot, radius=int( self.point_radius * scaleW ), color=self.colors.line_color.as_bgr(), thickness=-1 )
      cv2.ellipse( img=self.empty, center=spot, axes=( yard10, yard10 ), angle=0, startAngle=start, endAngle=end, color=self.colors.line_color.as_bgr(), thickness=2 )

    return self.empty

  def calcPointOffset( self, target_point: SelectionPoint ) -> Point2D:

    scaleW, scaleL = self.get_pitch_scale
    x = int( target_point.coords.x )
    y = int( target_point.coords.y )
    return Point2D( int( x*scaleW + self.padding ), int( y*scaleL + self.padding ) )

  def nearestFieldPoint( self, mx, my ) -> SelectionPoint:
    best_pt = self.cfg.vertices[ 0 ]
    best_dist = None
    idx = None

    scaleW, scaleL = self.get_pitch_scale
    for i, pt in enumerate( self.cfg.vertices ):
      px = pt[ 0 ] * scaleW + self.padding
      py = pt[ 1 ] * scaleL + self.padding
      d = ( mx - px )**2 + ( my - py )**2
      if i == 0:
        # First point is always best if found
        best_pt = pt
        best_dist = d
        idx = i
      else:
        if d < best_dist:
          best_dist = d
          best_pt = pt
          idx = i

    return SelectionPoint( idx, Point2D( int( best_pt[ 0 ] ), int( best_pt[ 1 ] ) ) )
