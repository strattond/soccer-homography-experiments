from dataclasses import dataclass, field
from tkinter import Canvas
from typing import List, Tuple
import cv2
import numpy as np
import supervision as sv

from dataTypes import Point2D, SelectionPoint

# From https://raw.githubusercontent.com/roboflow/sports/refs/heads/main/sports/configs/soccer.py
# But with modifications for size accuracy


# Standard FIFA dimensions https://publications.fifa.com/de/football-stadiums-guidelines/technical-guideline/stadium-guidelines/pitch-dimensions-and-surrounding-areas/
@dataclass
class SoccerPitchConfiguration:
  width: int = 68.00  # [cm]
  length: int = 105.00  # [cm]
  penalty_box_width: int = 40.23  # [cm] - From 44 yards (4023.36)
  penalty_box_length: int = 16.46  # [cm] - From 18 yards (1645.92)
  goal_box_width: int = 18.29  # [cm] - From 20 yards (8 + 6 + 6) - 1828.8
  goal_box_length: int = 5.48  # [cm] - From 6 yards - 548.64
  centre_circle_radius: int = 9.14  # [cm] - From 10 yards - 9.144
  penalty_spot_distance: int = 10.97  # [cm] - From 12 yards - 10.9728

  @property
  def vertices( self ) -> List[ Tuple[ int, int ] ]:
    return [ ( 0, 0 ),  # 1
             ( 0, ( self.width - self.penalty_box_width ) / 2 ),  # 2
             ( 0, ( self.width - self.goal_box_width ) / 2 ),  # 3
             ( 0, ( self.width + self.goal_box_width ) / 2 ),  # 4
             ( 0, ( self.width + self.penalty_box_width ) / 2 ),  # 5
             ( 0, self.width ),  # 6
             ( self.goal_box_length, ( self.width - self.goal_box_width ) / 2 ),  # 7
             ( self.goal_box_length, ( self.width + self.goal_box_width ) / 2 ),  # 8
             ( self.penalty_spot_distance, self.width / 2 ),  # 9
             ( self.penalty_box_length, ( self.width - self.penalty_box_width ) / 2 ),  # 10
             ( self.penalty_box_length, ( self.width - self.goal_box_width ) / 2 ),  # 11
             ( self.penalty_box_length, ( self.width + self.goal_box_width ) / 2 ),  # 12
             ( self.penalty_box_length, ( self.width + self.penalty_box_width ) / 2 ),  # 13
             ( self.length / 2, 0 ),  # 14
             ( self.length / 2, self.width / 2 - self.centre_circle_radius ),  # 15
             ( self.length / 2, self.width / 2 + self.centre_circle_radius ),  # 16
             ( self.length / 2, self.width ),  # 17
             ( self.length - self.penalty_box_length, ( self.width - self.penalty_box_width ) / 2 ),  # 18
             ( self.length - self.penalty_box_length, ( self.width - self.goal_box_width ) / 2 ),  # 19
             ( self.length - self.penalty_box_length, ( self.width + self.goal_box_width ) / 2 ),  # 20
             ( self.length - self.penalty_box_length, ( self.width + self.penalty_box_width ) / 2 ),  # 21
             ( self.length - self.penalty_spot_distance, self.width / 2 ),  # 22
             ( self.length - self.goal_box_length, ( self.width - self.goal_box_width ) / 2 ),  # 23
             ( self.length - self.goal_box_length, ( self.width + self.goal_box_width ) / 2 ),  # 24
             ( self.length, 0 ),  # 25
             ( self.length, ( self.width - self.penalty_box_width ) / 2 ),  # 26
             ( self.length, ( self.width - self.goal_box_width ) / 2 ),  # 27
             ( self.length, ( self.width + self.goal_box_width ) / 2 ),  # 28
             ( self.length, ( self.width + self.penalty_box_width ) / 2 ),  # 29
             ( self.length, self.width ),  # 30
             ( self.length / 2 - self.centre_circle_radius, self.width / 2 ),  # 31
             ( self.length / 2 + self.centre_circle_radius, self.width / 2 ),  # 32
            ]

  edges: List[ Tuple[ int, int ] ] = field(
      default_factory=lambda: [ ( 1, 2 ), ( 2, 3 ), ( 3, 4 ), ( 4, 5 ), ( 5, 6 ), ( 7, 8 ), ( 10, 11 ), ( 11, 12 ), ( 12, 13 ),
                                ( 14, 15 ), ( 15, 16 ), ( 16, 17 ), ( 18, 19 ), ( 19, 20 ), ( 20, 21 ), ( 23, 24 ), ( 25, 26 ),
                                ( 26, 27 ), ( 27, 28 ), ( 28, 29 ), ( 29, 30 ), ( 1, 14 ), ( 2, 10 ), ( 3, 7 ), ( 4, 8 ),
                                ( 5, 13 ), ( 6, 17 ), ( 14, 25 ), ( 18, 26 ), ( 23, 27 ), ( 24, 28 ), ( 21, 29 ), ( 17, 30 ) ]
  )

  labels: List[ str ] = field(
      default_factory=lambda: [
          "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "15", "16", "17", "18", "20", "21",
          "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "14", "19"
      ]
  )

  colors: List[ str ] = field(
      default_factory=lambda: [
          "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
          "#FF1493", "#FF1493", "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347", "#FF6347", "#FF6347",
          "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
          "#00BFFF", "#00BFFF"
      ]
  )

  @property
  def STD_PITCH_LENGTH( self ) -> float:
    return self.length

  @property
  def STD_PITCH_WIDTH( self ) -> float:
    return self.width


@dataclass
class SoccerPitchColors:
  background_color: sv.Color = sv.Color( 34, 139, 34 )
  line_color: sv.Color = sv.Color.WHITE
  point_color: sv.Color = sv.Color.YELLOW
  highlight_color: sv.Color = sv.Color.GREEN
  hover_color: sv.Color = sv.Color.RED
  sel_color: sv.Color = sv.Color.from_hex( "#FF00FF" )


@dataclass
class SoccerPitchImage:
  # Our pitch is on it's side
  width: int = 420  # [px]
  height: int = 272  # [px]
  padding: int = 50  # [px]
  line_thickness: int = 4
  point_radius: int = 1
  cfg: SoccerPitchConfiguration = field( default_factory=SoccerPitchConfiguration )
  colors: SoccerPitchColors = field( default_factory=SoccerPitchColors )
  empty: np.ndarray = None  # The empty pitch once constructed

  @property
  def get_pitch_scale( self ):
    return [ ( self.width / self.cfg.STD_PITCH_LENGTH ), ( self.height / self.cfg.STD_PITCH_WIDTH ) ]

  @property
  def get_pitch_centre( self ):
    return [ int( self.width // 2 + self.padding ), int( self.height // 2 + self.padding ) ]

  def draw_empty_pitch( self ) -> np.ndarray:

    scaleW, scaleL = self.get_pitch_scale
    centreX, centreY = self.get_pitch_centre
    yard10 = int( self.cfg.centre_circle_radius * scaleW )
    scaled_penalty = int( self.cfg.penalty_spot_distance * scaleL )

    # Blank out the image
    self.empty = np.ones( ( self.height + 2 * self.padding, self.width + 2 * self.padding, 3 ), dtype=np.uint8 ) * np.array(
        self.colors.background_color.as_bgr(), dtype=np.uint8
    )

    for start, end in self.cfg.edges:
      point1 = (
          int( self.cfg.vertices[ start - 1 ][ 0 ] * scaleW ) + self.padding,
          int( self.cfg.vertices[ start - 1 ][ 1 ] * scaleL ) + self.padding
      )
      point2 = (
          int( self.cfg.vertices[ end - 1 ][ 0 ] * scaleW ) + self.padding,
          int( self.cfg.vertices[ end - 1 ][ 1 ] * scaleL ) + self.padding
      )
      cv2.line( img=self.empty, pt1=point1, pt2=point2, color=self.colors.line_color.as_bgr(), thickness=self.line_thickness )

    centre_circle_center = ( centreX, centreY )
    cv2.circle(
        img=self.empty,
        center=centre_circle_center,
        radius=yard10,
        color=self.colors.line_color.as_bgr(),
        thickness=self.line_thickness
    )

    penalty_spots = [ ( int( scaled_penalty + self.padding ), centreY ),
                      ( int( self.width - scaled_penalty + self.padding ), centreY ) ]
    arc_angles = [ ( -57, 57 ), ( 123, 237 ) ]
    for spot, ( start, end ) in zip( penalty_spots, arc_angles ):
      cv2.circle(
          img=self.empty,
          center=spot,
          radius=int( self.point_radius * scaleW ),
          color=self.colors.line_color.as_bgr(),
          thickness=-1
      )
      cv2.ellipse(
          img=self.empty,
          center=spot,
          axes=( yard10, yard10 ),
          angle=0,
          startAngle=start,
          endAngle=end,
          color=self.colors.line_color.as_bgr(),
          thickness=2
      )

    return self.empty

  def get_radar_location( self, frame ):
    fh, fw = frame.shape[ :2 ]
    ph, pw = self.empty.shape[ :2 ]

    # bottom-middle placement
    x0 = fw//2 - pw//2
    y0 = fh - ph - ( self.padding * 2 )

    return [ x0, y0, ph, pw ]

  def overlay_pitch( self, frame, pitch_img, alpha=0.7 ):

    x0, y0, ph, pw = self.get_radar_location( frame )
    # Copy existing part so we can alpha-blend
    roi = frame[ y0:y0 + ph, x0:x0 + pw ]
    blended = cv2.addWeighted( pitch_img, alpha, roi, 1 - alpha, 0 )
    frame[ y0:y0 + ph, x0:x0 + pw ] = blended

  def sel_key_color( self, sel_points: List[ SelectionPoint ], pt: str, highlight_label ) -> sv.Color:
    if pt == highlight_label:
      return self.colors.highlight_color
    elif any( sel.index == self.cfg.labels.index( pt ) for sel in sel_points ):
      return self.colors.sel_color
    else:
      return self.colors.point_color

  def draw_key_points(
      self,
      existing: np.ndarray,
      offsetX,
      offsetY,
      sel_points: List[ SelectionPoint ],  # Not using Location here
      highlight_label=None
  ):

    scaleW, scaleL = self.get_pitch_scale
    i = 0
    for vertex, pt in zip( self.cfg.vertices, self.cfg.labels ):
      mx = int( vertex[ 0 ] * scaleW + self.padding + offsetX )
      my = int( vertex[ 1 ] * scaleL + self.padding + offsetY )

      color = self.sel_key_color( sel_points, pt, highlight_label )
      radius = 5 if pt != highlight_label else 8
      cv2.circle( existing, ( mx, my ), radius, color.as_bgr(), -1 )
      cv2.putText( existing, pt, ( mx + 5, my - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.point_color.as_bgr(), 1 )
      i += 1

    return existing

  def draw_key_points( self, existing: Canvas ):

    scaleW, scaleL = self.get_pitch_scale
    i = 0
    for vertex, pt in zip( self.cfg.vertices, self.cfg.labels ):
      mx = int( vertex[ 0 ] * scaleW + self.padding )
      my = int( vertex[ 1 ] * scaleL + self.padding )

      radius = 6
      existing.create_oval(
          mx - radius,
          my - radius,
          mx + radius,
          my + radius,
          fill=self.colors.point_color.as_hex(),
          outline="black",
          width=1,
          tags=( "keypoints" )
      )
      existing.create_text( mx + 10, my - 10, text=pt, fill="white", font=( "Arial", 12 ) )
      i += 1

    return existing

  def draw_sel_points(
      self, existing: np.ndarray, img_points: List[ SelectionPoint ], point_color=sv.Color.YELLOW, scale: float = 1.0
  ):

    for pt in img_points:
      radius = 5
      i = pt.index
      x = int( pt.coords.x * scale )
      y = int( pt.coords.y * scale )
      label = self.cfg.labels[ i ] if i != None else "Next"
      cv2.circle( existing, ( x, y ), radius, point_color.as_bgr(), -1 )
      cv2.putText( existing, label, ( x + 5, y - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color.as_bgr(), 1 )

    return existing

  def draw_hover_point( self, existing: np.ndarray, hover_point: SelectionPoint ):

    x0, y0, ph, pw = self.get_radar_location( existing )
    scaleW, scaleL = self.get_pitch_scale
    radius = 5
    i = hover_point.index
    x = int( hover_point.coords.x )
    y = int( hover_point.coords.y )
    targX = int( x*scaleW + x0 + self.padding )
    targY = int( y*scaleL + y0 + self.padding )
    label = self.cfg.labels[ i ] if i != None else "Next"
    cv2.circle( existing, ( targX, targY ), radius, self.colors.hover_color.as_bgr(), -1 )
    cv2.putText( existing, label, ( targX + 5, targY - 5 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.hover_color.as_bgr(), 1 )

    return existing

  def calc_point_offset( self, hover_point: SelectionPoint ) -> Point2D:

    scaleW, scaleL = self.get_pitch_scale
    x = int( hover_point.coords.x )
    y = int( hover_point.coords.y )
    return Point2D( int( x*scaleW + self.padding ), int( y*scaleL + self.padding ) )

  def nearest_field_point( self, mx, my ) -> SelectionPoint:
    best_pt = None
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

    return SelectionPoint( idx, Point2D( best_pt[ 0 ], best_pt[ 1 ] ) )
