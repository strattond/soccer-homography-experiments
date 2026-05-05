from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np
import supervision as sv

PITCH_HEIGHT = 272
PITCH_WIDTH = 420

# From https://raw.githubusercontent.com/roboflow/sports/refs/heads/main/sports/configs/soccer.py
# But with modifications for size accuracy

# Standard FIFA dimensions https://publications.fifa.com/de/football-stadiums-guidelines/technical-guideline/stadium-guidelines/pitch-dimensions-and-surrounding-areas/
@dataclass
class SoccerPitchConfiguration:
  width: int = 6800  # [cm]
  length: int = 10500  # [cm]
  penalty_box_width: int = 4023  # [cm] - From 44 yards (4023.36)
  penalty_box_length: int = 1646  # [cm] - From 18 yards (1645.92)
  goal_box_width: int = 1829  # [cm] - From 20 yards (8 + 6 + 6) - 1828.8
  goal_box_length: int = 548  # [cm] - From 6 yards - 548.64
  centre_circle_radius: int = 914  # [cm] - From 10 yards - 9.144
  penalty_spot_distance: int = 1097  # [cm] - From 12 yards - 10.9728

  @property
  def vertices(self) -> List[Tuple[int, int]]:
    return [
      (0, 0),  # 1
      (0, (self.width - self.penalty_box_width) / 2),  # 2
      (0, (self.width - self.goal_box_width) / 2),  # 3
      (0, (self.width + self.goal_box_width) / 2),  # 4
      (0, (self.width + self.penalty_box_width) / 2),  # 5
      (0, self.width),  # 6
      (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
      (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
      (self.penalty_spot_distance, self.width / 2),  # 9
      (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
      (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
      (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
      (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
      (self.length / 2, 0),  # 14
      (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
      (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
      (self.length / 2, self.width),  # 17
      (
          self.length - self.penalty_box_length,
          (self.width - self.penalty_box_width) / 2
      ),  # 18
      (
          self.length - self.penalty_box_length,
          (self.width - self.goal_box_width) / 2
      ),  # 19
      (
          self.length - self.penalty_box_length,
          (self.width + self.goal_box_width) / 2
      ),  # 20
      (
          self.length - self.penalty_box_length,
          (self.width + self.penalty_box_width) / 2
      ),  # 21
      (self.length - self.penalty_spot_distance, self.width / 2),  # 22
      (
          self.length - self.goal_box_length,
          (self.width - self.goal_box_width) / 2
      ),  # 23
      (
          self.length - self.goal_box_length,
          (self.width + self.goal_box_width) / 2
      ),  # 24
      (self.length, 0),  # 25
      (self.length, (self.width - self.penalty_box_width) / 2),  # 26
      (self.length, (self.width - self.goal_box_width) / 2),  # 27
      (self.length, (self.width + self.goal_box_width) / 2),  # 28
      (self.length, (self.width + self.penalty_box_width) / 2),  # 29
      (self.length, self.width),  # 30
      (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
      (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
    ]

  edges: List[Tuple[int, int]] = field(default_factory=lambda: [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
    (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
    (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
  ])

  labels: List[str] = field(default_factory=lambda: [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
    "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
    "14", "19"
  ])

  colors: List[str] = field(default_factory=lambda: [
    "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
    "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
    "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
    "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
    "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
    "#00BFFF", "#00BFFF"
  ])

  @property
  def STD_PITCH_LENGTH(self) -> int: return int(self.length / 100)
  @property
  def STD_PITCH_WIDTH(self) -> int: return int(self.width / 100)

def get_pitch_scale( cfg: SoccerPitchConfiguration,
                     width = PITCH_WIDTH,
                     height = PITCH_HEIGHT ):
  return [ (width / cfg.STD_PITCH_LENGTH)/100, (height / cfg.STD_PITCH_WIDTH)/100 ]

def draw_empty_pitch( cfg: SoccerPitchConfiguration,
                      width                         = PITCH_WIDTH,
                      height                        = PITCH_HEIGHT,
                      background_color: sv.Color    = sv.Color(34, 139, 34),
                      line_color: sv.Color          = sv.Color.WHITE,
                      padding: int                  = 50,
                      line_thickness: int           = 4,
                      point_radius: int             = 8

) -> np.ndarray:

  scaleW, scaleL = get_pitch_scale( cfg, width, height )
  yard10         = int(cfg.centre_circle_radius * scaleW)
  centreX        = int(width // 2 + padding)
  centreY        = int(height // 2 + padding)
  scaled_penalty = int(cfg.penalty_spot_distance * scaleL)

  # Blank out the image
  pitch = np.ones( (height + 2 * padding, width + 2 * padding, 3), dtype=np.uint8 ) * np.array( background_color.as_bgr(), dtype=np.uint8 )

  for start, end in cfg.edges:
    point1 = (int(cfg.vertices[start - 1][0] * scaleW) + padding, int(cfg.vertices[start - 1][1] * scaleL) + padding)
    point2 = (int(cfg.vertices[end   - 1][0] * scaleW) + padding, int(cfg.vertices[end   - 1][1] * scaleL) + padding)
    cv2.line( img=pitch, pt1=point1, pt2=point2, color=line_color.as_bgr(), thickness=line_thickness )

  centre_circle_center = ( centreX, centreY )
  cv2.circle( img=pitch, center=centre_circle_center, radius=yard10, color=line_color.as_bgr(), thickness=line_thickness )

  penalty_spots = [
      ( int(scaled_penalty + padding),         centreY ),
      ( int(width - scaled_penalty + padding), centreY )
  ]
  arc_angles = [ (-57, 57), (123, 237) ]
  for spot, (start,end) in zip(penalty_spots, arc_angles):
    cv2.circle( img=pitch, center=spot, radius=int(point_radius * scaleW * 10), color=line_color.as_bgr(), thickness=-1 )
    cv2.ellipse( img=pitch, center=spot, axes=(yard10, yard10), angle=0, startAngle=start, endAngle=end, color=line_color.as_bgr(), thickness=2 )

  return pitch

def get_radar_location( frame, pitch_img, padding = 100):
  fh, fw = frame.shape[:2]
  ph, pw = pitch_img.shape[:2]

  # bottom-middle placement
  x0 = fw//2 - pw//2
  y0 = fh - ph - padding

  return [x0, y0, ph, pw]

def overlay_pitch( frame, pitch_img, padding = 100 ):

  x0, y0, ph, pw = get_radar_location( frame, pitch_img, padding )
  # Copy existing part so we can alpha-blend
  roi     = frame[y0:y0 + ph, x0:x0 + pw]
  alpha   = 0.7
  blended = cv2.addWeighted( pitch_img, alpha, roi, 1 - alpha, 0 )
  frame[y0:y0 + ph, x0:x0 + pw] = blended

def draw_key_points(  existing: np.ndarray,
                      cfg: SoccerPitchConfiguration,
                      offsetX,
                      offsetY,
                      width                         = PITCH_WIDTH,
                      height                        = PITCH_HEIGHT,
                      point_color: sv.Color         = sv.Color.YELLOW,
                      highlight_color: sv.Color     = sv.Color.GREEN,
                      padding: int                  = 50,
                      highlight_label = None
                    ):

  scaleW, scaleL = get_pitch_scale( cfg, width, height )
  i = 0
  for vertex, pt in zip( cfg.vertices, cfg.labels ):
    mx = int(vertex[0] * scaleW + padding + offsetX)
    my = int(vertex[1] * scaleL + padding + offsetY)

    color = point_color if pt != highlight_label else highlight_color
    radius = 5 if pt != highlight_label else 8
    cv2.circle( existing, (mx, my), radius, color.as_bgr(), -1 )
    cv2.putText( existing, pt, (mx + 5, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color.as_bgr(), 1 )
    i += 1
    #print( f"Map coordinate {pt} {mx},{my} => {mx - padding - offsetX},{my - padding - offsetY}" )

  return existing

def draw_sel_points(  existing: np.ndarray,
                      cfg: SoccerPitchConfiguration,
                      offsetX,
                      offsetY,
                      img_points,
                      width                         = PITCH_WIDTH,
                      height                        = PITCH_HEIGHT,
                      point_color: sv.Color         = sv.Color.RED,
                      highlight_color: sv.Color     = sv.Color.GREEN,
                      padding: int                  = 50
                    ):

  for i, (x, y) in enumerate(img_points):

    radius = 5
    cv2.circle( existing, (x, y), radius, point_color.as_bgr(), -1 )
    cv2.putText( existing, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color.as_bgr(), 1 )

  return existing