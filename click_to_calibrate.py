import argparse
import cv2
import json
import numpy as np
import supervision as sv
from dataTypes import SelectionPoint
from pitch import SoccerPitchConfiguration, draw_empty_pitch, overlay_pitch, draw_key_points, draw_sel_points, get_radar_location, get_pitch_scale
from typing import List

parser = argparse.ArgumentParser( description="Click to Calibrate" )
parser.add_argument( "-input",  help="Input video file",      type=str, default="test_homography_input.mp4" )
parser.add_argument( "-homo",   help="Calibrated homography", type=str, default="H_image_to_pitch.4k.npy")
parser.add_argument( "-index",  help="Frame index to use",    type=int, default=50)

args = parser.parse_args()

DISPLAY_W    = 1920
DISPLAY_H    = 1080
VIDEO_PATH   = args.input
MAX_POINTS   = 8
MIN_POINTS   = 4
FRAME_INDEX  = args.index                            # which frame to use for calibration
H_OUTPUT     = args.homo
img_pts_disp: List[SelectionPoint] = []
img_pts_4k:   List[SelectionPoint] = []
world_pts:    List[SelectionPoint] = []
scale_x      = None
scale_y      = None
cfg          = SoccerPitchConfiguration()
radar_bounds = []

last_image_click: SelectionPoint = None
hover_point: SelectionPoint      = None
sel_world_point: SelectionPoint  = None

def loadFrame():
  # --- grab calibration frame ---
  cap = cv2.VideoCapture( VIDEO_PATH )
  if not cap.isOpened():
    raise SystemExit( f"Could not open {VIDEO_PATH}" )

  cap.set( cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX )
  ret, raw_img = cap.read()

  cap.release()
  if not ret:
    raise SystemExit( f"Could not read frame {FRAME_INDEX} from {VIDEO_PATH}" )

  return raw_img

def mouseHandler( event, x, y, flags, param ):
  global last_image_click, sel_world_point, hover_point, img_pts_4k, img_pts_disp

  inside_map = radar_bounds[0] <= x <= radar_bounds[2] and radar_bounds[1] <= y <= radar_bounds[3]
  if event == cv2.EVENT_MOUSEMOVE:
    if inside_map:
      mx = x - radar_bounds[0]
      my = y - radar_bounds[1]
      hover_point = nearest_field_point( mx, my )
    else:
      hover_point = None

  if event == cv2.EVENT_LBUTTONDOWN:
    if len(img_pts_disp) >= MAX_POINTS:
      print("Already collected max points.")
      return

    if inside_map and last_image_click:
      mx = x - radar_bounds[0]
      my = y - radar_bounds[1]
      print( f"Map Click Offset   {mx},{my}" )
      sel_world_point = nearest_field_point( mx, my )
      print( f"Map Click Selected {x},{y} - {sel_world_point.coords} @ {sel_world_point.index}" )
    else:
      print( f"Click {x},{y}" )
      last_image_click = SelectionPoint( None, (x, y) )

def nearest_field_point( mx, my, padding = 50 ) -> SelectionPoint:
  best_pt   = None
  best_dist = None
  idx       = None

  scaleW, scaleL = get_pitch_scale( cfg )
  for i, pt in enumerate(cfg.vertices):
    px = pt[0] * scaleW + padding
    py = pt[1] * scaleL + padding
    d  = (mx - px)**2 + (my - py)**2
    if i == 0:
      # First point is always best if found
      best_pt   = pt
      best_dist = d
      idx       = i
    else:
      if d < best_dist:
        best_dist = d
        best_pt   = pt
        idx       = i

  return SelectionPoint( idx, best_pt )

def main():
  global last_image_click, radar_bounds, sel_world_point, img_pts_disp, img_pts_4k, world_pts
  img = loadFrame()
  orig_h, orig_w = img.shape[:2]

  # Compute display scaling
  scale = min( DISPLAY_W / orig_w, DISPLAY_H / orig_h )
  disp_w = int( orig_w * scale )
  disp_h = int( orig_h * scale )

  # Resize for display
  frame_disp = cv2.resize( img, (disp_w, disp_h), interpolation=cv2.INTER_AREA )

  # Compute scale factors back to 4K
  scale_x = orig_w / disp_w
  scale_y = orig_h / disp_h

  print( f"Original: {orig_w}x{orig_h}" )
  print( f"Displayed: {disp_w}x{disp_h}" )
  print( f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}" )

  cv2.namedWindow( "Click to Calibrate", cv2.WINDOW_NORMAL )
  cv2.setMouseCallback( "Click to Calibrate", mouseHandler, param=frame_disp )

  print( "Click field markings (sideline, circle arc, etc.)." )
  print( "Press ESC when done." )

  blankField     = draw_empty_pitch(cfg)
  x0, y0, ph, pw = get_radar_location( frame_disp, blankField )
  radar_bounds   = [x0, y0, x0 + pw, y0 + ph]

  while True:

    # Prepare and display what we know ...
    active = frame_disp.copy()
    radar = blankField.copy()
    # Overlay selected points
    overlay_pitch( active, radar, alpha=0.25 )
    draw_key_points( active, cfg, x0, y0 )
    draw_sel_points( active, img_pts_disp )
    if last_image_click != None:
      draw_sel_points( active, [last_image_click], point_color=sv.Color.BLUE )

    # Draw our map and selected points
    cv2.imshow( "Click to Calibrate", active )
    key = cv2.waitKey(1)
    if key == 27:  # ESC
      break

    # When both clicked, store it
    if last_image_click and sel_world_point:

      # Store in image space
      last_image_click.index = sel_world_point.index
      img_pts_disp.append( last_image_click )

      # Scale up and store in 4k space
      #idx, (x, y) = last_image_click
      (x, y) = last_image_click.coords
      x4k = x * scale_x
      y4k = y * scale_y
      img_pts_4k.append( SelectionPoint( last_image_click.index, [x4k, y4k] ) )

      world_pts.append( sel_world_point )

      # Ensure reset for next go around
      last_image_click = None
      sel_world_point  = None

  cv2.destroyAllWindows()

  if len( img_pts_disp ) < MIN_POINTS:
    raise SystemExit( f"Need at least {MIN_POINTS} points for homography." )

  print( "\nClick space points:" )
  for p in img_pts_disp:
      print(f"{p}")

  print( "\nFinal 4K-space points:" )
  for p in img_pts_4k:
      print(f"{p}")

  # Optionally save them
  #np.savetxt( "calibration_points_4k.txt", img_pts_4k_arr, fmt="%.2f" )
  with open( "calibration_points_4k.json", "w") as f:
    json.dump( [p.__dict__ for p in img_pts_4k], f, indent=2 )
  print( "\nSaved to calibration_points_4k.txt" )

  img_pts_4k_arr    = np.array( [ip.coords for ip in img_pts_4k], dtype=np.float32 )
  world_pts_arr     = np.array( [wp.coords for wp in world_pts],  dtype=np.float32 )
  
  H, mask = cv2.findHomography( img_pts_4k_arr, world_pts_arr, method=cv2.RANSAC )
  np.save( H_OUTPUT, H )

  print( f"\nSaved homography to {H_OUTPUT}" )
  print( "H =\n", H )

if __name__ == "__main__":
  main()