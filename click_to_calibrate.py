import argparse
import cv2
import json
import numpy as np
import supervision as sv
from dataTypes import Homography, SelectionPoint
from pitch import SoccerPitchConfiguration, SoccerPitchColors, SoccerPitchImage
from typing import List

parser = argparse.ArgumentParser( description="Click to Calibrate" )
parser.add_argument( "-input",  help="Input video file",      type=str, default="test_homography_input.mp4" )
parser.add_argument( "-homo",   help="Calibrated homography", type=str, default="H_image_to_pitch.json")
parser.add_argument( "-index",  help="Frame index to use",    type=int, default=50)

args = parser.parse_args()

VIDEO_PATH   = args.input
MAX_POINTS   = 8
MIN_POINTS   = 4
FRAME_INDEX  = args.index                            # which frame to use for calibration
H_OUTPUT     = args.homo
data         = Homography()
cfg          = SoccerPitchConfiguration()
colors       = SoccerPitchColors()
pitch        = SoccerPitchImage( cfg=cfg, colors=colors )
radar_bounds = []

last_image_click: SelectionPoint = None
hover_point: SelectionPoint      = None
sel_world_point: SelectionPoint  = None

data.load( "H_image_to_pitch.json" )

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
  global last_image_click, sel_world_point, hover_point, pitch

  inside_map = radar_bounds[0] <= x <= radar_bounds[2] and radar_bounds[1] <= y <= radar_bounds[3]
  if event == cv2.EVENT_MOUSEMOVE:
    if inside_map:
      mx = x - radar_bounds[0]
      my = y - radar_bounds[1]
      #print( f"Map Hover Offset   {mx},{my}" )
      hover_point = pitch.nearest_field_point( mx, my )
      #print( f"Map Hover Selected {x},{y} - {hover_point.coords} @ {hover_point.index}" )
    else:
      hover_point = None

  if event == cv2.EVENT_LBUTTONDOWN:
    if len(data.img_pts_disp) >= MAX_POINTS:
      print("Already collected max points.")
      return

    if inside_map and last_image_click:
      mx = x - radar_bounds[0]
      my = y - radar_bounds[1]
      #print( f"Map Click Offset   {mx},{my}" )
      sel_world_point = pitch.nearest_field_point( mx, my )
      #print( f"Map Click Selected {x},{y} - {sel_world_point.coords} @ {sel_world_point.index}" )
    else:
      #print( f"Click {x},{y}" )
      last_image_click = SelectionPoint( None, (x, y) )

def main():
  global last_image_click, radar_bounds, sel_world_point, data
  img = loadFrame()
  orig_h, orig_w = img.shape[:2]

  # Compute display scaling
  data.setSourceDimensions( orig_h, orig_w )

  # Resize for display
  frame_disp = cv2.resize( img, data.displayDimensions, interpolation=cv2.INTER_AREA )

  print( f"Original: {orig_w}x{orig_h}" )
  print( f"Displayed: {data.displayWidth}x{data.displayHeight}" )
  print( f"Scale factors: x={data.scaleUpX:.4f}, y={data.scaleUpY:.4f}" )

  cv2.namedWindow( "Click to Calibrate", cv2.WINDOW_NORMAL )
  cv2.setMouseCallback( "Click to Calibrate", mouseHandler, param=frame_disp )

  print( "Click field markings (sideline, circle arc, etc.)." )
  print( "Press ESC when done." )

  blankField     = pitch.draw_empty_pitch()
  x0, y0, ph, pw = pitch.get_radar_location( frame_disp )
  radar_bounds   = [x0, y0, x0 + pw, y0 + ph]

  while True:

    # Prepare and display what we know ...
    active = frame_disp.copy()
    radar = blankField.copy()
    # Overlay selected points
    pitch.overlay_pitch( active, radar, alpha=0.25 )
    pitch.draw_key_points( active, x0, y0, data.img_pts_disp )
    pitch.draw_sel_points( active, data.img_pts_disp )
    if last_image_click != None:
      pitch.draw_sel_points( active, [last_image_click], point_color=sv.Color.BLUE )
    if hover_point != None:
      pitch.draw_hover_point( active, hover_point )

    # Draw our map and selected points
    cv2.imshow( "Click to Calibrate", active )
    key = cv2.waitKey(1)
    if key == 27:  # ESC
      break

    # When both clicked, store it
    if last_image_click and sel_world_point:

      # Store in image space
      last_image_click.index = sel_world_point.index
      data.storeClickPair( last_image_click, sel_world_point )

      # Ensure reset for next go around
      last_image_click = None
      sel_world_point  = None

  cv2.destroyAllWindows()

  if len( data.img_pts_disp ) < MIN_POINTS:
    raise SystemExit( f"Need at least {MIN_POINTS} points for homography." )

  print( "\nClick space points:" )
  for p in data.img_pts_disp:
      print(f"{p}")

  print( "\nFinal 4K-space points:" )
  for p in data.img_pts_4k:
      print(f"{p}")

  data.save( H_OUTPUT )

if __name__ == "__main__":
  main()