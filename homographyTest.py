import argparse
import cv2
import numpy as np
import supervision as sv
from dataTypes import Homography
from detectionadapter import DetectionAdapter
from images import get_person_mask
from pitch import SoccerPitchConfiguration, SoccerPitchColors, SoccerPitchImage
from ultralytics import YOLO, SAM

parser = argparse.ArgumentParser( description="Homography test" )
parser.add_argument( "-input",   help="Input video file",      type=str, default="test_homography_input.mp4" )
parser.add_argument( "-output",  help="Output video file",     type=str, default="test_homography_output.mp4")
parser.add_argument( "-model",   help="YOLO model",            type=str, default=r"runs\detect\train16\weights\best.pt")
parser.add_argument( "-homo",    help="Calibrated homography", type=str, default="H_image_to_pitch.json")
parser.add_argument( "-segm",    help="Segmentation model",    type=str, default="models/sam2.1_l.pt")
parser.add_argument( "-limit",   help="Frame limit",           type=int, default=5)
parser.add_argument( "-padding", help="Padding around pitch",  type=int, default=50)

args    = parser.parse_args()
cfg     = SoccerPitchConfiguration()
colors  = SoccerPitchColors()
pitch   = SoccerPitchImage( cfg=cfg, colors=colors )
data    = Homography()

print( "Configuring paths" )
input_video_path = args.input
output_video_path = args.output

# Load a Model for detection an prepare it
print( "Loading model" )
model = YOLO( args.model, verbose=False )
#segm  = SAM( args.segm )

# Get Video Information
print( "Getting video information" )
cap    = cv2.VideoCapture( input_video_path )
width  = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
height = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
fourcc = int( cap.get( cv2.CAP_PROP_FOURCC ) )
fps    = int( cap.get( cv2.CAP_PROP_FPS ) )
new_w  = int( width * 0.5 )
new_h  = int( height * 0.5 )

# Progress resources
frame_limit: int = args.limit
frame_wrap:  int = 25
frame_count: int = 0
padding:     int = args.padding

BALL_CLASS_ID   = 32
PLAYER_CLASS_ID = 0
PLAYER_RADIUS   = 10

print( "Configuring Writer" )
out = cv2.VideoWriter( output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h) )

# Helper functions
def positions_to_pitch( positions, H ):
  pts = np.array(positions, dtype=np.float32).reshape( -1, 1, 2 )
  return cv2.perspectiveTransform(pts, H).reshape( -1, 2 )

def classify_team( hsv_patch ):
  h = hsv_patch[:,:,0].mean()
  s = hsv_patch[:,:,1].mean()
  v = hsv_patch[:,:,2].mean()

  # Referee: yellow
  if 20 < h < 35 and s > 80 and v > 80:
    return "referee"

  # Black team
  if v < 60:
    return "away"

  # Red/white stripes (high saturation red OR alternating red/white)
  if (h < 10 or h > 170) and s > 80:
    return "home"

  return "unknown"

def pitch_to_overlay( X, Y, overlay_w, overlay_h, padding ):
  # Default FIFA size is 105 x 68
  # X: 0–105, Y: 0–68
  px = int( (X / cfg.STD_PITCH_LENGTH) * overlay_w ) + padding
  py = int( (Y / cfg.STD_PITCH_WIDTH) * overlay_h ) + padding
  return px, py

def draw_player_on_pitch( pitch_img, X, Y, team, padding ):
  h, w = pitch_img.shape[:2]
  usedH = h - 2 * padding
  usedW = w - 2 * padding
  px, py = pitch_to_overlay( X, Y, usedW, usedH, padding )

  #print( f"{X:.2f},{Y:.2f} => {px:.2f},{py:.2f} (inside {usedH}x{usedW})" )

  color = {
      "away": (255,0,255),
      "home": (0,0,255),
      "referee": (0,255,255),
      "unknown": (200,200,200),
      "ball": (255,255,255)
  }[team]

  cv2.circle( pitch_img, (px, py), PLAYER_RADIUS, color, -1 )

def extract_jersey( frame, x1, y1, x2, y2, person_mask ):
  person = cv2.bitwise_and( frame, frame, mask=person_mask )
  roi = person[y1:y2, x1:x2]
  h, w = roi.shape[:2]

  # Torso band relative to bottom of bbox
  y_bot = int(0.65 * h)
  y_top = int(0.15 * h)

  x_left  = 0
  x_right = w

  torso_img  = roi[y_top:y_bot, x_left:x_right]

  return torso_img

data.load( args.homo )
print( f"Using homography {data.homDisplay}" )
print( "x range:", data.homDisplay[:,0].min(), data.homDisplay[:,0].max() )
print( "y range:", data.homDisplay[:,1].min(), data.homDisplay[:,1].max() )

pitch_base = pitch.draw_empty_pitch()

print( "Looping" )
while True:
  ret, frame = cap.read()
  if not ret:
    break
  frame_count += 1
  if frame_count > frame_limit and frame_limit != -1:
    break

  print('.', end='', flush=True)
  new_frame = cv2.resize( frame, (new_w, new_h))

  if frame_count % frame_wrap == 0:
    print( '', flush=True )

  #  Predicting
  results         = model.predict( new_frame, verbose=False )
  # Process results
  detections      = DetectionAdapter( results )
  keep_ids        = {PLAYER_CLASS_ID, BALL_CLASS_ID}
  all_mask        = [cid in keep_ids for cid in detections.class_id]
  detections      = detections[all_mask]
  balls           = detections[detections.class_id == BALL_CLASS_ID]
  players         = detections[detections.class_id == PLAYER_CLASS_ID]
  ball_dets       = np.hstack( (   balls.xyxy,   balls.confidence[:, None],   balls.class_id[:, None] ) )
  player_dets     = np.hstack( ( players.xyxy, players.confidence[:, None], players.class_id[:, None] ) )

  pitch_img = pitch_base.copy()
  positions = []
  teams     = []
  for det in player_dets:
    x1f, y1f, x2f, y2f, conf, cid = det

    # Team colour classifier
    x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))

    #person_mask = get_person_mask( new_frame, segm, x1, y1, x2, y2 )
    #if person_mask is None:
    #  continue
    #
    #jersey_patch = extract_jersey( new_frame, x1, y1, x2, y2, person_mask )
    #if jersey_patch is None:
    #  continue
    #
    #hsv = cv2.cvtColor( jersey_patch, cv2.COLOR_BGR2HSV )
    #team = classify_team( hsv )
    #if team == "referee":
    #  continue
    team = "home"

    # Homography mapping
    positions.append( [0.5 * (x1 + x2), y2] )
    teams.append( team )

  mapped = positions_to_pitch( positions, data.homDisplay )
  #for m, p in zip( mapped, positions ):
  #  print( f"XFormed {p} => {m}" )

  for (X, Y), team in zip(mapped, teams):
    draw_player_on_pitch( pitch_img, X, Y, team, padding )
  # Draw on mini-pitch
  # draw_player_on_pitch( pitch_img, X, Y, team )

  # Make sure we print the (hopefully just 1) ball

  #for det in ball_dets:
  #  x1, y1, x2, y2, conf, cid = det
  #
  #  # Homography mapping
  #  X, Y = bbox_bottom_center_to_pitch( x1, y1, x2, y2, H )
#
  #  # Draw on mini-pitch
  #  draw_player_on_pitch( pitch_img, X, Y, "ball" )

  # Copy our frame
  annotated_frame = new_frame.copy()
  # Overlay it with pitch stuff
  pitch.overlay_pitch( annotated_frame, pitch_img )

  x0, y0, _, _ = pitch.get_radar_location( annotated_frame )
  #draw_key_points( annotated_frame, cfg, x0, y0 )

  # And write to disk
  out.write( annotated_frame )

if out:
  out.release()
cap.release()
