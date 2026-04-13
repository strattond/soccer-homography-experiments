import cv2
import numpy as np
import supervision as sv
import torch
import argparse
from boxmot import ByteTrack, BotSort
from detectionadapter import DetectionAdapter
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser( description="Homography test" )
parser.add_argument( "-input",  help="Input video file",      type=str, default="test_homography_input.mp4" )
parser.add_argument( "-output", help="Output video file",     type=str, default="test_homography_output.mp4")
parser.add_argument( "-model",  help="YOLO model",            type=str, default=r"runs\detect\train16\weights\best.pt")
parser.add_argument( "-homo",   help="Calibrated homography", type=str, default="H_image_to_pitch.npy")
parser.add_argument( "-limit",  help="Frame limit",           type=int, default=50)

args = parser.parse_args()

print( "Configuring paths" )
input_video_path = args.input
output_video_path = args.output

# Load a Model for detection an prepare it
print( "Loading model" )
model = YOLO( args.model, verbose=False )

# Get Video Information
print( "Getting video information" )
cap    = cv2.VideoCapture( input_video_path )
width  = int(cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
height = int (cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
fourcc = int ( cap.get( cv2.CAP_PROP_FOURCC ) )
fps    = int ( cap.get( cv2.CAP_PROP_FPS ) )
new_w  = int(width * 0.5)
new_h  = int(height * 0.5)

# Progress resources
frame_limit: int = args.limit
frame_wrap:  int = 25
frame_count: int = 0

BALL_CLASS_ID = 32
PLAYER_CLASS_ID = 0

PITCH_HEIGHT = 270
PITCH_WIDTH = 480
PLAYER_RADIUS = 10

print( "Configuring trackers" )
device          = torch.device('cuda:0')
ball_tracker    = ByteTrack( frame_rate=fps, match_thresh=0.5, track_thresh=0.3, track_buffer=100 )
player_tracker  = BotSort( reid_weights=Path( "osnet_x0_25_msmt17.pt" ), device=device, frame_rate=fps, half=True )

print( "Configuring annotators" )
box_annotator   = sv.EllipseAnnotator( thickness=5 )
label_annotator = sv.LabelAnnotator( text_scale=0.25 )

print( "Configuring Writer" )
out = cv2.VideoWriter( output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h) )


PITCH_TEMPLATE_POINTS = np.array([
  [   0,     0],    # top-left corner
  [52.5,     0],    # top-middle
  [ 105,     0],    # top-right corner
  [ 105,    68],    # bottom-right corner
  [52.5,    68],    # bottom-middle
  [   0,    68],    # bottom-left corner
  [52.5, 29.43],    # top-centre-circle-edge
  [52.5, 38.57],    # bottom-centre-circle-edge
  [47.93,   34],    # left-centre-circle-edge
  [57.07,   34],    # right-centre-circle-edge
  [   0,  30.344],  # top-goalpost-edge
  [   0,  37.656],  # bottom-goalpost-edge
  # Add more known points if you want higher accuracy
], dtype=np.float32)

# Helper functions
def image_to_pitch( u, v, H ):
  pt = np.array( [ [ u, v, 1.0 ] ], dtype=np.float32 ).T
  mapped = H @ pt
  mapped /= mapped[ 2, 0 ]
  return mapped[ 0, 0 ], mapped[ 1, 0 ]

def bbox_bottom_center_to_pitch( x1, y1, x2, y2, H ):
  # Calculate the middle of the two x points
  u = 0.5 * (x1 + x2)
  # And the bottom y coordinate
  v = y2
  return image_to_pitch( u, v, H )

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

def draw_empty_pitch( width = PITCH_WIDTH, height = PITCH_HEIGHT ):
  
  # Blank out the image
  pitch = np.zeros( (height, width, 3), dtype=np.uint8 )

  # Draw outer rectangle
  cv2.rectangle( pitch, (5,5), (width-5, height-5), (255,255,255), 5 )

  # Centre line
  cv2.line( pitch, (width//2, 5), (width//2, height-5), (255,255,255), 5 )

  # Centre circle
  cv2.circle( pitch, (width//2, height//2), int(9.14*(width/105)), (255,255,255), 5 )

  return pitch

def pitch_to_overlay( X, Y, overlay_w, overlay_h ):
  # Default FIFA size is 105 x 68
  # X: 0–105, Y: 0–68
  px = int( (X / 105) * overlay_w )
  py = int( (Y / 68) * overlay_h )
  return px, py

def draw_player_on_pitch( pitch_img, X, Y, team ):
  h, w = pitch_img.shape[:2]
  px, py = pitch_to_overlay( X, Y, w, h )

  color = {
      "away": (255,0,255),
      "home": (0,0,255),
      "referee": (0,255,255),
      "unknown": (200,200,200),
      "ball": (255,255,255)
  }[team]

  cv2.circle( pitch_img, (px, py), PLAYER_RADIUS, color, -1 )
  
def overlay_pitch( frame, pitch_img ):
  fh, fw = frame.shape[:2]
  ph, pw = pitch_img.shape[:2]

  # bottom-middle placement
  x0 = fw//2 - pw//2
  y0 = fh - ph - 100

  # Copy existing part so we can alpha-blend
  roi = frame[y0:y0 + ph, x0:x0 + pw]
  alpha = 0.7
  blended = cv2.addWeighted( pitch_img, alpha, roi, 1 - alpha, 0 )
  frame[y0:y0 + ph, x0:x0 + pw] = blended
  
def extract_jersey( frame, x1, y1, x2, y2, person_mask ):
  roi = frame[y1:y2, x1:x2]
  h, w = roi.shape[:2]

  # Torso band relative to bottom of bbox
  y_bot = h
  y_top = 0

  x_left  = 0
  x_right = w

  torso_mask = person_mask[y_top:y_bot, x_left:x_right]
  torso_img  = roi[y_top:y_bot, x_left:x_right]

  # Apply mask
  jersey = cv2.bitwise_and( torso_img, torso_img, mask=torso_mask )

  if jersey.size < 50:
      return None

  return jersey
  
def get_person_mask( frame, x1, y1, x2, y2 ):
  
  # Crop region
  roi = frame[y1:y2, x1:x2]
  h, w = roi.shape[:2]

  if h < 10 or w < 10:
      return None  # too small

  # Init mask + models
  mask    = np.zeros( (h, w), np.uint8 )
  bgModel = np.zeros( (1, 65), np.float64 )
  fgModel = np.zeros( (1, 65), np.float64 )

  # Run GrabCut with rectangle init
  rect = (1, 1, w-2, h-2)  # slightly inset to avoid edges
  cv2.grabCut(roi, mask, rect, bgModel, fgModel, 3, cv2.GC_INIT_WITH_RECT)

  # Convert mask to binary person mask
  person_mask = np.where( (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0 ).astype("uint8")
  return person_mask

H = np.load(args.homo)
pitch_base = draw_empty_pitch()

print( "Looping" )
while True:
  ret, frame = cap.read()
  if not ret:
    break
  frame_count += 1
  if frame_count > frame_limit:
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
  for det in player_dets:
    x1f, y1f, x2f, y2f, conf, cid = det
    
    # Team colour classifier
    x1, y1, x2, y2 = map(int, (x1f, y1f, x2f, y2f))
    
    person_mask = get_person_mask( new_frame, x1, y1, x2, y2 )
    if person_mask is None:
      continue
    
    jersey_patch = extract_jersey( new_frame, x1, y1, x2, y2, person_mask )
    if jersey_patch is None:
      continue
    
    hsv = cv2.cvtColor( jersey_patch, cv2.COLOR_BGR2HSV )
    team = classify_team( hsv )
    if team == "referee":
      continue
    
    # Homography mapping
    X, Y = bbox_bottom_center_to_pitch( x1, y1, x2, y2, H )

    # Draw on mini-pitch
    draw_player_on_pitch( pitch_img, X, Y, team )
    
  for det in ball_dets:
    x1, y1, x2, y2, conf, cid = det
    
    # Homography mapping
    X, Y = bbox_bottom_center_to_pitch( x1, y1, x2, y2, H )

    # Draw on mini-pitch
    draw_player_on_pitch( pitch_img, X, Y, "ball" )
    
  annotated_frame = new_frame.copy()
  overlay_pitch( annotated_frame, pitch_img )
  
  out.write( annotated_frame )

if out:
  out.release()
cap.release()
