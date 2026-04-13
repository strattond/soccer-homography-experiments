import cv2
import numpy as np

#VIDEO_PATH  = r"D:\Media\Oxley\2026\Metro 10\20260321\Action\DJI_20260321130417_0060_D.MP4"          # change this
VIDEO_PATH  = r"test_homography_input.MP4"
FRAME_INDEX = 50                   # which frame to use for calibration
H_OUTPUT    = "H_image_to_pitch.npy"

# --- grab calibration frame ---
cap = cv2.VideoCapture( VIDEO_PATH )
if not cap.isOpened():
  raise SystemExit( f"Could not open {VIDEO_PATH}" )

cap.set( cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX )
ret, raw_img = cap.read()

width  = int(cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
height = int (cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
new_w  = int(width * 0.5 )
new_h  = int(height * 0.5 )
img    = cv2.resize( raw_img, (new_w, new_h) )

cap.release()
if not ret:
  raise SystemExit( f"Could not read frame {FRAME_INDEX} from {VIDEO_PATH}" )

img_pts = []

def on_click( event, x, y, flags, param ):
  if event == cv2.EVENT_LBUTTONDOWN:
    img_pts.append( [x, y] )
    print( f"Image point {len(img_pts)-1}: ({x}, {y})")
    cv2.circle(  img, (x, y), 4, (0, 0, 255), -1 )
    cv2.putText( img, str(len(img_pts)-1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1 )
    cv2.imshow(  "calibrate", img )

cv2.imshow( "calibrate", img )
cv2.setMouseCallback( "calibrate", on_click )
print( "Click field markings (sideline, circle arc, etc.)." )
print( "Press ESC when done." )
while True:
  key = cv2.waitKey(1)
  if key == 27:  # ESC
    break

cv2.destroyAllWindows()

if len(img_pts) < 4:
  raise SystemExit("Need at least 4 points for homography.")

world_pts = []
print("\nEnter world (pitch) coordinates for each point.")
print("Use your ideal pitch model, e.g. 105 x 68, in meters.")
for i, (u, v) in enumerate(img_pts):
  X = float( input( f"World X for point {i} (image {u},{v}): " ) )
  Y = float( input( f"World Y for point {i} (image {u},{v}): " ) )
  world_pts.append( [X, Y] )

img_pts   = np.array(img_pts,   dtype=np.float32)
world_pts = np.array(world_pts, dtype=np.float32)

H, mask = cv2.findHomography( img_pts, world_pts, method=cv2.RANSAC )
np.save( H_OUTPUT, H )

print( f"\nSaved homography to {H_OUTPUT}" )
print( "H =\n", H )
