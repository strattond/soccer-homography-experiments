from timer import Timer
import cv2
import threading
from ultralytics import YOLO

from display import PrintThread
from decoder import DecodeThread
from writer import WriterThread
from tracker import TrackerThread

print( "Configuring paths" )
input_video_path = r"test_input"
output_video_path = "test_action.mp4"

# Load a Model for detection an prepare it
print( "Loading model" )
model = YOLO(r"models\yolo26n.pt", verbose=False)

# Get Video Information
print( "Getting video information" )
cap     = cv2.VideoCapture( input_video_path )
width   = int( cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
height  = int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
fourcc  = int( cap.get( cv2.CAP_PROP_FOURCC ) )
fps     = int( cap.get( cv2.CAP_PROP_FPS ) )

# Progress resources
frame_limit: int = 500

print( "Initializing queues" )

printer = PrintThread( frame_wrap = 50 )
decoder = DecodeThread( input_video_path, cap, frame_limit )
writer = WriterThread( output_video_path, width, height, fps )
tracker = TrackerThread( model, fps )

threads: list[threading.Thread] = [ decoder, tracker, writer, printer ]

print( "Starting" )

for t in threads:
  t.start()
for t in threads:
  t.join()
