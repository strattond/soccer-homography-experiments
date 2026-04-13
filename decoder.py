import cv2
from cv2 import VideoCapture
from timer import Timer
from queues import print_queue, decode_queue, timing_queue
import threading

class DecodeThread(threading.Thread):
    
  def __init__(self, input_path, cap: VideoCapture, frame_limit: int ):
    super().__init__()
    self.input_path         = input_path
    self.cap: VideoCapture  = cap
    self.frame_limit: int   = frame_limit
      
  def run(self):
    frame_count: int = 0
    t = Timer()
    t.start()
    width   = int( self.cap.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    height  = int( self.cap.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    new_w   = int( width * 0.5 )
    new_h   = int( height * 0.5 )
    while True:
      ret, frame = self.cap.read()
      if not ret:
        decode_queue.put(None)  # Signal end
        break
      new_frame = cv2.resize( frame, (new_w, new_h) )
      decode_queue.put( new_frame )
      frame_count += 1
      print_queue.put('.')
      if frame_count > self.frame_limit:
        decode_queue.put( None )
        break
    t.stop()
    timing_queue.put( f"Decode completed {t.duration():.3f} seconds ({frame_count / t.duration():.2f} fps)" )
    self.cap.release()
