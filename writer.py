import cv2
import numpy as np
from queues import print_queue, annotate_queue, timing_queue
from timer import Timer
import threading

PITCH_TEMPLATE_POINTS = np.array([
    [0, 0],          # top-left corner
    [105, 0],        # top-right corner
    [105, 68],       # bottom-right corner
    [0, 68],         # bottom-left corner
    # Add more known points if you want higher accuracy
], dtype=np.float32)


class WriterThread(threading.Thread):
  def __init__(self, output_path, width: int, height: int, fps: int ):
    super().__init__()
    self.output_path = output_path
    self.width = width
    self.height = height
    self.fps = fps
      
  def run(self):
    frame_count = 0
    t = Timer()
    t.start()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    new_w = int(self.width * 0.5)
    new_h = int(self.height * 0.5)
    out = cv2.VideoWriter( self.output_path, fourcc, self.fps, (new_w, new_h) )
    while True:
      frame = annotate_queue.get()
      if frame is None:
        break
      resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
      out.write(resized_frame)
      frame_count += 1
      print_queue.put('o')

    if out:
      out.release()
        
    print_queue.put(None)
    timing_queue.put( f"Writing completed {t.duration():.3f} seconds ({frame_count / t.duration():.2f} fps)" )
