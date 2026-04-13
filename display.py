import threading
from queues import print_queue, timing_queue

class PrintThread(threading.Thread):
  def __init__(self, frame_wrap: int):
    super().__init__()
    self.frame_wrap: int = frame_wrap
    self.frame_count: int = 0

  def run(self):
    while True:
      frame = print_queue.get()
      if frame is None:
        break

      self.frame_count += 1
      print(frame, end='', flush=True)

      if self.frame_count % self.frame_wrap == 0:
        print('', flush=True)
    
    print( '' )
    while not timing_queue.empty():
      line = timing_queue.get()
      print( line )