# timer.py
import time

class Timer:
  def __init__(self):
    self.start_time = None
    self.end_time = None

  def start(self):
    self.start_time = time.perf_counter()
    self.end_time = None

  def stop(self):
    self.end_time = time.perf_counter()

  def duration(self):
    if self.start_time is None:
      raise RuntimeError("Timer was never started.")
    if self.end_time is None:
      return time.perf_counter() - self.start_time
    return self.end_time - self.start_time

  def reset(self):
    self.start_time = None
    self.end_time = None