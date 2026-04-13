import os
from pathlib import Path
import queue
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from boxmot import ByteTrack, BotSort, DeepOcSort
from timer import Timer
from queues import decode_queue, annotate_queue, print_queue, timing_queue
from detectionadapter import DetectionAdapter
import threading

def save_debug_image(name, image):
    cv2.imwrite(os.path.join("debug_outputs", name), image)

predict_queue = queue.Queue(maxsize=128)

# Class IDs from your RF-DETR model
BALL_CLASS_ID = 37
PLAYER_CLASS_ID = 0 # 1

class InternalPredictor(threading.Thread):
  BATCH_SIZE = 16
  batch_frames = []
  
  def __init__(self, model: YOLO):
    super().__init__()
    self.model = model
    
  def drainRemainder(self):
    if self.batch_frames:
      self.processBatch()          
    predict_queue.put( (None, None) )
    
  def processBatch(self):
    results = self.model.predict( self.batch_frames, verbose=False )
    for f, r in zip( self.batch_frames, results ):
      predict_queue.put( (f, [r]) )
      print_queue.put( '+' )
    self.batch_frames.clear()
        
  def run(self):
    frame_count = 0
    t = Timer()
    t.start()
    while True:
      frame = decode_queue.get()
      if frame is None:
        self.drainRemainder()
        break

      self.batch_frames.append( frame )
      if len(self.batch_frames) == self.BATCH_SIZE:
        self.processBatch()
      frame_count += 1
    t.stop()
    timing_queue.put( f"Predicting completed {t.duration():.3f} seconds ({frame_count / t.duration():.2f} fps)" )
    
class InternalTracker(threading.Thread):
  def __init__(self, model: YOLO, fps: int):
    super().__init__()
    self.model = model
    self.fps = fps

    # Initialize our trackers - one for the ball, one for the players
    print( "Initializing device and trackers" )

    self.device = torch.device('cuda:0')
    self.ball_tracker = ByteTrack(frame_rate=fps, match_thresh=0.5, track_thresh=0.3, track_buffer=100)
    self.player_tracker = BotSort(reid_weights=Path("osnet_x0_25_msmt17.pt"), device=self.device, frame_rate=fps, half=True)

    # Our annotators
    print( "Initializing annotators" )
    self.box_annotator = sv.EllipseAnnotator( thickness=5 )
    self.label_annotator = sv.LabelAnnotator( text_scale=0.25 )
        
  def run(self):
    frame_count = 0
    t = Timer()
    t.start()
    while True:
      frame, results = predict_queue.get()
      if frame is None:
        annotate_queue.put(None)
        break

      detections = DetectionAdapter(results)
      keep_ids = {PLAYER_CLASS_ID, BALL_CLASS_ID}
      all_mask = [cid in keep_ids for cid in detections.class_id]

      #print("YOLO class IDs:", detections.class_id)
      detections = detections[all_mask]
      
      # Split detections by COCO class ID
      ball_mask = detections.class_id == BALL_CLASS_ID
      player_mask = detections.class_id == PLAYER_CLASS_ID
      
      # Swizzle them into a way that BoxMot supports
      balls = detections[ball_mask]
      ball_dets = np.hstack((
          balls.xyxy,
          balls.confidence[:, None],
          balls.class_id[:, None]
      ))

      players = detections[player_mask]
      player_dets = np.hstack((
          players.xyxy,
          players.confidence[:, None],
          players.class_id[:, None]
      ))

      # Update trackers
      ball_tracks = self.ball_tracker.update(ball_dets, frame)
      player_tracks = self.player_tracker.update(player_dets, frame)
      
      # Visualise with BOXMOT's built-in trails
      annotated_frame = frame.copy()
      annotated_frame = self.ball_tracker.plot_results( annotated_frame, show_trajectories=True)
      annotated_frame = self.player_tracker.plot_results( annotated_frame, show_trajectories=True )
    
#                f"{COCO_CLASSES[class_id]} {confidence:.2f}"
      labels = [
          f"{class_id} {confidence:.2f}"
          for class_id, confidence
          in zip(detections.class_id, detections.confidence)
      ]
      sv_dets = detections.to_supervision()
      #print( str(len(labels)) + "," + str(len(sv_dets)) )
      annotated_frame = self.box_annotator.annotate( annotated_frame, detections=sv_dets )
      annotated_frame = self.label_annotator.annotate( annotated_frame, detections=sv_dets, labels=labels )
      annotate_queue.put(annotated_frame)
      frame_count += 1
      print_queue.put('x')
    t.stop()
    timing_queue.put( f"Tracking completed {t.duration():.3f} seconds ({frame_count / t.duration():.2f} fps)" )

class TrackerThread(threading.Thread):
  def __init__(self, model: YOLO, fps: int):
    super().__init__()
    self.model = model
    self.fps = fps
    self.predictor = InternalPredictor( model )
    self.tracker = InternalTracker( model, fps )
        
  def run(self):
    self.predictor.start()
    self.tracker.start()
    
    self.predictor.join()
    self.tracker.join()
      