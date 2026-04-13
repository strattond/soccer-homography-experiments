import numpy as np
import supervision as sv

class DetectionAdapter:
  """
  Normalises detector outputs (RF-DETR, YOLO, etc.)
  into a unified structure:
      .xyxy        → Nx4 numpy array
      .class_id    → N array of ints
      .confidence  → N array of floats
  """

  def __init__(self, raw):
    self.raw = raw

    # Detect YOLO (Ultralytics) format
    if isinstance( raw, list ) and hasattr( raw[0], "boxes" ):
      self.from_yolo( raw[0] )

    # Detect RF-DETR format (your old structure)
    elif hasattr( raw, "class_id" ) and hasattr( raw, "xyxy" ):
      self.from_rfdetr( raw )

    else:
      raise TypeError("Unknown detection format")

  # -----------------------------
  # YOLO converter
  # -----------------------------
  def from_yolo( self, result ):
    boxes = result.boxes

    self.xyxy       = boxes.xyxy.cpu().numpy()
    self.class_id   = boxes.cls.cpu().numpy().astype(int)
    self.confidence = boxes.conf.cpu().numpy()

  # -----------------------------
  # RF-DETR converter
  # -----------------------------
  def from_rfdetr(self, det):
    self.xyxy       = det.xyxy
    self.class_id   = det.class_id
    self.confidence = det.confidence

  # -----------------------------
  # Masking support
  # -----------------------------
  def mask(self, mask):
    return self.__getitem__( mask )

  # -----------------------------
  # Support slicing: detections[mask]
  # ----------------------------- 
  def __getitem__( self, mask ):
    new = DetectionAdapter.__new__(DetectionAdapter) 
    new.raw = None 
    new.xyxy = self.xyxy[mask] 
    new.class_id = self.class_id[mask] 
    new.confidence = self.confidence[mask] 
    return new

  def __len__( self ):
    return 0 if self.xyxy is None else len( self.xyxy )
  
  def to_supervision( self ):
    return sv.Detections( xyxy=self.xyxy, confidence=self.confidence, class_id=self.class_id, )