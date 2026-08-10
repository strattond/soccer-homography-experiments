import numpy as np
import supervision as sv
import ultralytics


class DetectionAdapter:
  """
  Normalises detector outputs (RF-DETR, YOLO, etc.)
  into a unified structure:
      .xyxy        → Nx4 numpy array
      .class_id    → N array of ints
      .confidence  → N array of floats
      .trackID     → N array of ints
  """

  def __init__( self, raw ):
    self.raw = raw
    # Detect YOLO (Ultralytics) format
    if isinstance( raw, list ) and hasattr( raw[ 0 ], "boxes" ):
      #print( raw[0] )
      self.from_yolo( raw[ 0 ] )

    elif isinstance( raw, ultralytics.engine.results.Results ):
      self.from_yolo( raw )

    # Detect RF-DETR format (your old structure)
    elif hasattr( raw, "class_id" ) and hasattr( raw, "xyxy" ):
      self.from_rfdetr( raw )

    else:
      raise TypeError( "Unknown detection format" )

  # -----------------------------
  # YOLO converter
  # -----------------------------
  def from_yolo( self, result ):
    boxes = result.cpu().boxes

    self.trackID = boxes.id.numpy().astype( int ) if hasattr( boxes, 'id' ) else np.full( len( boxes.xyxy ), -1, dtype=np.int32 )
    self.xyxy = boxes.xyxy.numpy()
    self.class_id = boxes.cls.numpy().astype( int )
    self.confidence = boxes.conf.numpy()

  # -----------------------------
  # RF-DETR converter
  # -----------------------------
  def from_rfdetr( self, det ):
    self.xyxy = det.xyxy
    self.class_id = det.class_id
    self.confidence = det.confidence

  # -----------------------------
  # Masking support
  # -----------------------------
  def mask( self, mask ):
    return self.__getitem__( mask )

  # -----------------------------
  # Support slicing: detections[mask]
  # -----------------------------
  def __getitem__( self, mask ):
    new = DetectionAdapter.__new__( DetectionAdapter )
    new.raw = None  # self.raw[mask]
    new.xyxy = self.xyxy[ mask ]
    new.class_id = self.class_id[ mask ]
    new.confidence = self.confidence[ mask ]
    new.trackID = self.trackID[ mask ] if self.trackID is not None else None
    return new

  def __len__( self ):
    return 0 if self.xyxy is None else len( self.xyxy )

  def to_supervision( self ):
    return sv.Detections(
        xyxy=self.xyxy,
        confidence=self.confidence,
        class_id=self.class_id,
    )
