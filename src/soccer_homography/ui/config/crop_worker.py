import queue
import threading
from bisect import bisect_left, insort
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from soccer_homography.dataTypes import BoundingBox
from soccer_homography.log import logger

CropSet = list[ tuple[ int, np.ndarray ] ]
CropCache = dict[ int, CropSet ]
TrackBoxes = tuple[ int, list[ BoundingBox ] ]


@dataclass( slots=True )
class CropJobMessage:
  generation: int
  kind: str
  completed: int = 0
  total: int = 0
  crops: CropCache | None = None
  error: Exception | None = None
  stage: str = ""


@dataclass( frozen=True, slots=True )
class PlannedCropFrame:
  frameNumber: int
  trackBoxes: tuple[ tuple[ int, BoundingBox ], ...]


def planCropFrames(
    tracks: list[ TrackBoxes ],
    max_crops: int = 6,
    cancel_event: threading.Event | None = None,
) -> list[ PlannedCropFrame ]:
  boxesByFrame: dict[ int, dict[ int, BoundingBox ] ] = {}
  framesByTrack: dict[ int, set[ int ] ] = {}
  for track_id, boxes in tracks:
    for box in boxes:
      boxesByFrame.setdefault( box.frame, {} ).setdefault( track_id, box )
      framesByTrack.setdefault( track_id, set() ).add( box.frame )

  frameNums = sorted( boxesByFrame )
  if not frameNums:
    return []

  center = ( frameNums[ 0 ] + frameNums[ -1 ] ) / 2
  trackRemaining = { track_id: min( max_crops, len( frame_numbers_for_track ) ) for track_id, frame_numbers_for_track in framesByTrack.items() }
  frameScores = { frame: sum( trackRemaining[ track_id ] > 0 for track_id in tracks_by_frame ) for frame, tracks_by_frame in boxesByFrame.items() }
  pendingFrames = set( frameNums )
  selFrames: list[ int ] = []
  sortedSelFrames: list[ int ] = []

  def distanceFromSelected( frame: int ) -> int:
    if not sortedSelFrames:
      return 0
    index = bisect_left( sortedSelFrames, frame )
    distances = []
    if index < len( sortedSelFrames ):
      distances.append( sortedSelFrames[ index ] - frame )
    if index > 0:
      distances.append( frame - sortedSelFrames[ index - 1 ] )
    return min( distances )

  while pendingFrames:
    if cancel_event is not None and cancel_event.is_set():
      return []
    frame = max(
        pendingFrames,
        key=lambda candidate: (
            frameScores[ candidate ],
            distanceFromSelected( candidate ),
            -abs( candidate - center ),
            -candidate,
        ),
    )
    if frameScores[ frame ] == 0:
      break

    pendingFrames.remove( frame )
    selFrames.append( frame )
    insort( sortedSelFrames, frame )
    for track_id in boxesByFrame[ frame ]:
      if trackRemaining[ track_id ] == 0:
        continue
      trackRemaining[ track_id ] -= 1
      if trackRemaining[ track_id ] > 0:
        continue
      for other_frame in framesByTrack[ track_id ] & pendingFrames:
        frameScores[ other_frame ] -= 1

  fallbackFrames = sorted(
      pendingFrames,
      key=lambda frame: (
          -len( boxesByFrame[ frame ] ),
          -distanceFromSelected( frame ),
          abs( frame - center ),
          frame,
      ),
  )
  orderedFrames = selFrames + fallbackFrames
  return [ PlannedCropFrame( frame, tuple( boxesByFrame[ frame ].items() ) ) for frame in orderedFrames ]


class CropExtractionWorker( threading.Thread ):

  def __init__(
      self,
      generation: int,
      video_file: str,
      tracks: list[ TrackBoxes ],
      results: queue.Queue[ CropJobMessage ],
  ) -> None:
    super().__init__( daemon=True, name=f"clip-crops-{generation}" )
    self.generation = generation
    self.video_file = video_file
    self.tracks = tracks
    self.cancel_event = threading.Event()
    self.results = results

  def cancel( self ) -> None:
    self.cancel_event.set()

  def run( self ) -> None:
    capture: cv2.VideoCapture | None = None
    try:
      plan = planCropFrames( self.tracks, cancel_event=self.cancel_event )
      if self.cancel_event.is_set():
        return
      capture = cv2.VideoCapture( self.video_file )
      if not capture.isOpened():
        raise RuntimeError( f"Could not open video: {self.video_file}" )

      cache: CropCache = { track[ 0 ]: [] for track in self.tracks }
      pendingByTrack = Counter( assignment[ 0 ] for plannedFrame in plan for assignment in plannedFrame.trackBoxes )
      self.results.put( CropJobMessage( self.generation, "progress", 0, len( plan ), stage="planned frames" ) )
      completed = 0
      for plannedFrame in plan:
        if self.cancel_event.is_set():
          return
        tracksNeedingCrops = [ assignment for assignment in plannedFrame.trackBoxes if len( cache[ assignment[ 0 ] ] ) < 6 ]
        if tracksNeedingCrops:
          capture.set( cv2.CAP_PROP_POS_FRAMES, plannedFrame.frameNumber )
          success, frame = capture.read()
          if success:
            for track_id, box in tracksNeedingCrops:
              crop = cropFromFrame( frame, box )
              if crop is not None and len( cache[ track_id ] ) < 6:
                cache[ track_id ].append( ( plannedFrame.frameNumber, crop ) )
          else:
            logger.warning( f"Could not read crop frame {plannedFrame.frameNumber}." )

        for assignment in plannedFrame.trackBoxes:
          pendingByTrack[ assignment[ 0 ] ] -= 1
        completed += 1
        self.results.put( CropJobMessage( self.generation, "progress", completed, len( plan ), stage="frames" ) )
        if all( len( cache[ track[ 0 ] ] ) >= 6 or pendingByTrack[ track[ 0 ] ] == 0 for track in self.tracks ):
          break

      if not self.cancel_event.is_set():
        for crops in cache.values():
          crops.sort( key=lambda crop: crop[ 0 ] )
        self.results.put( CropJobMessage( self.generation, "done", completed, len( plan ), cache ) )
    except Exception as error:
      if not self.cancel_event.is_set():
        self.results.put( CropJobMessage( self.generation, "error", error=error ) )
    finally:
      if capture is not None:
        capture.release()


def cropFromFrame( frame: np.ndarray, box: BoundingBox ) -> np.ndarray | None:
  height, width = frame.shape[ :2 ]
  x1 = max( 0, min( width, box.x1 ) )
  y1 = max( 0, min( height, box.y1 ) )
  x2 = max( 0, min( width, box.x2 ) )
  y2 = max( 0, min( height, box.y2 ) )
  if x2 <= x1 or y2 <= y1:
    return None
  crop = cv2.cvtColor( frame[ y1:y2, x1:x2 ], cv2.COLOR_BGR2RGB )
  crop_height, crop_width = crop.shape[ :2 ]
  scale = min( 100 / crop_width, 150 / crop_height, 1.0 )
  if scale < 1.0:
    crop = cv2.resize( crop, ( max( 1, round( crop_width * scale ) ), max( 1, round( crop_height * scale ) ) ) )
  return crop
