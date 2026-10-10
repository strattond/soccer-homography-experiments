import queue
import threading
from bisect import bisect_left, insort
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from soccer_homography.data import BoundingBox
from soccer_homography.log import logger

CropSet = list[ tuple[ int, np.ndarray ] ]
TrackSegmentKey = tuple[ int, int ]
CropCache = dict[ TrackSegmentKey, CropSet ]
TrackBoxes = tuple[ int, int, list[ BoundingBox ] ]
LegacyTrackBoxes = tuple[ int, list[ BoundingBox ] ]


def normalizeTrackBoxes( item: TrackBoxes | LegacyTrackBoxes ) -> TrackBoxes:
  if len( item ) == 2:
    track_id, boxes = cast( LegacyTrackBoxes, item )
    return track_id, min( ( box.frame for box in boxes ), default=0 ), boxes
  track_id, segment_start, boxes = cast( TrackBoxes, item )
  return track_id, segment_start, boxes


def deleteTrackCrops( clip_id: int, track_ids: set[ int ] | None = None ) -> int:
  crop_directory = Path( "crops" ) / str( clip_id )
  if not crop_directory.is_dir():
    return 0

  deleted = 0
  for crop_path in crop_directory.glob( "*.png" ):
    parts = crop_path.stem.split( "_" )
    if len( parts ) == 3 and all( part.isdecimal() for part in parts ):
      track_id = int( parts[ 0 ] )
    elif len( parts ) == 2 and all( part.isdecimal() for part in parts ):
      track_id = int( parts[ 1 ] )
    else:
      continue
    if track_ids is not None and track_id not in track_ids:
      continue
    crop_path.unlink()
    deleted += 1
  return deleted


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
  trackBoxes: tuple[ tuple[ TrackSegmentKey, BoundingBox ], ...]


def planCropFrames(
    tracks: Sequence[ TrackBoxes | LegacyTrackBoxes ],
    max_crops: int = 6,
    cancel_event: threading.Event | None = None,
) -> list[ PlannedCropFrame ]:
  boxesByFrame: dict[ int, dict[ TrackSegmentKey, BoundingBox ] ] = {}
  framesBySegment: dict[ TrackSegmentKey, set[ int ] ] = {}
  for item in tracks:
    track_id, segment_start, boxes = normalizeTrackBoxes( item )
    segment_key = ( track_id, segment_start )
    for box in boxes:
      boxesByFrame.setdefault( box.frame, {} ).setdefault( segment_key, box )
      framesBySegment.setdefault( segment_key, set() ).add( box.frame )

  frameNums = sorted( boxesByFrame )
  if not frameNums:
    return []

  center = ( frameNums[ 0 ] + frameNums[ -1 ] ) / 2
  segmentRemaining = {
      segment_key: min( max_crops, len( frame_numbers_for_segment ) )
      for segment_key, frame_numbers_for_segment in framesBySegment.items()
  }
  frameScores = { frame: sum( segmentRemaining[ key ] > 0 for key in segments_by_frame ) for frame, segments_by_frame in boxesByFrame.items() }
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
    for segment_key in boxesByFrame[ frame ]:
      if segmentRemaining[ segment_key ] == 0:
        continue
      segmentRemaining[ segment_key ] -= 1
      if segmentRemaining[ segment_key ] > 0:
        continue
      for other_frame in framesBySegment[ segment_key ] & pendingFrames:
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
      *,
      clip_id: int,
      max_crops: int = 6,
  ) -> None:
    if max_crops < 1:
      raise ValueError( "The maximum number of crops per track segment must be positive." )
    super().__init__( daemon=True, name=f"clip-crops-{generation}" )
    self.generation = generation
    self.video_file = video_file
    self.tracks = tracks
    self.clip_id = clip_id
    self.max_crops = max_crops
    self.cancel_event = threading.Event()
    self.results = results

  def cancel( self ) -> None:
    self.cancel_event.set()

  def run( self ) -> None:
    logger.info( f"Starting crop extraction worker for generation {self.generation}" )
    capture: cv2.VideoCapture | None = None
    try:
      logger.info( f"Planning crop frames for generation {self.generation}" )
      plan = planCropFrames( self.tracks, self.max_crops, cancel_event=self.cancel_event )
      if self.cancel_event.is_set():
        return
      logger.info( f"Planned {len( plan )} crop frames for generation {self.generation}" )

      segment_keys = {
          ( track_id, segment_start )
          for track_id, segment_start, _ in map( normalizeTrackBoxes, self.tracks )
      }
      cache: CropCache = { key: [] for key in segment_keys }
      frames_by_segment = {
          ( track_id, segment_start ): { box.frame for box in boxes }
          for track_id, segment_start, boxes in map( normalizeTrackBoxes, self.tracks )
      }
      crop_directory = Path( "crops" ) / str( self.clip_id )
      pendingBySegment = Counter( assignment[ 0 ] for plannedFrame in plan for assignment in plannedFrame.trackBoxes )
      self.loadCachedCrops( cache, crop_directory, frames_by_segment )
      self.results.put( CropJobMessage( self.generation, "progress", 0, len( plan ), stage="planned frames" ) )
      completed = 0
      for plannedFrame in plan:
        if self.cancel_event.is_set():
          return

        segmentsNeedingCrops = [
            assignment
            for assignment in plannedFrame.trackBoxes
            if len( cache[ assignment[ 0 ] ] ) < self.max_crops
            and all( frame_number != plannedFrame.frameNumber for frame_number, _ in cache[ assignment[ 0 ] ] )
        ]
        # Get the crops from disk if they exist, otherwise extract them from the video frame
        uncachedSegments = [
            assignment for assignment in segmentsNeedingCrops
            if not self.cropPath( crop_directory, assignment[ 0 ], plannedFrame.frameNumber ).is_file()
        ]
        cached_segments = [ assignment for assignment in segmentsNeedingCrops if assignment not in uncachedSegments ]
        self.getUncachedSegmentsFromDisk( cache, crop_directory, plannedFrame, cached_segments )
        if uncachedSegments:
          # Load it if we haven't yet
          if capture is None:
            capture = cv2.VideoCapture( self.video_file )
            if not capture.isOpened():
              raise RuntimeError( f"Could not open video: {self.video_file}" )
          # Move to the right frame
          capture.set( cv2.CAP_PROP_POS_FRAMES, plannedFrame.frameNumber )
          success, frame = capture.read()
          if success:
            logger.info( f"Extracting {len(uncachedSegments)} crops for frame {plannedFrame.frameNumber}." )
            for segment_key, box in uncachedSegments:
              crop = cropFromFrame( frame, box )
              self.persistCrop( cache, crop_directory, plannedFrame, segment_key, crop )
          else:
            logger.warning( f"Could not read crop frame {plannedFrame.frameNumber}." )

        for assignment in plannedFrame.trackBoxes:
          pendingBySegment[ assignment[ 0 ] ] -= 1
        completed += 1
        self.results.put( CropJobMessage( self.generation, "progress", completed, len( plan ), stage="frames" ) )
        if all( len( cache[ key ] ) >= self.max_crops or pendingBySegment[ key ] == 0 for key in segment_keys ):
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

  def cropPath( self, crop_directory: Path, segment_key: TrackSegmentKey, frame_number: int ) -> Path:
    track_id, segment_start = segment_key
    return crop_directory / f"{track_id}_{segment_start}_{frame_number}.png"

  def persistCrop( self, cache, crop_directory, plannedFrame, segment_key, crop ):
    if crop is not None and len( cache[ segment_key ] ) < self.max_crops:
      crop_directory.mkdir( parents=True, exist_ok=True )
      crop_path = self.cropPath( crop_directory, segment_key, plannedFrame.frameNumber )
      saved = cv2.imwrite(
          str( crop_path ),
          cv2.cvtColor( crop, cv2.COLOR_RGB2BGR ),
      )
      if not saved:
        raise RuntimeError( f"Could not save crop image: {crop_path}" )
      cache[ segment_key ].append( ( plannedFrame.frameNumber, crop ) )

  def getUncachedSegmentsFromDisk( self, cache, crop_directory, plannedFrame, segmentsNeedingCrops ):
    for segment_key, _ in segmentsNeedingCrops:
      crop_path = self.cropPath( crop_directory, segment_key, plannedFrame.frameNumber )
      if crop_path.is_file():
        crop = cv2.imread( str( crop_path ), cv2.IMREAD_COLOR )
        if crop is not None:
          crop = cv2.cvtColor( crop, cv2.COLOR_BGR2RGB )
          cache[ segment_key ].append( ( plannedFrame.frameNumber, crop ) )
          continue
        logger.warning( f"Could not load cached crop {crop_path}; re-extracting it." )

  def loadCachedCrops(
      self,
      cache: CropCache,
      crop_directory: Path,
      frames_by_segment: dict[ TrackSegmentKey, set[ int ] ],
  ) -> None:
    if not crop_directory.is_dir():
      return
    for segment_key, crops in cache.items():
      track_id, segment_start = segment_key
      prefix = f"{track_id}_{segment_start}_"
      cached_frames = sorted(
          (
              ( int( path.stem[len(prefix):] ), path )
              for path in crop_directory.glob( f"{prefix}*.png" )
              if path.stem[len(prefix):].isdecimal()
              and int( path.stem[len(prefix):] ) in frames_by_segment[ segment_key ]
          ),
          key=lambda item: item[ 0 ],
      )
      for frame_number, crop_path in cached_frames[: self.max_crops]:
        crop = cv2.imread( str( crop_path ), cv2.IMREAD_COLOR )
        if crop is None:
          logger.warning( f"Could not load cached crop {crop_path}; it will be re-extracted if selected." )
          continue
        crops.append( ( frame_number, cv2.cvtColor( crop, cv2.COLOR_BGR2RGB ) ) )


def cropFromFrame( frame: np.ndarray, box: BoundingBox ) -> np.ndarray | None:
  height, width = frame.shape[ :2 ]
  x1 = max( 0, min( width, int( box.x1 ) ) )
  y1 = max( 0, min( height, int( box.y1 ) ) )
  x2 = max( 0, min( width, int( box.x2 ) ) )
  y2 = max( 0, min( height, int( box.y2 ) ) )
  if x2 <= x1 or y2 <= y1:
    return None
  crop = cv2.cvtColor( frame[ y1:y2, x1:x2 ], cv2.COLOR_BGR2RGB )
  crop_height, crop_width = crop.shape[ :2 ]
  scale = min( 100 / crop_width, 150 / crop_height, 1.0 )
  if scale < 1.0:
    crop = cv2.resize( crop, ( max( 1, round( crop_width * scale ) ), max( 1, round( crop_height * scale ) ) ) )
  return crop
