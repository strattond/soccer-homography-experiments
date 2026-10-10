from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from enum import Enum, auto

import numpy as np

from soccer_homography.data import TrackSegment


class TrackingType( Enum ):
  PROCESSING = auto()
  CUR_TRACK = auto()


class MinimapTracking:

  def __init__( self, totalFrames: int, width: int, color: str, owner: FrameMinimap ):
    self.totalFrames = totalFrames
    self.color = color
    self.width = width
    self.owner = owner
    self.clearFrames()

  def clearFrames( self ):

    # One bit per frame
    self.processedFrames = np.zeros( self.totalFrames, dtype=bool )

  def markFramesAsDone( self, frame_indices ):
    """Mark multiple frames at once."""

    frame_indices = np.asarray( frame_indices )
    # Sanitise it to those that are in range
    frame_indices = frame_indices[ ( frame_indices >= 0 ) & ( frame_indices < self.totalFrames ) ]

    self.processedFrames[ frame_indices ] = True

  def clear( self ):
    self.processedFrames.fill( False )

  # ------------------------------------------------------------------
  # Range conversion
  # ------------------------------------------------------------------

  def convertMaskToRanges( self ):
    """
      Convert boolean mask into contiguous ranges.
      Example: [1,1,1,0,0,1,1,0]
      Returns: [(0,2), (5,6)]
      """

    if not np.any( self.processedFrames ):
      return []

    mask = self.processedFrames.astype( np.int8 )

    diff = np.diff( mask )

    starts = np.where( diff == 1 )[ 0 ] + 1
    ends = np.where( diff == -1 )[ 0 ]

    # If the frames start with one being processed, there's no transition to start
    if mask[ 0 ]:
      starts = np.insert( starts, 0, 0 )

    # And if the frames end with one being processed, make sure to capture that as well
    if mask[ -1 ]:
      ends = np.append( ends, len( mask ) - 1 )

    return list( zip( starts, ends ) )

  def redraw( self ):

    width = self.owner.winfo_width()
    height = self.owner.winfo_height()

    if width <= 1 or height <= 1:
      return

    for start, end in self.convertMaskToRanges():
      if self.totalFrames <= 0:
        continue
      y0 = height * ( 1 - start / self.totalFrames )
      y1 = height * ( 1 - ( end+1 ) / self.totalFrames )
      self.owner.create_rectangle( 0, y1, self.width, y0, fill=self.color, outline="" )


class FrameMinimap( tk.Canvas ):

  def __init__(
      self,
      master,
      *,
      totalFrames: int,
      width: int = 30,
      height: int = 200,
      colorBG="#2b2b2b",
      colorTrack="#404040",
      colorProcessed="#4CAF50",
      colorFrameCurrent="#FFD54F",
      on_frame_select: Callable[ [ int ], None ] | None = None,
      **kwargs,
  ):
    super().__init__( master, width=width, height=height, bg=colorBG, highlightthickness=0, **kwargs )

    self.tracking: dict[ TrackingType, MinimapTracking ] = {}
    self.processed = MinimapTracking( totalFrames, width, colorProcessed, self )
    self.curTrack = MinimapTracking( totalFrames, width // 2, "#FF0000", self )
    self.tracking[ TrackingType.PROCESSING ] = self.processed
    self.tracking[ TrackingType.CUR_TRACK ] = self.curTrack

    self.updateTotalFrames( totalFrames )

    self.colorTrack = colorTrack
    self.colorCurrentFrame = colorFrameCurrent
    self.on_frame_select = on_frame_select

    self.currentFrame = None
    self.trackSegments: list[ TrackSegment ] = []

    self.bind( "<Configure>", lambda _: self.redraw() )
    self.bind( "<Button-1>", self.onClick )

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def updateTotalFrames( self, newValue: int ):
    self.totalFrames = newValue
    for k, v in self.tracking.items():
      v.totalFrames = newValue
      self.clearFrames( k )
    self.redraw()

  def clearFrames( self, mode: TrackingType = TrackingType.PROCESSING ):

    self.tracking[ mode ].clearFrames()

  def markFramesAsDone( self, frame_indices, mode: TrackingType = TrackingType.PROCESSING ):
    """Mark multiple frames at once."""

    self.tracking[ mode ].markFramesAsDone( frame_indices )

  def clear( self, mode: TrackingType = TrackingType.PROCESSING ):
    self.tracking[ mode ].clear()
    self.redraw()

  def setCurrentFrame( self, frame_idx: int ):
    self.currentFrame = frame_idx
    self.redraw()

  def setTrackSegments( self, segments: list[ TrackSegment ] ) -> None:
    self.trackSegments = segments
    self.redraw()

  # ------------------------------------------------------------------
  # Drawing
  # ------------------------------------------------------------------

  def getYForFrame( self, frame_idx: int ) -> float:
    height = self.winfo_height()
    if self.totalFrames <= 1:
      return float( height )
    frame_idx = min( max( frame_idx, 0 ), self.totalFrames - 1 )
    return height * ( 1 - frame_idx / ( self.totalFrames - 1 ) )

  def getFrameForY( self, y: int ) -> int:
    height = self.winfo_height()
    if self.totalFrames <= 1 or height <= 1:
      return 0
    y = min( max( y, 0 ), height - 1 )
    return round( ( height - 1 - y ) * ( self.totalFrames - 1 ) / ( height - 1 ) )

  def onClick( self, event ) -> None:
    if self.on_frame_select is not None and self.totalFrames > 0:
      self.on_frame_select( self.getFrameForY( event.y ) )

  def redraw( self ):
    self.delete( "all" )

    width = self.winfo_width()
    height = self.winfo_height()

    if width <= 1 or height <= 1:
      return

    # Background track

    self.create_rectangle( 0, 0, width, height, fill=self.colorTrack, outline="" )

    # Processed regions

    for v in self.tracking.values():
      v.redraw()

    for segment in self.trackSegments:
      start_y = self.getYForFrame( segment.frame_start )
      end_y = self.getYForFrame( segment.frame_end )
      self.create_rectangle(
          0,
          min( start_y, end_y ),
          max( 1, self.winfo_width() // 2 ),
          max( min( start_y, end_y ) + 1, max( start_y, end_y ) ),
          fill="#0000ff" if segment.person_id is not None else "#ff0000",
          outline="",
      )

    # Current frame marker

    if self.currentFrame is not None:
      y = self.getYForFrame( self.currentFrame )
      self.create_line( 0, y, width, y, width=2, fill=self.colorCurrentFrame )
