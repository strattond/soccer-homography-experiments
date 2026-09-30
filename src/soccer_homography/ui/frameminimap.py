import tkinter as tk

import numpy as np


class FrameMinimap( tk.Canvas ):

  def __init__(
      self,
      master,
      *,
      total_frames: int,
      width: int = 20,
      height: int = 200,
      bg_color="#2b2b2b",
      track_color="#404040",
      processed_color="#4CAF50",
      current_frame_color="#FFD54F",
      **kwargs,
  ):
    super().__init__( master, width=width, height=height, bg=bg_color, highlightthickness=0, **kwargs )

    self.updateTotalFrames( total_frames )

    self.track_color = track_color
    self.processed_color = processed_color
    self.current_frame_color = current_frame_color

    self.current_frame = None

    self.bind( "<Configure>", lambda _: self.redraw() )

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def updateTotalFrames( self, newValue: int ):
    self.total_frames = newValue
    self.clearFrames()
    self.redraw()

  def clearFrames( self ):

    # One bit per frame
    self.processedFrames = np.zeros( self.total_frames, dtype=bool )

  def markFrameAsDone( self, frame_idx: int ):
    """Mark a single frame as processed."""
    # Bounds check it to ensure we don't walk outside the range
    if 0 <= frame_idx < self.total_frames:
      self.processedFrames[ frame_idx ] = True

  def markFramesAsDone( self, frame_indices ):
    """Mark multiple frames at once."""

    frame_indices = np.asarray( frame_indices )
    # Sanitise it to those that are in range
    frame_indices = frame_indices[ ( frame_indices >= 0 ) & ( frame_indices < self.total_frames ) ]

    self.processedFrames[ frame_indices ] = True

  def clear( self ):
    self.processedFrames.fill( False )
    self.redraw()

  def setCurrentFrame( self, frame_idx: int ):
    self.current_frame = frame_idx
    self.redraw()

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

  # ------------------------------------------------------------------
  # Drawing
  # ------------------------------------------------------------------

  def getYForFrame( self, frame_idx: int ) -> float:
    height = self.winfo_height()
    if self.total_frames <= 1:
      return float( height )
    frame_idx = min( max( frame_idx, 0 ), self.total_frames - 1 )
    return height * ( 1 - frame_idx / ( self.total_frames - 1 ) )

  def redraw( self ):
    self.delete( "all" )

    width = self.winfo_width()
    height = self.winfo_height()

    if width <= 1 or height <= 1:
      return

    # Background track

    self.create_rectangle( 0, 0, width, height, fill=self.track_color, outline="" )

    # Processed regions

    for start, end in self.convertMaskToRanges():
      if self.total_frames <= 0:
        continue
      y0 = height * ( 1 - start / self.total_frames )
      y1 = height * ( 1 - ( end + 1 ) / self.total_frames )
      self.create_rectangle( 0, y1, width, y0, fill=self.processed_color, outline="" )

    # Current frame marker

    if self.current_frame is not None:
      y = self.getYForFrame( self.current_frame )
      self.create_line( 0, y, width, y, width=2, fill=self.current_frame_color )
