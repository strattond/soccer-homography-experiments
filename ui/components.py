from dataclasses import dataclass, field
import time
import tkinter as tk
from tkinter import ttk


class Slider:

  def __init__( self, from_: int, to: int, root: tk.Tk, command, x: int, y: int, width: int, height: int ):
    self.root = root
    self.min = from_
    self.max = to
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.command = command
    self.boundVar = tk.DoubleVar()
    self.interSnap = False
    self.slider = ttk.Scale( root, from_=from_, to=to, orient='horizontal', command=self.internalCommand, variable=self.boundVar )
    self.slider.place( x=x, y=y, width=width, height=height )
    self.lblRadar = tk.Label( self.root, fg="#000000", font=( "Arial", 10 ), anchor="center", text="" )
    self.lblRadar.place( x=x, y=y - height, width=width, height=height )
    self._debounce_job = None

  def setMax( self, value ):
    self.slider.config( to=value )

  def setEnabled( self, value ):
    self.slider.config( state=tk.NORMAL if value else tk.DISABLED )

  def internalCommand( self, value ):

    # Cancel any pending callback
    if self._debounce_job is not None:
      self.root.after_cancel( self._debounce_job )

    # Schedule a new one
    self.lblRadar.config( text=str( round( float( value ) ) ) )
    self._debounce_job = self.root.after( 200, lambda: self.on_debounce( value ) )

  def on_debounce( self, value ):
    if self.command is not None:
      if self.interSnap:
        return
      self.interSnap = True
      frame = round( float( value ) )
      self.boundVar.set( frame )
      self.command( frame )
      self.interSnap = False


class LabelledSpinBox:

  def __init__( self, from_: int, to: int, root: tk.Tk, x: int, y: int, width: int, height: int, label: str, offset: int, command=None, initValue=0 ):
    self.root = root
    self.min = from_
    self.max = to
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.command = command
    self.label = label
    self.internalVar = tk.IntVar()
    self.spinner = ttk.Spinbox( master=self.root, from_=0, to=100, command=self.internalCommand, textvariable=self.internalVar )
    self.spinner.place( x=x, y=y, width=width, height=height )
    self.label = ttk.Label( master=self.root, text=label, font=( "Arial", 12 ) )
    self.label.place( x=x - offset, y=y, width=offset, height=height )
    self._debounce_job = None
    self.spinner.set( initValue )

  def internalCommand( self ):
    # Cancel any pending callback
    if self._debounce_job is not None:
      self.root.after_cancel( self._debounce_job )

    # Schedule a new one
    self._debounce_job = self.root.after( 200, lambda: self.on_debounce( self.get() ) )

  def on_debounce( self, value: int ):
    if self.command is not None:
      self.command( value )

  def setMax( self, value: int ):
    self.spinner.config( to=value )  # type: ignore
    if self.get() > value:
      self.spinner.set( value )

  def get( self ) -> int:
    return int( self.spinner.get() )


@dataclass
class ProgressBarETA:

  # yapf: disable
  root:       tk.Widget | tk.Tk
  x:          int
  y:          int
  width:      int
  height:     int
  min:        int          = 0
  max:        int          = 100
  canvas:     tk.Canvas    = field( init=False )
  bgBar:      int          = field( init=False )
  fgBar:      int          = field( init=False )
  lblPercent: int          = field( init=False )
  lblETATime:    int          = field( init=False )
  currentVal: int          = field( default=0, init=False )
  startTime:  float | None = field( default=None, init=False )
  # yapf: enable

  def __post_init__( self ):
    # Create canvas
    self.canvas = tk.Canvas( self.root, width=self.width, height=self.height, bg="#222222", highlightthickness=0 )
    self.canvas.place( x=self.x, y=self.y )

    # Background bar
    self.bgBar = self.canvas.create_rectangle( 0, 0, self.width, self.height, fill="#444444", outline="" )

    # Foreground bar (starts empty)
    self.fgBar = self.canvas.create_rectangle( 0, self.height, self.width, self.height, fill="#00aa00", outline="" )

    # Rotated labels
    self.lblPercent = self.canvas.create_text( self.width // 2, self.height // 2 - 20, text="0%", fill="white", angle=90, font=( "Arial", 12, "bold" ) )
    self.lblETATime = self.canvas.create_text( self.width // 2, self.height // 2 + 20, text="--:--", fill="white", angle=90, font=( "Arial", 10 ) )
    self.lblDuratin = self.canvas.create_text( self.width // 2, self.height // 2 - 60, text="--:--", fill="white", angle=90, font=( "Arial", 10 ) )

  def setRange( self, val: int, max: int ):
    self.min = val
    self.max = max
    self.currentVal = self.min
    self.redraw()

  def tick( self ):
    self.currentVal += 1

    self.redraw()

  def start( self ):
    self.startTime = time.time()

  def stop( self ):
    self.canvas.itemconfig( self.lblPercent, text="0%" )
    self.canvas.itemconfig( self.lblETATime, text="--:--" )
    self.canvas.itemconfig( self.lblDuratin, text="--:--" )
    pass

  def redraw( self ):
    len = self.max - self.min
    if len == 0:
      # Avoid divide by 0 e.g. nothing to do
      return

    ratio = ( self.currentVal - self.min ) / len
    percent = int( ratio * 100 )
    fill_height = self.height * ratio

    # Update bar fill (grows upward)
    self.canvas.coords( self.fgBar, 0, self.height - fill_height, self.width, self.height )
    self.canvas.itemconfig( self.lblPercent, text=f"{percent}%" )

    if self.startTime is not None:

      # Calculate ETA
      elapsed = time.time() - self.startTime
      if self.currentVal > 0:
        timePerStep = elapsed / self.currentVal
        remSteps = self.max - self.currentVal
        etaInSeconds = int( timePerStep * remSteps )
        etaDisplay = time.strftime( "%M:%S", time.gmtime( etaInSeconds ) )
        self.canvas.itemconfig( self.lblETATime, text=f"{etaDisplay}" )
      else:
        self.canvas.itemconfig( self.lblETATime, text="--:--" )
      timeSpent = time.strftime( "%M:%S", time.gmtime( elapsed ) )
      self.canvas.itemconfig( self.lblDuratin, text=timeSpent )
