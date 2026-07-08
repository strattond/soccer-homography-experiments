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
