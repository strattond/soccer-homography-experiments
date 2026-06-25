import tkinter as tk


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
    self.slider = tk.Scale( root, from_=from_, to=to, orient='horizontal', command=self.internalCommand )
    self.slider.place( x=x, y=y, width=width, height=height )
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
    self._debounce_job = self.root.after( 200, lambda: self.on_debounce( value ) )

  def on_debounce( self, value ):
    if self.command is not None:
      self.command( value )
