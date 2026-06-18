import tkinter as tk

from appState import AppState

class LivePreview:
  """
    Handles:
      - pitch image layer
      - keypoint layer
      - selection layer
      - hover layer
      - hit‑testing for nearest point
      - redraw of only changed items
    """

  def __init__(
      self,
      canvas: tk.Canvas,
      pitch_photo,
      state: AppState
  ):

    self.canvas: tk.Canvas = canvas
    self.pitch_photo = pitch_photo
    self.state: AppState = state

    # Build layers
    self.createLayers()
    self.drawPitch()

  def drawPitch( self ):
    self.canvas.create_image( 0, 0, anchor="nw", image=self.pitch_photo, tags=( "pitch",) )

  # -------------------------------------------------------------
  # Layer setup
  # -------------------------------------------------------------
  def createLayers( self ):
    # These tags define your layer stack
    self.canvas.addtag_withtag( "pitch", "pitch" )
    self.canvas.addtag_withtag( "mapping", "mapping" )
