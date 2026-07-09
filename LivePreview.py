import tkinter as tk
from PIL import ImageTk

from appState import AppState
from dataTypes import Track
from pitch import SoccerPitchImage


class LivePreview:
  # yapf: disable
  canvas:      tk.Canvas
  pitch_photo: ImageTk.PhotoImage
  state:       AppState
  pitch:       SoccerPitchImage
  # yapf: enable

  def __init__( self, canvas: tk.Canvas, pitch_photo: ImageTk.PhotoImage, state: AppState ):

    self.canvas = canvas
    self.pitch_photo = pitch_photo
    self.state = state
    self.pitch = state.pitch

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

  def updateMappings( self, tracks: dict[ int, Track ], frame_index: int ):
    self.canvas.delete( "mapping" )
    scaleW, scaleL = self.pitch.get_pitch_scale
    for track in tracks:
      lkpIndex = tracks[ track ].getListIndex( frame_index )
      if lkpIndex is not None:
        homography = tracks[ track ].homog[ lkpIndex ]
        mx = int( homography.x * scaleW + self.pitch.padding )
        my = int( homography.y * scaleL + self.pitch.padding )
        radius = 6
        self.canvas.create_oval( mx - radius, my - radius, mx + radius, my + radius, fill=self.pitch.colors.point_color.as_hex(), outline="black", width=1, tags=( "mapping" ) )
