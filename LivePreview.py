import tkinter as tk
from PIL import Image, ImageTk, ImageGrab

from appState import AppState
from dataTypes import Point2D, Track
from encoder import BaseVideoEncoder
from pitch import SoccerPitchImage


class LivePreview:
  # yapf: disable
  canvas:      tk.Canvas
  pitch_photo: ImageTk.PhotoImage
  state:       AppState
  pitch:       SoccerPitchImage
  preserved:   list[ Image.Image ] = []
  preserve:    bool                = False
  # yapf: enable

  def __init__( self, canvas: tk.Canvas, pitch_photo: ImageTk.PhotoImage, state: AppState, bumpFunc ):

    self.canvas = canvas
    self.pitch_photo = pitch_photo
    self.state = state
    self.pitch = state.pitch
    self.bump = bumpFunc

    # Build layers
    self.createLayers()
    self.drawPitch()

  def drawPitch( self ):
    self.canvas.create_image( 0, 0, anchor="nw", image=self.pitch_photo, tags=( "pitch",) )

  def draw( self, homography: Point2D, color: str, scaleW, scaleL ):
    mx = int( homography.x * scaleW + self.pitch.padding )
    my = int( homography.y * scaleL + self.pitch.padding )
    radius = 6
    self.canvas.create_oval( mx - radius, my - radius, mx + radius, my + radius, fill=color, outline="black", width=1, tags=( "mapping" ) )

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
        self.draw( tracks[ track ].homog[ lkpIndex ], self.pitch.colors.point_color.as_hex(), scaleW, scaleL )
        self.draw( tracks[ track ].homog_smooth[ lkpIndex ], self.pitch.colors.hover_color.as_hex(), scaleW, scaleL )

  def play( self, min: int, max: int, encoder: BaseVideoEncoder | None ):
    self.preserved = []
    self.preserve = True
    self.homogIdx = min
    self.homogMax = max
    self.encoder = encoder
    self.canvas.after( 50, self.homographyPlayLoop )

  def homographyPlayLoop( self ):
    self.updateMappings( self.state.tracks, self.homogIdx )
    self.homogIdx += 1
    if self.homogIdx >= self.homogMax:
      if self.preserve and self.state.cap is not None and self.encoder is not None:
        self.preserve = False
        self.encoder.save( self.state.cap, self.preserved )
        self.preserved = []
      return
    if self.preserve:
      self.preserved.append( self.saveFrame( self.canvas ) )
    if self.bump is not None:
      self.bump()
    self.canvas.after( 50, self.homographyPlayLoop )

  def saveFrame( self, canvas ):
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    w = x + canvas.winfo_width()
    h = y + canvas.winfo_height()
    return ImageGrab.grab( bbox=( x, y, w, h ) )
