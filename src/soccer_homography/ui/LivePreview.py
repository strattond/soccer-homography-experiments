import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk

from soccer_homography.appState import AppState
from soccer_homography.data import ParticipationRole, Point2D, Track, roles
from soccer_homography.encoder import BaseVideoEncoder
from soccer_homography.pitch import SoccerPitchImage

# Marker colours keyed by participation role
# yapf: disable
role_colors: dict[ ParticipationRole, str ] = {
  "home_player":        "#ff0000",   # Red for Home Player
  "away_player":        "#0000ff",   # Blue for Away Player
  "home_goalkeeper":    "#00ff00",   # Bright Green for home goalkeeper
  "away_goalkeeper":    "#66ccff",   # Light Blue for away goalkeeper
  "unknown":            "#000000",   # Black for unknown
  "referee":            "#ffffff",   # White for referee
}
# yapf: enable


class LivePreview:
  # yapf: disable
  root:        tk.Tk
  canvas:      tk.Canvas
  pitch_photo: ImageTk.PhotoImage
  state:       AppState
  pitch:       SoccerPitchImage
  preserved:   list[ Image.Image ]
  dimensions:  Point2D = Point2D( 420, 272 )
  offsets:     Point2D = Point2D( 20, 20 )
  heatmaps:    dict[ParticipationRole, ImageTk.PhotoImage | None] = {}
  preserve:    bool                = False
  # yapf: enable

  def __init__( self, root: tk.Tk, coords: tuple[int,int], pitch_photo: ImageTk.PhotoImage, state: AppState, bumpFunc ):

    self.root = root
    # livePreview
    # 1460,400
    self.canvas = tk.Canvas( self.root, bg="#bfbfbf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.canvas.place( x=coords[0], y=coords[1], width=self.dimensions.x + self.offsets.x, height=self.dimensions.y + self.offsets.y )

    # lblLivePreview
    self.lblLivePreview = tk.Label( self.root, text="Live Preview", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblLivePreview.place( x=coords[0], y=coords[1] - 50, width=100, height=24 )

    self.heatmapSelection = ttk.Combobox(
        self.root,
        state="readonly",
        values=[ f"{role.replace( "_", " " )}" for role in roles ],
        width=42,
    )
    self.heatmapSelection.place( x=coords[0] + 100, y=coords[1] - 50, width=320, height=24 )
    for role in roles:
      self.heatmaps[role] = None

    self.preserved = []
    self.pitch_photo = pitch_photo
    self.state = state
    self.pitch = state.pitch
    self.bump = bumpFunc

    # Build layers
    self.createLayers()
    self.drawPitch()
    self.heatmapSelection.bind( "<<ComboboxSelected>>", lambda _event: self.refreshHeatmap() )

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
    self.canvas.addtag_withtag( "heatmap", "heatmap" )

  def updateMappings( self, tracks: dict[ int, Track ], frame_index: int ):
    self.canvas.delete( "mapping" )
    scaleW, scaleL = self.pitch.get_pitch_scale
    for track in tracks.values():
      lkpIndex = track.getListIndex( frame_index )
      if lkpIndex is not None and lkpIndex < len( track.homog ) and lkpIndex < len( track.homog_smooth ):
        color = self.role_color( track.role )
        self.draw( track.homog[ lkpIndex ], color, scaleW, scaleL )
        self.draw( track.homog_smooth[ lkpIndex ], color, scaleW, scaleL )

  def role_color( self, role: ParticipationRole ) -> str:
    return role_colors.get( role, "#000000" )

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
      if self.bump is not None:
        self.bump()
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

  def refreshHeatmap( self ):
    self.canvas.delete( "heatmap" )
    role = self.selected_role()
    if self.heatmaps[role] is not None:
      self.canvas.create_image( self.offsets.x / 2, self.offsets.y / 2, anchor="nw", image=self.heatmaps[role], tags=( "heatmap",) )

  def selected_role(self) -> ParticipationRole:
    return roles[self.heatmapSelection.current()]

  def updateHeatmap( self, role: ParticipationRole, heatmap_image: np.ndarray ):
    # Before we turn it into a photoimage ... let's resize it first
    new_image = cv2.resize( heatmap_image, ( int(self.dimensions.x), int(self.dimensions.y) ), interpolation=cv2.INTER_CUBIC )
    self.heatmaps[role] = ImageTk.PhotoImage( Image.fromarray( new_image ) )
    self.refreshHeatmap()