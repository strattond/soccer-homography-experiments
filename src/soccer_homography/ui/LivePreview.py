import tkinter as tk
from collections.abc import Callable
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
    self.on_track_select: Callable[ [ int ], None ] | None = None
    self.hoveredTrackID: int | None = None
    self.pointerPosition: tuple[ int, int ] | None = None

    # Build layers
    self.createLayers()
    self.drawPitch()
    self.heatmapSelection.bind( "<<ComboboxSelected>>", lambda _event: self.refreshHeatmap() )
    self.canvas.bind( "<Motion>", self.onCanvasMotion )
    self.canvas.bind( "<Leave>", self.onCanvasLeave )
    self.canvas.bind( "<Button-1>", self.onCanvasClick )

  def drawPitch( self ):
    self.canvas.create_image( 0, 0, anchor="nw", image=self.pitch_photo, tags=( "pitch",) )

  def draw( self, homography: Point2D, color: str, scaleW, scaleL, track_id: int ):
    mx = int( homography.x * scaleW + self.pitch.padding )
    my = int( homography.y * scaleL + self.pitch.padding )
    radius = 6
    self.canvas.create_oval(
        mx - radius,
        my - radius,
        mx + radius,
        my + radius,
        fill=color,
        outline="black",
        width=1,
        tags=( "mapping", "homography_marker", f"track:{track_id}" ),
    )

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
    self.hoveredTrackID = None
    scaleW, scaleL = self.pitch.get_pitch_scale
    for track in tracks.values():
      lkpIndex = track.getListIndex( frame_index )
      if lkpIndex is not None and lkpIndex < len( track.homog ) and lkpIndex < len( track.homog_smooth ):
        color = self.role_color( track.role )
        self.draw( track.homog[ lkpIndex ], color, scaleW, scaleL, track.id )
        self.draw( track.homog_smooth[ lkpIndex ], color, scaleW, scaleL, track.id )
    if self.pointerPosition is not None:
      self.updateHoveredTrack( *self.pointerPosition )

  def setTrackSelectCallback( self, callback: Callable[ [ int ], None ] ) -> None:
    self.on_track_select = callback

  def markerTrackAt( self, x: int, y: int ) -> int | None:
    markers = self.canvas.find_overlapping( x - 8, y - 8, x + 8, y + 8 )
    closest_marker: int | None = None
    closest_distance = 8**2
    for marker in markers:
      tags = self.canvas.gettags( marker )
      track_tag = next( ( tag for tag in tags if tag.startswith( "track:" ) ), None )
      if track_tag is None or "homography_marker" not in tags:
        continue
      left, top, right, bottom = self.canvas.coords( marker )
      distance = ( ( left + right ) / 2 - x ) ** 2 + ( ( top + bottom ) / 2 - y ) ** 2
      if distance <= closest_distance:
        closest_distance = distance
        closest_marker = int( track_tag.split( ":", 1 )[ 1 ] )
    return closest_marker

  def updateHoveredTrack( self, x: int, y: int ) -> None:
    self.pointerPosition = ( x, y )
    track_id = self.markerTrackAt( x, y )
    if track_id == self.hoveredTrackID:
      return
    if self.hoveredTrackID is not None:
      self.canvas.itemconfigure( f"track:{self.hoveredTrackID}", outline="black", width=1 )
    self.hoveredTrackID = track_id
    if track_id is not None:
      self.canvas.itemconfigure( f"track:{track_id}", outline="#ffff00", width=3 )

  def onCanvasMotion( self, event ) -> None:
    self.updateHoveredTrack( event.x, event.y )

  def onCanvasLeave( self, _event=None ) -> None:
    if self.hoveredTrackID is not None:
      self.canvas.itemconfigure( f"track:{self.hoveredTrackID}", outline="black", width=1 )
      self.hoveredTrackID = None
    self.pointerPosition = None

  def onCanvasClick( self, event ) -> None:
    track_id = self.markerTrackAt( event.x, event.y )
    if track_id is not None and self.on_track_select is not None:
      self.on_track_select( track_id )

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