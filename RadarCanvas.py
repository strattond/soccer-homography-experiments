import tkinter as tk
from dataclasses import dataclass
from typing import List
from PIL import Image, ImageTk

from dataTypes import SelectionPoint
from pitch import SoccerPitchConfiguration, SoccerPitchImage


class RadarCanvas:
  """
    Handles:
      - pitch image layer
      - keypoint layer
      - selection layer
      - hover layer
      - hit‑testing for nearest point
      - redraw of only changed items
    """

  def __init__( self, canvas: tk.Canvas, pitch_photo, cfg: SoccerPitchConfiguration, pitch: SoccerPitchImage ):

    self.canvas: tk.Canvas = canvas
    self.pitch_photo = pitch_photo
    self.cfg: SoccerPitchConfiguration = cfg
    self.pitch: SoccerPitchImage = pitch
    self.selected: SelectionPoint = None

    # Build layers
    self.createLayers()
    self.drawPitch()
    self.drawKeypoints()
    self.createHoverMarker()

    # Bind events
    self.canvas.bind( "<Motion>", self.on_hover )
    self.canvas.bind( "<Button-1>", self.on_click )

  def drawPitch( self ):
    self.canvas.create_image( 0, 0, anchor="nw", image=self.pitch_photo, tags=( "pitch",) )

  # -------------------------------------------------------------
  # Layer setup
  # -------------------------------------------------------------
  def createLayers( self ):
    # These tags define your layer stack
    self.canvas.addtag_withtag( "pitch", "pitch" )
    self.canvas.addtag_withtag( "keypoints", "keypoints" )
    self.canvas.addtag_withtag( "selection", "selection" )
    self.canvas.addtag_withtag( "hover", "hover" )

  # -------------------------------------------------------------
  # Draw pitch image (bottom layer)
  # -------------------------------------------------------------
  def _draw_pitch( self ):
    self.pitch_item = self.canvas.create_image( 0, 0, anchor="nw", image=self.pitch_photo, tags=( "pitch",) )
    self.canvas.tag_lower( "pitch" )

  # -------------------------------------------------------------
  # Draw keypoints (static layer)
  # -------------------------------------------------------------
  def drawKeypoints( self ):
    scaleW, scaleL = self.pitch.get_pitch_scale
    i = 0
    for vertex, pt in zip( self.cfg.vertices, self.cfg.labels ):
      mx = int( vertex[ 0 ] * scaleW + self.pitch.padding )
      my = int( vertex[ 1 ] * scaleL + self.pitch.padding )

      radius = 6
      self.canvas.create_oval(
          mx - radius,
          my - radius,
          mx + radius,
          my + radius,
          fill=self.pitch.colors.point_color.as_hex(),
          outline="black",
          width=1,
          tags=( "keypoints" )
      )
      self.canvas.create_text( mx + 10, my - 10, text=pt, fill="white", font=( "Arial", 12 ), tags=( "keypoints" ) )
      i += 1

  # -------------------------------------------------------------
  # Hover marker (top layer)
  # -------------------------------------------------------------
  def createHoverMarker( self ):
    self.hover_item = self.canvas.create_oval(
        0, 0, 0, 0, fill=self.pitch.colors.hover_color.as_hex(), outline="black", width=2, tags=( "hover",)
    )
    self.canvas.itemconfig( self.hover_item, state="hidden" )

  # -------------------------------------------------------------
  # Event handlers
  # -------------------------------------------------------------
  def on_hover( self, event ):

    rx, ry = event.x, event.y
    nearest = self.pitch.nearest_field_point( rx, ry )
    x, y = self.pitch.calc_point_offset( nearest )
    self.canvas.coords( self.hover_item, x - 6, y - 6, x + 6, y + 6 )
    self.canvas.itemconfig( self.hover_item, state="normal" )

  def on_click( self, event ):

    rx, ry = event.x, event.y
    self.selected = self.pitch.nearest_field_point( rx, ry )

  # -------------------------------------------------------------
  # Selection markers
  # -------------------------------------------------------------
  def _draw_selection_marker( self, label, mx, my ):
    sid = self.canvas.create_oval( mx - 8, my - 8, mx + 8, my + 8, outline="cyan", width=2, tags=( "selection", label ) )
    self.selection_items[ label ] = sid
    self.canvas.tag_raise( "selection", "keypoints" )

  def _remove_selection_marker( self, label ):
    if label in self.selection_items:
      self.canvas.delete( self.selection_items[ label ] )
      del self.selection_items[ label ]
