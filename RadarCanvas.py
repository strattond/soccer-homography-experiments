import tkinter as tk
from dataclasses import dataclass
from typing import List
from PIL import Image, ImageTk
from supervision import Color

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

  def __init__(
      self,
      canvas: tk.Canvas,
      pitch_photo,
      cfg: SoccerPitchConfiguration,
      pitch: SoccerPitchImage,
      on_click=None,
      on_hover=None
  ):

    self.canvas: tk.Canvas = canvas
    self.pitch_photo = pitch_photo
    self.cfg: SoccerPitchConfiguration = cfg
    self.pitch: SoccerPitchImage = pitch
    self.selected: List[ SelectionPoint ] = []
    self.mapping: SelectionPoint = None

    # Callbacks
    self.on_click = on_click
    self.on_hover = on_hover

    # Build layers
    self.createLayers()
    self.drawPitch()
    self.drawKeypoints()
    self.hover_item = self.createSingleMarker( self.pitch.colors.hover_color, "hover" )
    self.mapping_item = self.createSingleMarker( self.pitch.colors.sel_color, "mapping" )

    # Bind events
    self.canvas.bind( "<Motion>", self.handle_hover )
    self.canvas.bind( "<Button-1>", self.handle_click )

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
    self.canvas.addtag_withtag( "mapping", "mapping" )
    self.canvas.addtag_withtag( "hover", "hover" )

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
  # Single marker on nominated layer
  # -------------------------------------------------------------
  def createSingleMarker( self, color: Color, tag: str = "hover" ) -> int:
    item = self.canvas.create_oval( 0, 0, 0, 0, fill=color.as_hex(), outline="black", width=2, tags=( tag,) )
    self.canvas.itemconfig( item, state="hidden" )
    return item

  # -------------------------------------------------------------
  # Event handlers
  # -------------------------------------------------------------
  def handle_hover( self, event ):

    rx, ry = event.x, event.y
    nearest = self.pitch.nearest_field_point( rx, ry )
    x, y = self.pitch.calc_point_offset( nearest )
    self.canvas.coords( self.hover_item, x - 6, y - 6, x + 6, y + 6 )
    self.canvas.itemconfig( self.hover_item, state="normal" )
    if self.on_hover:
      self.on_hover( rx, ry )

  def handle_click( self, event ):

    rx, ry = event.x, event.y
    self.mapping = self.pitch.nearest_field_point( rx, ry )
    x, y = self.pitch.calc_point_offset( self.mapping )
    self.canvas.coords( self.mapping_item, x - 6, y - 6, x + 6, y + 6 )
    self.canvas.itemconfig( self.mapping_item, state="normal" )
    if self.on_click:
      self.on_click( rx, ry, self.mapping )

  # -------------------------------------------------------------
  # Mapped selection markers
  # -------------------------------------------------------------
  def update_selection_markers( self, selected: List[ SelectionPoint ] ):
    self.canvas.delete( "selection" )
    self.selected = selected
    color = self.pitch.colors.highlight_color.as_hex()
    for vertex in self.selected:
      mx, my = self.pitch.calc_point_offset( vertex )

      radius = 6
      self.canvas.create_oval(
          mx - radius, my - radius, mx + radius, my + radius, fill=color, outline="black", width=1, tags=( "selection" )
      )
