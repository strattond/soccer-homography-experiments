import tkinter as tk
from typing import List
from PIL import Image, ImageTk
import cv2
from supervision import Color

from dataTypes import Point2D, SelectionPoint, VideoData, ViewTransform
from pitch import SoccerPitchConfiguration, SoccerPitchImage


class MainCanvasController:
  """
    Handles:
      - displaying a video frame (PIL Image)
      - zooming (mouse wheel)
      - panning (right-drag)
      - accurate canvas→image coordinate mapping
      - click callbacks
      - hover callbacks
    """

  def __init__( self, canvas: tk.Canvas, cfg: SoccerPitchConfiguration, pitch: SoccerPitchImage, on_click=None, on_hover=None ):

    self.canvas = canvas
    self.cap: cv2.VideoCapture = None
    self.cfg: SoccerPitchConfiguration = cfg
    self.pitch: SoccerPitchImage = pitch
    self.selected: List[SelectionPoint] = []
    self.mapping: SelectionPoint = None

    # Callbacks
    self.on_click = on_click
    self.on_hover = on_hover

    # State
    self.transform = ViewTransform()
    self.drag_start = None

    # Canvas items
    self.createLayers()
    self.pil_image = Image.new( mode="RGB", size=( 1280, 720 ), color=( 255, 255, 255 ) )
    self.rsz_image = self.pil_image.copy()
    self.frame_photo = ImageTk.PhotoImage( self.rsz_image )
    self.frame_item = self.canvas.create_image( 0, 0, anchor="nw", image=self.frame_photo, tags=( "frame",) )
    self.frame_num = 0
    self.mapping_item = self.createSingleMarker( self.pitch.colors.sel_color, "mapping" )

    # Bind events
    self.canvas.bind( "<Button-1>", self.handle_click )
    self.canvas.bind( "<Motion>", self.handle_hover )
    self.canvas.bind( "<ButtonPress-3>", self.start_pan )
    self.canvas.bind( "<B3-Motion>", self.do_pan )

    # Mouse wheel (Windows/macOS)
    self.canvas.bind( "<MouseWheel>", self.do_zoom )
    # Linux scroll
    self.canvas.bind( "<Button-4>", self.do_zoom )
    self.canvas.bind( "<Button-5>", self.do_zoom )

  def createLayers( self ):
    # These tags define your layer stack
    self.canvas.addtag_withtag( "frame", "frame" )
    self.canvas.addtag_withtag( "selection", "selection" )
    self.canvas.addtag_withtag( "mapping", "mapping" )
    
  # -------------------------------------------------------------
  # Load a video frame (PIL Image)
  # -------------------------------------------------------------
  def set_frame( self, frame_index: int ):
    print( f"Moving to {frame_index}" )
    self.frame_num = frame_index
    self.cap.set( cv2.CAP_PROP_POS_FRAMES, frame_index )
    ret, raw_img = self.cap.read()
    if not ret:
      print( f"Failed reading cap {ret}" )
      return

    self.pil_image = Image.fromarray( cv2.cvtColor( raw_img, cv2.COLOR_BGR2RGB ) )

    # Reset transforms when loading a new frame
    self.set_resized_image()
    self.apply_transform()

  def set_resized_image( self ):
    self.rsz_image = self.pil_image.resize( self.transform.scaledDimensions(), Image.Resampling.LANCZOS )
    self.frame_photo = ImageTk.PhotoImage( self.rsz_image )
    self.canvas.itemconfig( self.frame_item, image=self.frame_photo )
    
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
  def handle_click( self, event ):
    if self.cap is None:
      return
    ix, iy = self.transform.toImage( event.x, event.y )
    
    self.mapping = SelectionPoint( None, Point2D( ix, iy ) )
    self.canvas.coords( self.mapping_item, event.x - 6, event.y - 6, event.x + 6, event.y + 6 )
    self.canvas.itemconfig( self.mapping_item, state="normal" )
    
    if self.on_click:
      self.on_click( ix, iy, self.mapping )

  def handle_hover( self, event ):
    if self.cap is None:
      return
    ix, iy = self.transform.toImage( event.x, event.y )
    if self.on_hover:
      self.on_hover( ix, iy )

  # -------------------------------------------------------------
  # Panning
  # -------------------------------------------------------------
  def start_pan( self, event ):
    if self.cap is None:
      return
    self.drag_start = ( event.x, event.y )

  def do_pan( self, event ):
    if self.cap is None:
      return
    dx = event.x - self.drag_start[ 0 ]
    dy = event.y - self.drag_start[ 1 ]
    self.transform.offset.x += dx
    self.transform.offset.y += dy
    self.drag_start = ( event.x, event.y )
    self.apply_transform()

  # -------------------------------------------------------------
  # Zooming
  # -------------------------------------------------------------
  def do_zoom( self, event ):
    if self.cap is None:
      return
    old_zoom = self.transform.scale

    # Windows/macOS
    if hasattr( event, "delta" ) and event.delta != 0:
      factor = 1.1 if event.delta > 0 else 0.9
    else:
      # Linux
      factor = 1.1 if event.num == 4 else 0.9
      
    newValue = self.transform.scale * factor
    if newValue < 0.1:
      return
    
    if newValue > 10:
      return

    self.transform.scale = newValue

    print( f"Zooming from {old_zoom:.3f} to {self.transform.scale:.3f}" )

    # Zoom around cursor
    cx, cy = event.x, event.y
    self.transform.offset.x = cx - ( cx - self.transform.offset.x ) * ( self.transform.scale / old_zoom )
    self.transform.offset.x = cx - ( cx - self.transform.offset.x ) * ( self.transform.scale / old_zoom )
    self.transform.offset.y = cy - ( cy - self.transform.offset.y ) * ( self.transform.scale / old_zoom )

    self.apply_transform()

  # -------------------------------------------------------------
  # Apply transform to canvas
  # -------------------------------------------------------------
  def apply_transform( self ):
    # Reset transform
    print( f"Offset {self.transform.offset.x}, {self.transform.offset.y}" )
    self.canvas.coords( self.frame_item, self.transform.offset.x, self.transform.offset.y )
    print( f"Scale  {self.transform.scale:.3f}x{self.transform.scale:.3f}" )
    self.set_resized_image()
    #self.canvas.scale( "frame", 0, 0, self.transform.scale, self.transform.scale )

    # Ensure scroll region matches new bounds
    #self.canvas.configure( scrollregion=self.canvas.bbox( "all" ) )

  def load( self, cap: cv2.VideoCapture, vidData: VideoData ):
    self.cap = cap
    self.transform = ViewTransform( vidData )
    self.transform.scale = 1280 / vidData.width

  # -------------------------------------------------------------
  # Mapped selection markers
  # -------------------------------------------------------------
  def update_selection_markers( self, selected: List[ SelectionPoint ] ):
    self.canvas.delete( "selection" )
    self.selected = selected
    color = self.pitch.colors.highlight_color.as_hex()
    points = self.transform.getScaledPoints( selected )
    for vertex in points:
      mx, my = vertex.coords

      radius = 6
      self.canvas.create_oval(
          mx - radius, my - radius, mx + radius, my + radius, fill=color, outline="black", width=1, tags=( "selection" )
      )
