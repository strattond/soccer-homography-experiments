import cv2
import numpy as np
import tkinter as tk
from log import logger
from PIL import Image, ImageTk
from supervision import Color

from LineDetector import LineDetector
from appState import AppState
from dataTypes import Point2D, SelectionPoint, Track, VideoData, ViewTransform
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

  def __init__( self, canvas: tk.Canvas, appState: AppState, on_click=None, on_hover=None, on_view_change=None ):

    self.canvas = canvas
    self.appState = appState
    self.cap: cv2.VideoCapture | None = None
    self.cfg: SoccerPitchConfiguration = appState.cfg
    self.pitch: SoccerPitchImage = appState.pitch
    self.selected: list[ SelectionPoint ] = []
    self.mapping: SelectionPoint | None = None

    # Callbacks
    self.on_click = on_click
    self.on_hover = on_hover
    self.on_view_change = on_view_change

    # State
    self.transform = ViewTransform()
    self.drag_start: tuple[ int | None, int | None ] = ( None, None )

    # Canvas items
    self.createLayers()
    self.pil_image = Image.new( mode="RGB", size=( 1280, 720 ), color=( 255, 255, 255 ) )
    self.rsz_image = self.pil_image.copy()
    self.frame_photo = ImageTk.PhotoImage( self.rsz_image )
    self.frame_item = self.canvas.create_image( 0, 0, anchor="nw", image=self.frame_photo, tags=( "frame",) )
    self.frame_num = 0
    self.mapping_item = self.createSingleMarker( self.pitch.colors.sel_color, "mapping" )

    # Bind events
    self.canvas.bind( "<Button-1>", self.handleClick )
    self.canvas.bind( "<Motion>", self.handleHover )
    self.canvas.bind( "<ButtonPress-3>", self.startPan )
    self.canvas.bind( "<B3-Motion>", self.doPan )

    # Mouse wheel (Windows/macOS)
    self.canvas.bind( "<MouseWheel>", self.doZoom )
    # Linux scroll
    self.canvas.bind( "<Button-4>", self.doZoom )
    self.canvas.bind( "<Button-5>", self.doZoom )

  def createLayers( self ):
    # These tags define your layer stack
    self.canvas.addtag_withtag( "frame", "frame" )
    self.canvas.addtag_withtag( "hough", "hough" )
    self.canvas.addtag_withtag( "tracking", "tracking" )
    self.canvas.addtag_withtag( "selection", "selection" )
    self.canvas.addtag_withtag( "mapping", "mapping" )

  def stratify( self ):
    self.canvas.tag_lower( "frame" )
    self.canvas.tag_raise( "hough" )
    self.canvas.tag_raise( "tracking" )
    self.canvas.tag_raise( "selection" )
    self.canvas.tag_raise( "mapping" )
    

  # -------------------------------------------------------------
  # Load a video frame (PIL Image)
  # -------------------------------------------------------------
  def setFrame( self, frame_index: int ):
    self.frame_num = frame_index
    if not self.cap:
      return
    self.cap.set( cv2.CAP_PROP_POS_FRAMES, frame_index )
    ret, raw_img = self.cap.read()
    if not ret:
      logger.error( f"Failed reading cap {ret}" )
      return

    self.pil_image = Image.fromarray( cv2.cvtColor( raw_img, cv2.COLOR_BGR2RGB ) )

    # Reset transforms when loading a new frame
    self.setResizedImage()
    self.applyTransform()

    self.applyHoughTransform()
    self.updateBoundingBoxes( self.appState.tracks, frame_index )

  def setResizedImage( self ):
    self.rsz_image = self.pil_image.resize( self.transform.scaledDimensions(), Image.Resampling.LANCZOS )
    self.frame_photo = ImageTk.PhotoImage( self.rsz_image )
    self.canvas.itemconfig( self.frame_item, image=self.frame_photo )

  def applyHoughTransform( self ):
    self.canvas.delete( 'hough' )
    if not self.appState.imgOpts.showHough:
      self.stratify()
      return

    # Grayscale it
    gray = cv2.cvtColor( np.array( self.rsz_image ), cv2.COLOR_BGR2GRAY )
    detector = LineDetector( self.appState.imgOpts )
    lines = detector.getLines( gray )

    self.nosky = ImageTk.PhotoImage( Image.fromarray( gray ) )
    self.canvas.create_image( 0, 0, anchor="nw", image=self.nosky, tags=( "hough",) )
    if lines is None:
      self.stratify()
      return
    for ( x1, y1, x2, y2 ) in lines[ :, 0 ]:
      self.canvas.create_line( x1, y1, x2, y2, fill="cyan", width=2, tags=( "hough",) )
    self.stratify()

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
  def handleClick( self, event ):
    if self.cap is None:
      return
    ix, iy = self.transform.toImage( event.x, event.y )

    self.mapping = SelectionPoint( None, Point2D( int( ix ), int( iy ) ) )
    self.canvas.coords( self.mapping_item, event.x - 6, event.y - 6, event.x + 6, event.y + 6 )
    self.canvas.itemconfig( self.mapping_item, state="normal" )

    if self.on_click:
      self.on_click( ix, iy, self.mapping )

  def handleHover( self, event ):
    if self.cap is None:
      return
    ix, iy = self.transform.toImage( event.x, event.y )
    if self.on_hover:
      self.on_hover( ix, iy )

  # -------------------------------------------------------------
  # Panning
  # -------------------------------------------------------------
  def startPan( self, event ):
    if self.cap is None:
      return
    self.drag_start = ( event.x, event.y )

  def doPan( self, event ):
    if self.cap is None:
      return
    dx = event.x - self.drag_start[ 0 ]
    dy = event.y - self.drag_start[ 1 ]
    self.transform.offset.x += dx
    self.transform.offset.y += dy
    self.drag_start = ( event.x, event.y )
    self.applyTransform()

  # -------------------------------------------------------------
  # Zooming
  # -------------------------------------------------------------
  def doZoom( self, event ):
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

    # Zoom around cursor
    cx, cy = event.x, event.y
    self.transform.offset.x = cx - ( cx - self.transform.offset.x ) * ( self.transform.scale / old_zoom )
    self.transform.offset.y = cy - ( cy - self.transform.offset.y ) * ( self.transform.scale / old_zoom )

    self.applyTransform()

  # -------------------------------------------------------------
  # Apply transform to canvas
  # -------------------------------------------------------------
  def applyTransform( self ):
    # Reset transform
    self.canvas.coords( self.frame_item, self.transform.offset.x, self.transform.offset.y )
    self.setResizedImage()
    if self.on_view_change is not None:
      self.on_view_change()

  def load( self, cap: cv2.VideoCapture, vidData: VideoData ):
    self.cap = cap
    self.transform = ViewTransform( vidData )
    self.transform.scale = 1280 / vidData.width

  # -------------------------------------------------------------
  # Mapped selection markers
  # -------------------------------------------------------------
  def updateSelectionMarkers( self, selected: list[ SelectionPoint ] ):
    self.canvas.delete( "selection" )
    self.selected = selected
    color = self.pitch.colors.highlight_color.as_hex()
    points = self.transform.getScaledPoints( selected )
    for vertex in points:
      mx, my = vertex.coords

      radius = 6
      self.canvas.create_oval( mx - radius, my - radius, mx + radius, my + radius, fill=color, outline="black", width=1, tags=( "selection",) )

  def refreshHough( self ):
    self.applyHoughTransform()

  def updateBoundingBoxes( self, tracks: dict[ int, Track ], index: int ):
    self.canvas.delete( "tracking" )
    for track in tracks:
      box = tracks[ track ].getByIndex( index )
      if box is not None:
        tlx, tly = self.transform.toDisplay( box.x1, box.y1 )
        brx, bry = self.transform.toDisplay( box.x2, box.y2 )
        #self.canvas.create_rectangle( box.x1, box.y1, box.x2, box.y2, outline='yellow', width=5, tags=( "tracking" ) )
        # Now we need to scale the box coordinates to our image
        self.canvas.create_rectangle( tlx, tly, brx, bry, outline='yellow', tags=( "tracking",) )
