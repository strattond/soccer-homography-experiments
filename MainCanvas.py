import cv2
import numpy as np
import tkinter as tk
from cv2.typing import MatLike
from PIL import Image, ImageTk
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

  def __init__(
      self,
      canvas: tk.Canvas,
      cfg: SoccerPitchConfiguration,
      pitch: SoccerPitchImage,
      on_click=None,
      on_hover=None,
      on_view_change=None
  ):

    self.canvas = canvas
    self.cap: cv2.VideoCapture | None = None
    self.cfg: SoccerPitchConfiguration = cfg
    self.pitch: SoccerPitchImage = pitch
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
    self.canvas.addtag_withtag( "selection", "selection" )
    self.canvas.addtag_withtag( "mapping", "mapping" )

  # -------------------------------------------------------------
  # Load a video frame (PIL Image)
  # -------------------------------------------------------------
  def setFrame( self, frame_index: int ):
    print( f"Moving to {frame_index}" )
    self.frame_num = frame_index
    if not self.cap:
      print( "Capture not supplied" )
      return
    self.cap.set( cv2.CAP_PROP_POS_FRAMES, frame_index )
    ret, raw_img = self.cap.read()
    if not ret:
      print( f"Failed reading cap {ret}" )
      return

    self.pil_image = Image.fromarray( cv2.cvtColor( raw_img, cv2.COLOR_BGR2RGB ) )

    # Reset transforms when loading a new frame
    self.setResizedImage()
    self.applyTransform()

    #self.applyHoughTransform()

  def setResizedImage( self ):
    self.rsz_image = self.pil_image.resize( self.transform.scaledDimensions(), Image.Resampling.LANCZOS )
    self.frame_photo = ImageTk.PhotoImage( self.rsz_image )
    self.canvas.itemconfig( self.frame_item, image=self.frame_photo )

  def getSkyMask( self, image: MatLike ):
    _, binary = cv2.threshold( image, 0, 255, cv2.THRESH_OTSU )
    contours, _ = cv2.findContours( binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    if contours:
      sky_contour = max( contours, key=cv2.contourArea )
      sky_mask_clean = np.zeros_like( image )
      cv2.drawContours( sky_mask_clean, [ sky_contour ], -1, 255, thickness=cv2.FILLED )
      return sky_mask_clean
    else:
      return binary

  def blurIt( self, image: MatLike ):
    return cv2.bilateralFilter( image, d=9, sigmaColor=75, sigmaSpace=75 )

  def enhanceIt( self, image: MatLike ):
    clahe = cv2.createCLAHE( clipLimit=2.0, tileGridSize=( 8, 8 ) )
    return clahe.apply( image )

  def getHoughEdges( self, image: MatLike ):
    return cv2.Canny( image, 50, 150 )

  def getHoughLines( self, edges ):
    return cv2.HoughLinesP( edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10 )

  def getScharrEdges( self, image: MatLike ):
    grad_x = cv2.Scharr( image, cv2.CV_32F, 1, 0 )
    grad_y = cv2.Scharr( image, cv2.CV_32F, 0, 1 )
    mag = cv2.magnitude( grad_x, grad_y )
    mag = cv2.convertScaleAbs( mag )
    smooth = cv2.normalize( mag, mag, 0, 255, cv2.NORM_MINMAX )
    return cv2.adaptiveThreshold( smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -5 )

  def rebuildEdges( self, edges ):
    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 9, 9 ) )
    return cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )

  def drawOnMask( self, lines, image: MatLike ):
    mask = np.zeros_like( image, dtype=np.uint8 )
    for line in lines:
      x1, y1, x2, y2 = map( int, line[ 0 ] )
      cv2.line( mask, ( x1, y1 ), ( x2, y2 ), 255, 2 )
    return mask

  def applyHoughTransform( self ):
    lsd = cv2.createLineSegmentDetector( refine=cv2.LSD_REFINE_ADV )
    gray = cv2.cvtColor( np.array( self.rsz_image ), cv2.COLOR_BGR2GRAY )
    skyMask = self.getSkyMask( gray )
    img_no_sky = cv2.bitwise_and( gray, gray, mask=cv2.bitwise_not( skyMask ) )
    img_no_sky = self.blurIt( img_no_sky )
    img_no_sky = self.enhanceIt( img_no_sky )
    lines, _, _, _ = lsd.detect( img_no_sky )
    masked = self.drawOnMask( lines, img_no_sky )
    edges = self.rebuildEdges( masked )
    lines = self.getHoughLines( edges )

    self.canvas.delete( 'hough' )
    self.nosky = ImageTk.PhotoImage( Image.fromarray( img_no_sky ) )
    self.canvas.create_image( 0, 0, anchor="nw", image=self.nosky, tags=( "hough",) )
    if lines is None:
      return
    for ( x1, y1, x2, y2 ) in lines[ :, 0 ]:
      self.canvas.create_line( x1, y1, x2, y2, fill="cyan", width=2, tags=( "hough",) )

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

    print( f"Zooming from {old_zoom:.3f} to {self.transform.scale:.3f}" )

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
    print( f"Offset {self.transform.offset.x}, {self.transform.offset.y}" )
    self.canvas.coords( self.frame_item, self.transform.offset.x, self.transform.offset.y )
    print( f"Scale  {self.transform.scale:.3f}x{self.transform.scale:.3f}" )
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
      self.canvas.create_oval(
          mx - radius, my - radius, mx + radius, my + radius, fill=color, outline="black", width=1, tags=( "selection" )
      )
