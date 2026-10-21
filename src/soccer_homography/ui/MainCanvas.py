import tkinter as tk
from itertools import pairwise

import cv2
import numpy as np
from PIL import Image, ImageTk
from supervision import Color

from soccer_homography.appState import AppState
from soccer_homography.data import (
    BoundingBox,
    Point2D,
    SelectionPoint,
    Track,
    VideoData,
    ViewTransform,
)
from soccer_homography.LineDetector import LineDetector
from soccer_homography.log import logger
from soccer_homography.pitch import SoccerPitchConfiguration, SoccerPitchImage
from soccer_homography.ui.components import TrackOverlay


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

  def __init__( self, canvas: tk.Canvas, appState: AppState, on_click=None, on_hover=None, on_view_change=None, on_selection_move=None, on_selection_delete=None ):

    self.canvas: tk.Canvas = canvas
    self.appState: AppState = appState
    self.cap: cv2.VideoCapture | None = None
    self.cfg: SoccerPitchConfiguration = appState.cfg
    self.pitch: SoccerPitchImage = appState.pitch
    self.selected: list[ SelectionPoint ] = []
    self.mapping: SelectionPoint | None = None
    self.trackOverlay: TrackOverlay = TrackOverlay( "Track Data", 320, 120, ( 127, 255, 127 ), 0.4, None, -1 )

    # Callbacks
    self.on_click = on_click
    self.on_hover = on_hover
    self.on_view_change = on_view_change
    self.on_selection_move = on_selection_move
    self.on_selection_delete = on_selection_delete

    # State
    self.transform = ViewTransform()
    self.initial_scale = self.transform.scale
    self.initial_offset = Point2D()
    self.drag_start: tuple[ int | None, int | None ] = ( None, None )
    self.dragged_selection: SelectionPoint | None = None
    self.selection_items: dict[ int, SelectionPoint ] = {}

    # Canvas items
    self.createLayers()
    self.pil_image = Image.new( mode="RGB", size=( 1280, 720 ), color=( 255, 255, 255 ) )
    self.rsz_image = self.pil_image.copy()
    self.frame_photo = ImageTk.PhotoImage( self.rsz_image )
    self.frame_item = self.canvas.create_image( 0, 0, anchor="nw", image=self.frame_photo, tags=( "frame",) )
    self.frame_num = 0
    self.mapping_item = self.createSingleMarker( self.pitch.colors.sel_color, "mapping" )

    # Bind events
    self.canvas.bind( "<ButtonPress-1>", self.handleMousePress )
    self.canvas.bind( "<B1-Motion>", self.handleMouseDrag )
    self.canvas.bind( "<ButtonRelease-1>", self.handleMouseRelease )
    self.canvas.bind( "<Motion>", self.handleHover )
    self.canvas.bind( "<ButtonPress-3>", self.handleRightClick )
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
    self.canvas.addtag_withtag( "boxes", "boxes" )
    self.canvas.addtag_withtag( "tracking", "tracking" )
    self.canvas.addtag_withtag( "selection", "selection" )
    self.canvas.addtag_withtag( "mapping", "mapping" )
    self.canvas.addtag_withtag( "projfield", "projfield" )
    self.canvas.addtag_withtag( "trackoverlay", "trackoverlay" )

  def stratify( self ):
    self.canvas.tag_lower( "frame" )
    self.canvas.tag_raise( "hough" )
    self.canvas.tag_raise( "boxes" )
    self.canvas.tag_raise( "tracking" )
    self.canvas.tag_raise( "selection" )
    self.canvas.tag_raise( "mapping" )
    self.canvas.tag_raise( "projfield" )
    self.canvas.tag_raise( "trackoverlay" )

  # -------------------------------------------------------------
  # Load a video frame (PIL Image)
  # -------------------------------------------------------------
  def setFrame( self, frame_index: int, trackID: int | None = None ):
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
    self.updateBoundingBoxes( self.appState.boxes, frame_index )
    self.updateTracks( self.appState.tracks, frame_index, trackID )
    self.updateOverlay( frame_index, trackID )
    self.stratify()

  def setResizedImage( self ):
    self.rsz_image = self.pil_image.resize( self.transform.scaledDimensions(), Image.Resampling.LANCZOS )
    self.frame_photo = ImageTk.PhotoImage( self.rsz_image )
    self.canvas.itemconfig( self.frame_item, image=self.frame_photo )

  def updateOverlay( self, frame_index, trackID ):
    self.canvas.delete( 'trackoverlay' )
    if trackID is None:
      return
    self.trackOverlay.data = self.appState.tracks[trackID]
    self.trackOverlay.frameIndex = frame_index
    image = self.trackOverlay.render()
    self.trackOverlayImage = ImageTk.PhotoImage(image)
    self.canvas.create_image( 940, 20, anchor="nw", image=self.trackOverlayImage, tags=( 'trackoverlay', ) )    

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
    self.canvas.create_image( self.transform.offset.x, self.transform.offset.y, anchor="nw", image=self.nosky, tags=( "hough",) )
    if lines is None:
      self.stratify()
      return
    for ( x1, y1, x2, y2 ) in lines[ :, 0 ]:
      display_x1 = x1 + self.transform.offset.x
      display_y1 = y1 + self.transform.offset.y
      display_x2 = x2 + self.transform.offset.x
      display_y2 = y2 + self.transform.offset.y
      self.canvas.create_line( display_x1, display_y1, display_x2, display_y2, fill="cyan", width=2, tags=( "hough",) )
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

  def handleMousePress( self, event ):
    if self.cap is None:
      return
    overlapping = self.canvas.find_overlapping( event.x - 8, event.y - 8, event.x + 8, event.y + 8 )
    for item_id in reversed( overlapping ):
      if item_id in self.selection_items:
        self.dragged_selection = self.selection_items[ item_id ]
        return "break"
    self.handleClick( event )

  def handleMouseDrag( self, event ):
    if self.dragged_selection is None:
      return
    ix, iy = self.transform.toImage( event.x, event.y )
    self.dragged_selection.coords = Point2D( int( ix ), int( iy ) )
    self.updateSelectionMarkers( self.selected )
    if self.on_selection_move is not None:
      self.on_selection_move()
    return "break"

  def handleMouseRelease( self, _event ):
    self.dragged_selection = None

  def handleRightClick( self, event ):
    overlapping = self.canvas.find_overlapping( event.x - 8, event.y - 8, event.x + 8, event.y + 8 )
    for item_id in reversed( overlapping ):
      point = self.selection_items.get( item_id )
      if point is not None:
        if self.on_selection_delete is not None:
          self.on_selection_delete( point )
        return "break"
    self.startPan( event )

  def clearPendingMapping( self ):
    self.mapping = None
    self.canvas.itemconfig( self.mapping_item, state="hidden" )

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
    self.updateMappingMarker()
    if self.on_view_change is not None:
      self.on_view_change()
    self.drawFieldLines()

  def resetView( self ):
    self.transform.scale = self.initial_scale
    self.transform.offset = Point2D( self.initial_offset.x, self.initial_offset.y )
    self.applyTransform()

  def resetZoom( self ):
    self.transform.scale = self.initial_scale
    self.applyTransform()

  def resetTranslation( self ):
    self.transform.offset = Point2D( self.initial_offset.x, self.initial_offset.y )
    self.applyTransform()

  def load( self, cap: cv2.VideoCapture, vidData: VideoData ):
    self.cap = cap
    self.transform = ViewTransform( vidData )
    self.transform.scale = 1280 / vidData.width
    self.initial_scale = self.transform.scale
    self.initial_offset = Point2D()

  # -------------------------------------------------------------
  # Mapped selection markers
  # -------------------------------------------------------------
  def updateSelectionMarkers( self, selected: list[ SelectionPoint ] ):
    self.canvas.delete( "selection" )
    self.selected = selected
    self.selection_items.clear()
    color = self.pitch.colors.highlight_color.as_hex()
    for point in selected:
      mx, my = self.transform.toDisplay( point.coords.x, point.coords.y )

      radius = 6
      item = self.canvas.create_oval( mx - radius, my - radius, mx + radius, my + radius, fill=color, outline="black", width=1, tags=( "selection",) )
      self.selection_items[ item ] = point

  def updateMappingMarker( self ):
    if self.mapping is None:
      return
    mx, my = self.transform.toDisplay( self.mapping.coords.x, self.mapping.coords.y )
    self.canvas.coords( self.mapping_item, mx - 6, my - 6, mx + 6, my + 6 )

  def refreshHough( self ):
    self.applyHoughTransform()

  def updateBoundingBoxes( self, boxes: dict[ int, list[ BoundingBox ] ], index: int ):
    self.canvas.delete( "boxes" )
    boxData = boxes.get( index, [] )
    for i, box in enumerate( boxData ):
      if box is not None:
        # Now we need to scale the box coordinates to our image
        tlx, tly = self.transform.toDisplay( box.x1, box.y1 )
        brx, bry = self.transform.toDisplay( box.x2, box.y2 )
        # Put in a slightly smaller bounding box so we can see it compared to the track box
        self.canvas.create_rectangle( tlx + 2, tly + 2, brx - 2, bry - 2, outline='yellow', tags=( "boxes",) )

  def getTrackColor( self, trackIDSelected: int | None, trackID: int ) -> str:
    if trackIDSelected is not None and trackIDSelected == trackID:
      return 'red'
    else:
      return 'lightgreen'

  def updateTracks( self, tracks: dict[ int, Track ], frameIndex: int, trackID: int | None = None ):
    self.canvas.delete( "tracking" )
    for k, v in tracks.items():
      box = v.getByIndex( frameIndex )
      if box is not None:
        # Now we need to scale the box coordinates to our image
        tlx, tly = self.transform.toDisplay( box.x1, box.y1 )
        brx, bry = self.transform.toDisplay( box.x2, box.y2 )
        trackColor = self.getTrackColor( trackID, k )
        self.canvas.create_rectangle( tlx, tly, brx, bry, outline=trackColor, tags=( "tracking",) )
        self.canvas.create_text( tlx, tly, text="Track " + str( k ), tags=( "tracking",), fill=trackColor, font=( 'Arial', 20 ) )

  def invert( self, invH, point ):
      imagePoint = invH @ np.array( [ point[ 0 ], point[ 1 ], 1.0 ] )
      scale = np.linalg.norm( invH[ 2 ] ) * np.linalg.norm( [ point[ 0 ], point[ 1 ], 1.0 ] )
      if not np.isfinite( imagePoint ).all() or abs( imagePoint[ 2 ] ) <= np.finfo( float ).eps * scale * 16:
        return None
      imagePoint /= imagePoint[ 2 ]
      return imagePoint[ 0 ], imagePoint[ 1 ]

  def clipToImage( self, start, end ):
    width = self.transform.dimensions.x
    height = self.transform.dimensions.y
    if width <= 0 or height <= 0:
      return None

    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    lower = 0.0
    upper = 1.0
    for p, q in ( ( -dx, x1 ), ( dx, width - 1 - x1 ), ( -dy, y1 ), ( dy, height - 1 - y1 ) ):
      if p == 0:
        if q < 0:
          return None
        continue
      ratio = q / p
      if p < 0:
        lower = max( lower, ratio )
      else:
        upper = min( upper, ratio )
      if lower > upper:
        return None

    clipped = (
        ( x1 + lower * dx, y1 + lower * dy ),
        ( x1 + upper * dx, y1 + upper * dy ),
    )
    return tuple(
        ( min( max( x, 0.0 ), width - 1 ), min( max( y, 0.0 ), height - 1 ) )
        for x, y in clipped
    )

  def drawProjectedSegment( self, invH, start, end ):
    # A line crossing the projective horizon has no finite segment between its endpoints.
    start_denominator = invH[ 2 ] @ np.array( [ start[ 0 ], start[ 1 ], 1.0 ] )
    end_denominator = invH[ 2 ] @ np.array( [ end[ 0 ], end[ 1 ], 1.0 ] )
    if start_denominator * end_denominator <= 0:
      return

    image_start = self.invert( invH, start )
    image_end = self.invert( invH, end )
    if image_start is None or image_end is None:
      return
    clipped = self.clipToImage( image_start, image_end )
    if clipped is None:
      return
    display_start = self.transform.toDisplay( *clipped[ 0 ] )
    display_end = self.transform.toDisplay( *clipped[ 1 ] )
    self.canvas.create_line( *display_start, *display_end, fill="cyan", width=2, tags=( "projfield",) )

  def drawFieldLines( self ):
    self.canvas.delete( "projfield" )
    if not self.appState.imgOpts.showProjectedField or self.appState.data is None or self.appState.data.hom4k is None:
      return

    invH = np.linalg.inv( self.appState.data.hom4k )

    for start, end in self.appState.cfg.edges:
      vStart = self.appState.cfg.vertices[ start - 1 ]
      vEnd = self.appState.cfg.vertices[ end - 1 ]
      self.drawProjectedSegment( invH, vStart, vEnd )

    center = ( self.appState.cfg.length / 2, self.appState.cfg.width / 2 )
    radius = self.appState.cfg.centre_circle_radius
    if radius <= 0:
      return
    circle_points = [
        ( center[ 0 ] + radius * np.cos( angle ), center[ 1 ] + radius * np.sin( angle ) )
        for angle in np.linspace( 0, 2 * np.pi, 97 )
    ]
    for start, end in pairwise( circle_points ):
      self.drawProjectedSegment( invH, start, end )
