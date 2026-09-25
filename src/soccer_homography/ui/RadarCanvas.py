import tkinter as tk

from supervision import Color

from soccer_homography.dataTypes import Point2D, SelectionPoint
from soccer_homography.pitch import SoccerPitchConfiguration, SoccerPitchImage


class RadarCanvas:

  FIELD_POINT_SNAP_DISTANCE = 12

  def __init__( self, canvas: tk.Canvas, pitch_photo, cfg: SoccerPitchConfiguration, pitch: SoccerPitchImage, on_click=None, on_hover=None, on_selection_move=None ):

    self.canvas: tk.Canvas = canvas
    self.pitch_photo = pitch_photo
    self.cfg: SoccerPitchConfiguration = cfg
    self.pitch: SoccerPitchImage = pitch
    self.selected: list[ SelectionPoint ] = []
    self.mapping: SelectionPoint | None = None

    # Callbacks
    self.on_click = on_click
    self.on_hover = on_hover
    self.on_selection_move = on_selection_move
    self.selection_items: dict[ int, tuple[ SelectionPoint, tuple[ tuple[ float, float ], tuple[ float, float ] ] | None ] ] = {}
    self.dragged_selection: SelectionPoint | None = None
    self.dragged_edge: tuple[ tuple[ float, float ], tuple[ float, float ] ] | None = None

    # Build layers
    self.createLayers()
    self.drawPitch()
    self.drawKeypoints()
    self.hover_item = self.createSingleMarker( self.pitch.colors.hover_color, "hover" )
    self.mapping_item = self.createSingleMarker( self.pitch.colors.sel_color, "mapping" )

    # Bind events
    self.canvas.bind( "<Motion>", self.handle_hover )
    self.canvas.bind( "<ButtonPress-1>", self.handle_press )
    self.canvas.bind( "<B1-Motion>", self.handle_drag )
    self.canvas.bind( "<ButtonRelease-1>", self.handle_release )

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
    for i, (vertex, pt) in enumerate(zip( self.cfg.vertices, self.cfg.labels )):
      mx = int( vertex[ 0 ] * scaleW + self.pitch.padding )
      my = int( vertex[ 1 ] * scaleL + self.pitch.padding )

      radius = 6
      self.canvas.create_oval( mx - radius, my - radius, mx + radius, my + radius, fill=self.pitch.colors.point_color.as_hex(), outline="black", width=1, tags=( "keypoints" ) )
      self.canvas.create_text( mx + 10, my - 10, text=pt, fill="white", font=( "Arial", 12 ), tags=( "keypoints" ) )

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
  def screen_vertex( self, index: int ) -> tuple[ float, float ]:
    vertex = self.cfg.vertices[ index ]
    scale_w, scale_l = self.pitch.get_pitch_scale
    return ( vertex[ 0 ] * scale_w + self.pitch.padding, vertex[ 1 ] * scale_l + self.pitch.padding )

  def screen_to_world( self, x: float, y: float ) -> tuple[ float, float ]:
    scale_w, scale_l = self.pitch.get_pitch_scale
    return ( ( x - self.pitch.padding ) / scale_w, ( y - self.pitch.padding ) / scale_l )

  def nearest_line( self, x: float, y: float ) -> tuple[ SelectionPoint, tuple[ tuple[ float, float ], tuple[ float, float ] ] ]:
    nearest_point: tuple[ float, float ] | None = None
    nearest_edge = None
    nearest_distance = float( "inf" )
    for start, end in self.cfg.edges:
      start_screen = self.screen_vertex( start - 1 )
      end_screen = self.screen_vertex( end - 1 )
      dx = end_screen[ 0 ] - start_screen[ 0 ]
      dy = end_screen[ 1 ] - start_screen[ 1 ]
      length_squared = dx * dx + dy * dy
      ratio = 0.0 if length_squared == 0 else ( ( x - start_screen[ 0 ] ) * dx + ( y - start_screen[ 1 ] ) * dy ) / length_squared
      ratio = max( 0.0, min( 1.0, ratio ) )
      candidate = ( start_screen[ 0 ] + ratio * dx, start_screen[ 1 ] + ratio * dy )
      distance = ( x - candidate[ 0 ] )**2 + ( y - candidate[ 1 ] )**2
      if distance < nearest_distance:
        nearest_distance = distance
        nearest_point = candidate
        nearest_edge = ( start_screen, end_screen )

    if nearest_point is None or nearest_edge is None:
      raise ValueError( "The pitch has no line segments" )
    world_x, world_y = self.screen_to_world( *nearest_point )
    return SelectionPoint( None, Point2D( world_x, world_y ) ), nearest_edge

  def point_at_cursor( self, x: float, y: float ) -> tuple[ SelectionPoint, tuple[ tuple[ float, float ], tuple[ float, float ] ] | None ]:
    nearest = self.pitch.nearestFieldPoint( x, y )
    nearest_screen = self.screen_vertex( nearest.index if nearest.index is not None else 0 )
    distance = ( x - nearest_screen[ 0 ] )**2 + ( y - nearest_screen[ 1 ] )**2
    if distance <= self.FIELD_POINT_SNAP_DISTANCE**2:
      return nearest, None
    return self.nearest_line( x, y )

  def handle_hover( self, event ):

    rx, ry = event.x, event.y
    nearest, _ = self.point_at_cursor( rx, ry )
    x, y = self.pitch.calcPointOffset( nearest ) if nearest else ( 0, 0 )
    self.canvas.coords( self.hover_item, x - 6, y - 6, x + 6, y + 6 )
    self.canvas.itemconfig( self.hover_item, state="normal" )
    if self.on_hover:
      self.on_hover( rx, ry )

  def handle_click( self, event ):

    rx, ry = event.x, event.y
    self.mapping, _ = self.point_at_cursor( rx, ry )
    x, y = self.pitch.calcPointOffset( self.mapping ) if self.mapping else ( 0, 0 )
    self.canvas.coords( self.mapping_item, x - 6, y - 6, x + 6, y + 6 )
    self.canvas.itemconfig( self.mapping_item, state="normal" )
    if self.on_click:
      self.on_click( rx, ry, self.mapping )

  def handle_press( self, event ):
    overlapping = self.canvas.find_overlapping( event.x - 8, event.y - 8, event.x + 8, event.y + 8 )
    for item_id in reversed( overlapping ):
      selection = self.selection_items.get( item_id )
      if selection is not None and selection[ 0 ].index is None:
        self.dragged_selection, self.dragged_edge = selection
        return "break"
    self.handle_click( event )

  def handle_drag( self, event ):
    if self.dragged_selection is None or self.dragged_edge is None:
      return
    start, end = self.dragged_edge
    dx = end[ 0 ] - start[ 0 ]
    dy = end[ 1 ] - start[ 1 ]
    length_squared = dx * dx + dy * dy
    ratio = 0.0 if length_squared == 0 else ( ( event.x - start[ 0 ] ) * dx + ( event.y - start[ 1 ] ) * dy ) / length_squared
    ratio = max( 0.0, min( 1.0, ratio ) )
    point = ( start[ 0 ] + ratio * dx, start[ 1 ] + ratio * dy )
    world_x, world_y = self.screen_to_world( *point )
    self.dragged_selection.coords = Point2D( world_x, world_y )
    self.updateSelectionMarkers( self.selected )
    if self.on_selection_move:
      self.on_selection_move()
    return "break"

  def handle_release( self, _event ):
    self.dragged_selection = None
    self.dragged_edge = None

  # -------------------------------------------------------------
  # Mapped selection markers
  # -------------------------------------------------------------
  def updateSelectionMarkers( self, selected: list[ SelectionPoint ] ):
    self.canvas.delete( "selection" )
    self.selected = selected
    self.selection_items.clear()
    color = self.pitch.colors.highlight_color.as_hex()
    for vertex in self.selected:
      mx, my = self.pitch.calcPointOffset( vertex )

      radius = 6
      edge = None
      if vertex.index is None:
        _, edge = self.nearest_line( mx, my )
      item_id = self.canvas.create_oval( mx - radius, my - radius, mx + radius, my + radius, fill=color, outline="black", width=1, tags=( "selection" ) )
      self.selection_items[ item_id ] = ( vertex, edge )
