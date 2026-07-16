from dataclasses import dataclass, field
from log import logging
import tkinter as tk
from tkinter import BooleanVar, StringVar, ttk
from tkinter.scrolledtext import ScrolledText

from appState import AppState
from dataTypes import ViewTransform


class TkinterLogHandler( logging.Handler ):

  def __init__( self, text_widget ):
    super().__init__()
    self.text_widget = text_widget

  def emit( self, record ):
    msg = self.format( record )
    # Append text safely from Tkinter main thread
    self.text_widget.after( 0, self._append, msg )

  def _append( self, msg ):
    self.text_widget.insert( tk.END, msg + "\n" )
    self.text_widget.see( tk.END )  # auto-scroll


@dataclass
class UIOptions:
  # yapf: disable
  showHough:   BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - Show Hough layer
  preBlur:     BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - blur for edge detection
  removeSky:   BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - try to remove sky
  edgeEnhance: BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - apply CLAHE enhancement
  closeEdges:  BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - close edges
  edgeType:    StringVar  = field( default_factory=tk.StringVar )  # Combo box - edge type - Canny, Scharr
  lineType:    StringVar  = field( default_factory=tk.StringVar )  # Combo box - line type - Hough, LineSegmentDetector
  # yapf: enable


class ImagePreview:

  def __init__( self, tab: ttk.Frame ) -> None:
    self.tab = tab

  def setup( self ):
    self.tblPreview = ttk.Treeview( self.tab, columns=( "Property", "Value" ), show="headings" )
    self.tblPreview.place( x=0, y=24, width=600, height=160 )
    #
    self.tblPreview.heading( "Property", text="Property" )
    self.tblPreview.heading( "Value", text="Value" )
    #
    initialData = [ ( "Zoom", "" ), ( "Offset X", "" ), ( "Offset Y", "" ) ]
    self.tblPreview.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblPreview.tag_configure( "evenrow", background="#993333", foreground="white" )
    #
    self.constantRowIDs = []
    for i, row in enumerate( initialData ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      iid = self.tblPreview.insert( "", tk.END, values=row, tags=( tag,) )
      self.constantRowIDs.append( iid )

  def refresh( self, xf: ViewTransform ):
    updates = [ ( "Zoom", f"{xf.scale:.3f}" ), ( "Offset X", f"{xf.offset.x:.3f}" ), ( "Offset Y", f"{xf.offset.y:.3f}" ) ]
    for iid, new_values in zip( self.constantRowIDs, updates ):
      self.tblPreview.item( iid, values=new_values )


class HomographyData:

  def __init__( self, appState: AppState, tab: ttk.Frame ) -> None:
    self.appState = appState
    self.tab = tab

  def setup( self ):
    self.tblHomographyData = ttk.Treeview( self.tab, columns=( "Field Point", "World Position", "Image Position" ), show="headings" )
    self.tblHomographyData.place( x=0, y=24, width=600, height=160 )

    self.tblHomographyData.heading( "Field Point", text="Field Point" )
    self.tblHomographyData.heading( "World Position", text="World Position" )
    self.tblHomographyData.heading( "Image Position", text="Image Position" )

    self.tblHomographyData.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblHomographyData.tag_configure( "evenrow", background="#993333", foreground="white" )

  def refresh( self ):
    for iid in self.tblHomographyData.get_children():
      self.tblHomographyData.delete( iid )

    for i, ( sel, world ) in enumerate( zip( self.appState.data.img_pts_4k, self.appState.data.world_pts ) ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      row = (
          str( self.appState.cfg.labels[ world.index ] ) if
          ( self.appState.cfg and world.index is not None ) else "", f"{world.coords.x:.3f},{world.coords.y:.3f}", f"{sel.coords.x},{sel.coords.y}"
      )
      self.tblHomographyData.insert( "", tk.END, values=row, tags=( tag,) )


class ImageOptionsUI:

  def __init__( self, state: AppState, tab: ttk.Frame, on_change=None ) -> None:
    self.tab = tab
    self.appState: AppState = state
    self.uiOpts = UIOptions()
    self.on_change = on_change

  def createCheck( self, text, variable ):
    return ttk.Checkbutton( self.tab, text=text, variable=variable, command=self.uiToState, compound='left' )

  def setup( self ):
    self.optShowHough = self.createCheck( text="Show Edge Detection", variable=self.uiOpts.showHough )
    self.optPreBlur = self.createCheck( text="Blur before detection", variable=self.uiOpts.preBlur )
    self.optRemoveSky = self.createCheck( text="Remove sky?", variable=self.uiOpts.removeSky )
    self.optEdgeEnhance = self.createCheck( text="Edge enhancement", variable=self.uiOpts.edgeEnhance )
    self.optCloseEdges = self.createCheck( text="Try close edges", variable=self.uiOpts.closeEdges )
    edgeTypes = ( 'Canny', 'Scharr' )
    self.optEdgeType = ttk.Combobox( self.tab, textvariable=self.uiOpts.edgeType, values=edgeTypes )
    lineTypes = ( 'Hough', 'LineSegmentDetector' )
    self.optLineType = ttk.Combobox( self.tab, textvariable=self.uiOpts.lineType, values=lineTypes )

    self.optShowHough.pack( anchor='w' )
    self.optPreBlur.pack( anchor='w' )
    self.optRemoveSky.pack( anchor='w' )
    self.optEdgeEnhance.pack( anchor='w' )
    self.optCloseEdges.pack( anchor='w' )
    self.optEdgeType.pack( anchor='w' )
    self.optLineType.pack( anchor='w' )

    self.optEdgeType.bind( '<<ComboboxSelected>>', self.comboChange )
    self.optLineType.bind( '<<ComboboxSelected>>', self.comboChange )

  def uiToState( self ):
    self.appState.imgOpts.closeEdges = self.uiOpts.closeEdges.get()
    self.appState.imgOpts.edgeEnhance = self.uiOpts.edgeEnhance.get()
    self.appState.imgOpts.edgeType = self.uiOpts.edgeType.get()
    self.appState.imgOpts.lineType = self.uiOpts.lineType.get()
    self.appState.imgOpts.preBlur = self.uiOpts.preBlur.get()
    self.appState.imgOpts.removeSky = self.uiOpts.removeSky.get()
    self.appState.imgOpts.showHough = self.uiOpts.showHough.get()
    self.appState.imgOpts.edgeType = self.uiOpts.edgeType.get()
    self.appState.imgOpts.lineType = self.uiOpts.lineType.get()
    if self.on_change is not None:
      self.on_change()

  def comboChange( self, event ):
    self.uiToState()

  def stateToUI( self ):
    self.uiOpts.closeEdges.set( self.appState.imgOpts.closeEdges )
    self.uiOpts.edgeEnhance.set( self.appState.imgOpts.edgeEnhance )
    self.uiOpts.edgeType.set( self.appState.imgOpts.edgeType )
    self.uiOpts.lineType.set( self.appState.imgOpts.lineType )
    self.uiOpts.preBlur.set( self.appState.imgOpts.preBlur )
    self.uiOpts.removeSky.set( self.appState.imgOpts.removeSky )
    self.uiOpts.showHough.set( self.appState.imgOpts.showHough )


class Log:

  def __init__( self, tab: ttk.Frame ) -> None:
    self.tab = tab

  def setup( self ):
    self.txtLog = ScrolledText( self.tab, width=80, height=20, state="normal" )
    self.txtLog.pack( fill="both", expand=True )


class Tracks:

  def __init__( self, appState: AppState, tab: ttk.Frame ) -> None:
    self.appState = appState
    self.tab = tab

  def setup( self ):
    self.tblTrackData = ttk.Treeview( self.tab, columns=( "Track ID", "Num Frames", "Person" ), show="headings" )
    self.tblTrackData.place( x=0, y=24, width=600, height=160 )

    self.tblTrackData.heading( "Track ID", text="Track ID" )
    self.tblTrackData.heading( "Num Frames", text="Num Frames" )
    self.tblTrackData.heading( "Person", text="Person" )

    self.tblTrackData.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblTrackData.tag_configure( "evenrow", background="#993333", foreground="white" )

  def refresh( self ):
    for iid in self.tblTrackData.get_children():
      self.tblTrackData.delete( iid )

    for i, ( key, track ) in enumerate( self.appState.tracks.items() ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      row = ( str( key ), str( len( track.boxes ) ), track.person.name if track.person is not None else "<Unknown>" )
      self.tblTrackData.insert( "", tk.END, values=row, tags=( tag,) )


class Configuration:

  def __init__( self, parent: tk.Tk, state: AppState, on_change=None ):

    self.parent = parent
    self.appState: AppState = state

    # Build UI
    self.createLayout( on_change )

    self.tabImageOptions.stateToUI()

    handler = TkinterLogHandler( self.tabLog.txtLog )
    formatter = logging.Formatter( "%(asctime)s - %(levelname)s - %(message)s" )
    handler.setFormatter( formatter )

    logger = logging.getLogger( "SportsTracker" )
    logger.addHandler( handler )

  # -------------------------------------------------------------
  # Layout controls
  # -------------------------------------------------------------
  def createLayout( self, on_change ):

    self.root = ttk.Frame( master=self.parent, width=1280, height=240, borderwidth=5, relief='groove' )
    self.root.place( x=50, y=720 + 56 )

    self.nbControl = ttk.Notebook( self.root, width=1280, height=240 )
    self.tabImagePreview = ImagePreview( self.createTab( "Image Preview" ) )
    self.tabHomographyData = HomographyData( self.appState, self.createTab( "Homography Data" ) )
    self.tabImageOptions = ImageOptionsUI( self.appState, self.createTab( "Image Options" ), on_change )
    self.tabLog = Log( self.createTab( "Log" ) )
    self.tabTracks = Tracks( self.appState, self.createTab( "Tracks" ) )
    self.nbControl.pack( expand=1, fill='both' )

    self.tabImagePreview.setup()
    self.tabHomographyData.setup()
    self.tabImageOptions.setup()
    self.tabLog.setup()
    self.tabTracks.setup()

  def createTab( self, text ):
    newTab = ttk.Frame( self.nbControl )
    self.nbControl.add( newTab, text=text )
    return newTab
