from dataclasses import dataclass, field
import tkinter as tk
from tkinter import BooleanVar, StringVar, ttk

from appState import AppState
from dataTypes import ViewTransform


@dataclass
class UIOptions:
  # Checkbox - Show Hough layer
  showHough: BooleanVar = field( default_factory=tk.BooleanVar )
  # Checkbox - blur for edge detection
  preBlur: BooleanVar = field( default_factory=tk.BooleanVar )
  # Checkbox - try to remove sky
  removeSky: BooleanVar = field( default_factory=tk.BooleanVar )
  # Checkbox - apply CLAHE enhancement
  edgeEnhance: BooleanVar = field( default_factory=tk.BooleanVar )
  # Checkbox - close edges
  closeEdges: BooleanVar = field( default_factory=tk.BooleanVar )
  # Combo box - edge type - Canny, Scharr
  edgeType: StringVar = field( default_factory=tk.StringVar )
  # Combo box - line type - Hough, LineSegmentDetector
  lineType: StringVar = field( default_factory=tk.StringVar )


class Configuration:

  def __init__( self, parent: tk.Tk, state: AppState, on_change=None ):

    self.parent = parent
    self.appState: AppState = state
    self.uiOpts = UIOptions()
    self.on_change = on_change

    self.stateToUI()

    # Build UI
    self.createLayout()

  def stateToUI( self ):
    self.uiOpts.closeEdges.set( self.appState.imgOpts.closeEdges )
    self.uiOpts.edgeEnhance.set( self.appState.imgOpts.edgeEnhance )
    self.uiOpts.edgeType.set( self.appState.imgOpts.edgeType )
    self.uiOpts.lineType.set( self.appState.imgOpts.lineType )
    self.uiOpts.preBlur.set( self.appState.imgOpts.preBlur )
    self.uiOpts.removeSky.set( self.appState.imgOpts.removeSky )
    self.uiOpts.showHough.set( self.appState.imgOpts.showHough )

  # -------------------------------------------------------------
  # Layout controls
  # -------------------------------------------------------------
  def createLayout( self ):

    self.root = ttk.Frame( master=self.parent, width=1280, height=240, borderwidth=5, relief='groove' )
    self.root.place( x=50, y=720 + 56 )

    self.nbControl = ttk.Notebook( self.root, width=1280, height=240 )
    self.tabImagePreview = self.createTab( "Image Preview" )
    self.tabHomographyData = self.createTab( "Homography Data" )
    self.tabImageOptions = self.createTab( "Image Options" )
    self.nbControl.pack( expand=1, fill='both' )

    self.setupImagePreview()
    self.setupHomographyData()
    self.setupImageOptions()

  def createTab( self, text ):
    newTab = ttk.Frame( self.nbControl )
    self.nbControl.add( newTab, text=text )
    return newTab

  def refreshImagePreview( self, xf: ViewTransform ):
    updates = [ ( "Zoom", f"{xf.scale:.3f}" ), ( "Offset X", f"{xf.offset.x:.3f}" ), ( "Offset Y", f"{xf.offset.y:.3f}" ) ]
    for iid, new_values in zip( self.constantRowIDs, updates ):
      self.tblPreview.item( iid, values=new_values )

  def refreshHomographyData( self ):
    for iid in self.tblHomographyData.get_children():
      self.tblHomographyData.delete( iid )

    for i, ( sel, world ) in enumerate( zip( self.appState.data.img_pts_4k, self.appState.data.world_pts ) ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      row = (
          str( self.appState.cfg.labels[ world.index ] ) if
          ( self.appState.cfg and world.index is not None ) else "", f"{world.coords.x:.3f},{world.coords.y:.3f}", f"{sel.coords.x},{sel.coords.y}"
      )
      self.tblHomographyData.insert( "", tk.END, values=row, tags=( tag,) )

  def setupImagePreview( self ):
    self.tblPreview = ttk.Treeview( self.tabImagePreview, columns=( "Property", "Value" ), show="headings" )
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

  def setupHomographyData( self ):
    self.tblHomographyData = ttk.Treeview( self.tabHomographyData, columns=( "Field Point", "World Position", "Image Position" ), show="headings" )
    self.tblHomographyData.place( x=0, y=24, width=600, height=160 )

    self.tblHomographyData.heading( "Field Point", text="Field Point" )
    self.tblHomographyData.heading( "World Position", text="World Position" )
    self.tblHomographyData.heading( "Image Position", text="Image Position" )

    self.tblHomographyData.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblHomographyData.tag_configure( "evenrow", background="#993333", foreground="white" )

  def createCheck( self, text, variable ):
    return ttk.Checkbutton( self.tabImageOptions, text=text, variable=variable, command=self.uiToState, compound='left' )

  def setupImageOptions( self ):
    self.optShowHough = self.createCheck( text="Show Edge Detection", variable=self.uiOpts.showHough )
    self.optPreBlur = self.createCheck( text="Blur before detection", variable=self.uiOpts.preBlur )
    self.optRemoveSky = self.createCheck( text="Remove sky?", variable=self.uiOpts.removeSky )
    self.optEdgeEnhance = self.createCheck( text="Edge enhancement", variable=self.uiOpts.edgeEnhance )
    self.optCloseEdges = self.createCheck( text="Try close edges", variable=self.uiOpts.closeEdges )
    edgeTypes = ( 'Canny', 'Scharr' )
    self.optEdgeType = ttk.Combobox( self.tabImageOptions, textvariable=self.uiOpts.edgeType, values=edgeTypes )
    lineTypes = ( 'Hough', 'LineSegmentDetector' )
    self.optLineType = ttk.Combobox( self.tabImageOptions, textvariable=self.uiOpts.lineType, values=lineTypes )

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
