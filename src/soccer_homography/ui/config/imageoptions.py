import tkinter as tk
from dataclasses import dataclass, field
from tkinter import BooleanVar, StringVar, ttk

from soccer_homography.appState import AppState


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