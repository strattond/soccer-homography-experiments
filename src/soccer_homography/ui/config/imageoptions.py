import tkinter as tk
from dataclasses import dataclass, field
from tkinter import BooleanVar, StringVar, ttk

import cv2

from soccer_homography.appState import AppState


@dataclass
class UIOptions:
  # yapf: disable
  showHough:   BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - Show Hough layer
  showProjectedField: BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - Show projected pitch overlay
  fitAllHomographyPoints: BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - use all calibration pairs
  preBlur:     BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - blur for edge detection
  removeSky:   BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - try to remove sky
  edgeEnhance: BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - apply CLAHE enhancement
  closeEdges:  BooleanVar = field( default_factory=tk.BooleanVar ) # Checkbox - close edges
  edgeType:    StringVar  = field( default_factory=tk.StringVar )  # Combo box - edge type - Canny, Scharr
  lineType:    StringVar  = field( default_factory=tk.StringVar )  # Combo box - line type - Hough, LineSegmentDetector
  # yapf: enable


class ImageOptionsUI:

  def __init__( self, state: AppState, tab: ttk.Frame, on_change=None, on_homography_fit_change=None ) -> None:
    self.tab = tab
    self.appState: AppState = state
    self.uiOpts = UIOptions()
    self.on_change = on_change
    self.on_homography_fit_change = on_homography_fit_change
    self.uiOpts.showProjectedField.set( self.appState.imgOpts.showProjectedField )
    self.uiOpts.fitAllHomographyPoints.set( self.appState.imgOpts.fitAllHomographyPoints )

  def createCheck( self, text, variable, parent=None ):
    return ttk.Checkbutton( parent or self.tab, text=text, variable=variable, command=self.uiToState, compound='left' )

  def setup( self ):
    optionsFrame = ttk.Frame( self.tab )
    optionsFrame.pack( side="left", anchor="nw", fill="y", padx=( 4, 8 ), pady=4 )
    self.optShowHough = self.createCheck( text="Show Edge Detection", variable=self.uiOpts.showHough, parent=optionsFrame )
    self.optShowProjectedField = self.createCheck( text="Show Projected Field", variable=self.uiOpts.showProjectedField, parent=optionsFrame )
    self.optFitAllHomographyPoints = self.createCheck(
        text="Fit all calibration points (no outlier rejection)",
        variable=self.uiOpts.fitAllHomographyPoints,
        parent=optionsFrame,
    )
    self.optPreBlur = self.createCheck( text="Blur before detection", variable=self.uiOpts.preBlur, parent=optionsFrame )
    self.optRemoveSky = self.createCheck( text="Remove sky?", variable=self.uiOpts.removeSky, parent=optionsFrame )
    self.optEdgeEnhance = self.createCheck( text="Edge enhancement", variable=self.uiOpts.edgeEnhance, parent=optionsFrame )
    self.optCloseEdges = self.createCheck( text="Try close edges", variable=self.uiOpts.closeEdges, parent=optionsFrame )
    edgeTypes = ( 'Canny', 'Scharr' )
    self.optEdgeType = ttk.Combobox( optionsFrame, textvariable=self.uiOpts.edgeType, values=edgeTypes )
    lineTypes = ( 'Hough', 'LineSegmentDetector' )
    self.optLineType = ttk.Combobox( optionsFrame, textvariable=self.uiOpts.lineType, values=lineTypes )

    for widget in (
        self.optShowHough,
        self.optShowProjectedField,
        self.optFitAllHomographyPoints,
        self.optPreBlur,
        self.optRemoveSky,
        self.optEdgeEnhance,
        self.optCloseEdges,
        self.optEdgeType,
        self.optLineType,
    ):
      widget.pack( anchor="w" )

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
    self.appState.imgOpts.showProjectedField = self.uiOpts.showProjectedField.get()
    fit_all_points_changed = self.appState.imgOpts.fitAllHomographyPoints != self.uiOpts.fitAllHomographyPoints.get()
    self.appState.imgOpts.fitAllHomographyPoints = self.uiOpts.fitAllHomographyPoints.get()
    self.appState.imgOpts.edgeType = self.uiOpts.edgeType.get()
    self.appState.imgOpts.lineType = self.uiOpts.lineType.get()
    if self.on_change is not None:
      self.on_change()
    if fit_all_points_changed and self.on_homography_fit_change is not None:
      self.on_homography_fit_change()

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
    self.uiOpts.showProjectedField.set( self.appState.imgOpts.showProjectedField )
    self.uiOpts.fitAllHomographyPoints.set( self.appState.imgOpts.fitAllHomographyPoints )

  def getHomographyMethod( self ) -> int:
    return 0 if self.appState.imgOpts.fitAllHomographyPoints else cv2.RANSAC
