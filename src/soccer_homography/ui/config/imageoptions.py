import tkinter as tk
from dataclasses import dataclass, field
from tkinter import BooleanVar, StringVar, messagebox, ttk

from soccer_homography.appState import AppState
from soccer_homography.inference.crop_inference import (
  DEFAULT_IDENTIFICATION_PROMPT,
  loadIdentificationPrompt,
  saveIdentificationPrompt,
)


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
  identificationModel: StringVar = field( default_factory=lambda: tk.StringVar( value="VLM" ) )
  # yapf: enable


class ImageOptionsUI:

  def __init__( self, state: AppState, tab: ttk.Frame, on_change=None ) -> None:
    self.tab = tab
    self.appState: AppState = state
    self.uiOpts = UIOptions()
    self.on_change = on_change
    self.identificationPrompt = DEFAULT_IDENTIFICATION_PROMPT

  def createCheck( self, text, variable, parent=None ):
    return ttk.Checkbutton( parent or self.tab, text=text, variable=variable, command=self.uiToState, compound='left' )

  def setup( self ):
    optionsFrame = ttk.Frame( self.tab )
    optionsFrame.pack( side="left", anchor="nw", fill="y", padx=( 4, 8 ), pady=4 )
    self.optShowHough = self.createCheck( text="Show Edge Detection", variable=self.uiOpts.showHough, parent=optionsFrame )
    self.optPreBlur = self.createCheck( text="Blur before detection", variable=self.uiOpts.preBlur, parent=optionsFrame )
    self.optRemoveSky = self.createCheck( text="Remove sky?", variable=self.uiOpts.removeSky, parent=optionsFrame )
    self.optEdgeEnhance = self.createCheck( text="Edge enhancement", variable=self.uiOpts.edgeEnhance, parent=optionsFrame )
    self.optCloseEdges = self.createCheck( text="Try close edges", variable=self.uiOpts.closeEdges, parent=optionsFrame )
    edgeTypes = ( 'Canny', 'Scharr' )
    self.optEdgeType = ttk.Combobox( optionsFrame, textvariable=self.uiOpts.edgeType, values=edgeTypes )
    lineTypes = ( 'Hough', 'LineSegmentDetector' )
    self.optLineType = ttk.Combobox( optionsFrame, textvariable=self.uiOpts.lineType, values=lineTypes )
    self.optIdentificationModel = ttk.Combobox(
        optionsFrame,
        textvariable=self.uiOpts.identificationModel,
        values=( "VLM", "Clip" ),
        state="readonly",
    )

    for widget in (
        self.optShowHough,
        self.optPreBlur,
        self.optRemoveSky,
        self.optEdgeEnhance,
        self.optCloseEdges,
        self.optEdgeType,
        self.optLineType,
    ):
      widget.pack( anchor="w" )
    ttk.Label( optionsFrame, text="Person identification model" ).pack( anchor="w" )
    self.optIdentificationModel.pack( anchor="w" )

    promptFrame = ttk.LabelFrame( self.tab, text="Person identification prompt" )
    promptFrame.pack( side="left", anchor="nw", fill="both", expand=True, padx=4, pady=4 )
    self.promptText = tk.Text( promptFrame, height=5, wrap="word", undo=True )
    self.promptText.pack( side="left", fill="both", expand=True, padx=( 4, 0 ), pady=4 )
    promptScrollbar = ttk.Scrollbar( promptFrame, orient="vertical", command=self.promptText.yview )
    promptScrollbar.pack( side="right", fill="y", padx=( 0, 4 ), pady=4 )
    self.promptText.configure( yscrollcommand=promptScrollbar.set )
    self.promptSaveButton = ttk.Button( promptFrame, text="Save prompt", command=self.savePrompt )
    self.promptSaveButton.pack( side="bottom", anchor="e", padx=4, pady=( 0, 4 ) )
    self.identificationPrompt = loadIdentificationPrompt()
    self.promptText.insert( "1.0", self.identificationPrompt )

    self.optEdgeType.bind( '<<ComboboxSelected>>', self.comboChange )
    self.optLineType.bind( '<<ComboboxSelected>>', self.comboChange )

  def getIdentificationModel( self ) -> str:
    return self.uiOpts.identificationModel.get()

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

  def getIdentificationPrompt( self ) -> str:
    return self.promptText.get( "1.0", "end-1c" ).strip()

  def savePrompt( self, prompt: str | None = None ) -> bool:
    prompt = self.getIdentificationPrompt() if prompt is None else prompt.strip()
    if not prompt:
      messagebox.showerror( "Prompt not saved", "The identification prompt cannot be empty.", parent=self.tab )
      return False
    try:
      saveIdentificationPrompt( prompt )
    except OSError as error:
      messagebox.showerror( "Prompt not saved", str( error ), parent=self.tab )
      return False
    self.identificationPrompt = prompt
    return True