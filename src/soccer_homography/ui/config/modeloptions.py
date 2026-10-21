import tkinter as tk
from dataclasses import dataclass, field
from tkinter import IntVar, StringVar, messagebox, ttk

from soccer_homography.appState import AppState
from soccer_homography.data import ExponentialSmoother, KalmanSmoother, PositionSmoother
from soccer_homography.inference.crop_inference import (
    DEFAULT_IDENTIFICATION_PROMPT,
    loadIdentificationPrompt,
    saveIdentificationPrompt,
)

DETECTOR_PROFILES = {
    "Quality: YOLO26x TensorRT 1280": ( "x", "engine", ( 1280, 1280 ), 1 ),
    "Balanced: YOLO26m PyTorch 640": ( "m", "pt", ( 640, 640 ), 8 ),
    "Fast: YOLO26m PyTorch 320": ( "m", "pt", ( 320, 320 ), 8 ),
}


@dataclass
class ModelOptionsState:
  # yapf: disable
  detectorProfile: StringVar = field( default_factory=lambda: tk.StringVar( value="Quality: YOLO26x TensorRT 1280" ) )
  identificationModel: StringVar = field( default_factory=lambda: tk.StringVar( value="Clip" ) )
  identificationPrompt: StringVar = field( default_factory=lambda: tk.StringVar( value=DEFAULT_IDENTIFICATION_PROMPT ) )
  cropsPerSegment: IntVar = field( default_factory=lambda: tk.IntVar( value=6 ) )
  positionSmoother: StringVar = field( default_factory=lambda: tk.StringVar( value="Kalman" ) )
  # yapf: enable


class ModelOptions:

  def __init__( self, state: AppState, tab: ttk.Frame, on_change=None ) -> None:
    self.tab = tab
    self.appState: AppState = state
    self.modelOpts = ModelOptionsState()
    self.on_change = on_change

  def setup( self ):
    optionsFrame = ttk.Frame( self.tab )
    optionsFrame.pack( side="left", anchor="nw", fill="y", padx=( 4, 8 ), pady=4 )
    ttk.Label( optionsFrame, text="Detection speed / quality" ).pack( anchor="w" )
    self.optDetectorProfile = ttk.Combobox(
        optionsFrame,
        textvariable=self.modelOpts.detectorProfile,
        values=tuple( DETECTOR_PROFILES ),
        state="readonly",
        width=34,
    )
    self.optDetectorProfile.pack( anchor="w" )
    self.optDetectorProfile.bind( "<<ComboboxSelected>>", self.onDetectorProfileChanged )

    self.optIdentificationModel = ttk.Combobox(
        optionsFrame,
        textvariable=self.modelOpts.identificationModel,
        values=( "Clip", "VLM" ),
        state="readonly",
    )

    ttk.Label( optionsFrame, text="Person identification model" ).pack( anchor="w" )
    self.optIdentificationModel.pack( anchor="w" )

    ttk.Label( optionsFrame, text="Crops per track segment" ).pack( anchor="w", pady=( 6, 0 ) )
    self.optCropsPerSegment = ttk.Spinbox(
        optionsFrame,
        textvariable=self.modelOpts.cropsPerSegment,
        from_=1,
        to=50,
        increment=1,
        state="readonly",
        width=8,
    )
    self.optCropsPerSegment.pack( anchor="w" )

    ttk.Label(
        optionsFrame,
        text="Position smoother",
    ).pack(
        anchor="w", pady=( 6, 0 )
    )
    self.optPositionSmoother = ttk.Combobox(
        optionsFrame,
        textvariable=self.modelOpts.positionSmoother,
        values=( "Exponential", "Kalman" ),
        state="readonly",
    )
    self.optPositionSmoother.pack( anchor="w" )
    self.optPositionSmoother.bind( "<<ComboboxSelected>>", self.onSmootherChanged )

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

  def onDetectorProfileChanged( self, _event=None ) -> None:
    profile = self.modelOpts.detectorProfile.get()
    try:
      size, engine, img_size, batch_size = DETECTOR_PROFILES[ profile ]
    except KeyError as error:
      raise ValueError( f"Unsupported detector profile: {profile}" ) from error
    self.appState.mdlOpts.size = size
    self.appState.mdlOpts.engine = engine
    self.appState.mdlOpts.imgSz = img_size
    self.appState.mdlOpts.batchSize = batch_size

  def getIdentificationModel( self ) -> str:
    return self.modelOpts.identificationModel.get()

  def getCropsPerSegment( self ) -> int:
    return self.modelOpts.cropsPerSegment.get()

  def getSmootherType( self ) -> type[ PositionSmoother ]:
    name = self.modelOpts.positionSmoother.get()
    if name == "Exponential":
      return ExponentialSmoother
    if name == "Kalman":
      return KalmanSmoother
    raise ValueError( f"Unsupported position smoother: {name}" )

  def onSmootherChanged( self, _event=None ) -> None:
    if self.on_change is not None:
      self.on_change()

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
