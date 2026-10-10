import tkinter as tk
from dataclasses import dataclass, field
from tkinter import IntVar, StringVar, messagebox, ttk

from soccer_homography.appState import AppState
from soccer_homography.inference.crop_inference import (
    DEFAULT_IDENTIFICATION_PROMPT,
    loadIdentificationPrompt,
    saveIdentificationPrompt,
)


@dataclass
class ModelOptionsState:
  # yapf: disable
  identificationModel: StringVar = field( default_factory=lambda: tk.StringVar( value="Clip" ) )
  identificationPrompt: StringVar = field( default_factory=lambda: tk.StringVar( value=DEFAULT_IDENTIFICATION_PROMPT ) )
  cropsPerSegment: IntVar = field( default_factory=lambda: tk.IntVar( value=6 ) )
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
    self.optIdentificationModel = ttk.Combobox(
        optionsFrame,
        textvariable=self.modelOpts.identificationModel,
        values=( "Clip", "VLM" ),
        state="readonly",
    )

    ttk.Label( optionsFrame, text="Person identification model" ).pack( anchor="w" )
    self.optIdentificationModel.pack( anchor="w" )

    ttk.Label( optionsFrame, text="Crops per track segment" ).pack( anchor="w", pady=( 12, 0 ) )
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

  def getIdentificationModel( self ) -> str:
    return self.modelOpts.identificationModel.get()

  def getCropsPerSegment( self ) -> int:
    return self.modelOpts.cropsPerSegment.get()

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
