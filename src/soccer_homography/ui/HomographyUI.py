import tkinter as tk
from tkinter import ttk

from soccer_homography.appState import AppState
from soccer_homography.encoder import GifEncoder, Mp4Encoder


class HomographyUI:

  # yapf: disable
  root:        tk.Tk
  appState:    AppState
  # yapf: enable

  def __init__( self, root: tk.Tk, state: AppState, x: int, y: int, playFunc, homoReplaceFunc, saveHomographyFunc, loadHomographyFunc ):
    self.root = root
    self.appState = state
    self.playFunc = playFunc
    self.homoReplaceFunc = homoReplaceFunc
    self.saveHomographyFunc = saveHomographyFunc
    self.loadHomographyFunc = loadHomographyFunc
    self.x = x
    self.y = y
    self.createWidgets()

  def createWidgets( self ):
    # lblHomography
    self.lblHomographyAction = tk.Label( self.root, text="Homography", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblHomographyAction.place( x=self.x, y=self.y + 6, width=100, height=24 )

    # btnPlayHomography
    self.btnPlayHomography = tk.Button( self.root, text="Play", font=( "Arial", 10 ), command=self.cmdPlayHomography, state=tk.DISABLED )
    self.btnPlayHomography.place( x=self.x + 110, y=self.y, width=48, height=28 )

    self.btnExportHomography = tk.Button( self.root, text="Export", font=( "Arial", 10 ), command=self.cmdExportHomography, state=tk.DISABLED )
    self.btnExportHomography.place( x=self.x + 158, y=self.y, width=68, height=28 )

    self.btnSaveHomographyDB = tk.Button( self.root, text="Save", font=( "Arial", 10 ), command=self.saveHomographyFunc, state=tk.DISABLED )
    self.btnSaveHomographyDB.place( x=self.x + 226, y=self.y, width=48, height=28 )

    # btnLoadHomography
    self.btnLoadHomography = tk.Button( self.root, text="Load", font=( "Arial", 10 ), command=self.loadHomographyFunc, state=tk.DISABLED )
    self.btnLoadHomography.place( x=self.x + 274, y=self.y, width=48, height=28 )

  def cmdPlayHomography( self ):
    self.playFunc( None )

  def cmdExportHomography( self ) -> None:
    dialog = tk.Toplevel( self.root )
    dialog.title( "Export Homography" )
    dialog.transient( self.root )
    dialog.grab_set()

    format_value = tk.StringVar( master=dialog, value="MP4" )
    include_heatmap = tk.BooleanVar( master=dialog, value=True )
    full_clip_range = tk.BooleanVar( master=dialog, value=False )

    ttk.Label( dialog, text="Format" ).grid( row=0, column=0, padx=10, pady=( 10, 4 ), sticky="w" )
    format_selection = ttk.Combobox(
        dialog,
        textvariable=format_value,
        values=( "MP4", "GIF" ),
        state="readonly",
        width=12,
    )
    format_selection.grid( row=0, column=1, padx=10, pady=( 10, 4 ), sticky="ew" )
    ttk.Checkbutton(
        dialog,
        text="Include heatmap",
        variable=include_heatmap,
    ).grid( row=1, column=0, columnspan=2, padx=10, pady=4, sticky="w" )
    ttk.Checkbutton(
        dialog,
        text="Use full clip range",
        variable=full_clip_range,
    ).grid( row=2, column=0, columnspan=2, padx=10, pady=4, sticky="w" )

    buttons = ttk.Frame( dialog )
    buttons.grid( row=3, column=0, columnspan=2, padx=10, pady=( 8, 10 ), sticky="e" )
    ttk.Button( buttons, text="OK", command=lambda: self.confirmExport(
        dialog,
        format_value.get(),
        include_heatmap.get(),
        full_clip_range.get(),
    ) ).pack( side=tk.LEFT, padx=( 0, 6 ) )
    ttk.Button( buttons, text="Cancel", command=dialog.destroy ).pack( side=tk.LEFT )

    dialog.columnconfigure( 1, weight=1 )
    dialog.resizable( False, False )

  def confirmExport(
      self,
      dialog: tk.Toplevel,
      file_format: str,
      include_heatmap: bool,
      full_clip_range: bool,
  ) -> None:
    encoder = Mp4Encoder() if file_format == "MP4" else GifEncoder()
    self.playFunc( encoder, include_heatmap, full_clip_range )
    dialog.destroy()

  def setEnableStatus( self, hasHomography, hasTracks, hasClip=False ):
    self.btnLoadHomography.config( state=tk.NORMAL if self.appState.curClipID > 0 else tk.DISABLED )
    self.btnSaveHomographyDB.config( state=tk.NORMAL if hasHomography and hasClip else tk.DISABLED )
    self.btnPlayHomography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
    self.btnExportHomography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
