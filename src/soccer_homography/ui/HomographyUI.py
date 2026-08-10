import tkinter as tk
from tkinter import filedialog

from soccer_homography.appState import AppState
from soccer_homography.dataTypes import Homography
from soccer_homography.encoder import GifEncoder, Mp4Encoder


class HomographyUI:

  # yapf: disable
  root:        tk.Tk
  appState:    AppState
  # yapf: enable

  def __init__( self, root: tk.Tk, state: AppState, x: int, y: int, playFunc, homoReplaceFunc ):
    self.root = root
    self.appState = state
    self.playFunc = playFunc
    self.homoReplaceFunc = homoReplaceFunc
    self.x = x
    self.y = y
    self.createWidgets()

  def createWidgets( self ):
    # lblHomography
    self.lblHomographyAction = tk.Label( self.root, text="Homography", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblHomographyAction.place( x=self.x, y=self.y + 6, width=100, height=24 )

    # btnLoadHomography
    self.btnLoadHomography = tk.Button( self.root, text="Load", font=( "Arial", 12 ), command=self.cmdLoadHomography )
    self.btnLoadHomography.place( x=self.x + 140, y=self.y, width=60, height=36 )

    # btnSaveHomography
    self.btnSaveHomography = tk.Button( self.root, text="Save", font=( "Arial", 12 ), command=self.cmdSaveHomography, state=tk.DISABLED )
    self.btnSaveHomography.place( x=self.x + 200, y=self.y, width=60, height=36 )

    # btnPlayHomography
    self.btnPlayHomography = tk.Button( self.root, text="Play", font=( "Arial", 12 ), command=self.cmdPlayHomography, state=tk.DISABLED )
    self.btnPlayHomography.place( x=self.x + 260, y=self.y, width=60, height=36 )

    # btnGIFHomography
    self.btnGIFHomography = tk.Button( self.root, text="GIF", font=( "Arial", 12 ), command=self.cmdGIFHomography, state=tk.DISABLED )
    self.btnGIFHomography.place( x=self.x + 320, y=self.y, width=60, height=36 )

    # btnMP4Homography
    self.btnMP4Homography = tk.Button( self.root, text="MP4", font=( "Arial", 12 ), command=self.cmdMP4Homography, state=tk.DISABLED )
    self.btnMP4Homography.place( x=self.x + 380, y=self.y, width=60, height=36 )

  def cmdLoadHomography( self ):
    filetypes = ( ( 'Homography files', '*.json' ),)

    filename = filedialog.askopenfilename( title='Open homography', initialdir='.', filetypes=filetypes )
    if filename is not None:
      # Reset the homography
      self.appState.data = Homography()
      # And then load it
      self.appState.data.load( filename )
      self.homoReplaceFunc()

  def cmdSaveHomography( self ):
    filetypes = ( ( 'Homography files', '*.json' ),)
    filename = filedialog.asksaveasfilename( title='Save homography', initialdir='.', filetypes=filetypes )
    if filename is not None:
      # And then save it
      self.appState.data.save( filename )

  def cmdPlayHomography( self ):
    self.playFunc( None )

  def cmdGIFHomography( self ):
    self.playFunc( GifEncoder() )

  def cmdMP4Homography( self ):
    self.playFunc( Mp4Encoder() )

  def setEnableStatus( self, hasHomography, hasTracks ):
    self.btnSaveHomography.config( state=tk.NORMAL if hasHomography else tk.DISABLED )
    self.btnPlayHomography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
    self.btnGIFHomography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
    self.btnMP4Homography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
