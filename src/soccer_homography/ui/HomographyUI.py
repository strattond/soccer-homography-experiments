import tkinter as tk

from soccer_homography.appState import AppState
from soccer_homography.encoder import GifEncoder, Mp4Encoder


class HomographyUI:

  # yapf: disable
  root:        tk.Tk
  appState:    AppState
  # yapf: enable

  def __init__( self, root: tk.Tk, state: AppState, x: int, y: int, playFunc, homoReplaceFunc, saveHomographyFunc ):
    self.root = root
    self.appState = state
    self.playFunc = playFunc
    self.homoReplaceFunc = homoReplaceFunc
    self.saveHomographyFunc = saveHomographyFunc
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

    # btnGIFHomography
    self.btnGIFHomography = tk.Button( self.root, text="GIF", font=( "Arial", 10 ), command=self.cmdGIFHomography, state=tk.DISABLED )
    self.btnGIFHomography.place( x=self.x + 158, y=self.y, width=48, height=28 )

    # btnMP4Homography
    self.btnMP4Homography = tk.Button( self.root, text="MP4", font=( "Arial", 10 ), command=self.cmdMP4Homography, state=tk.DISABLED )
    self.btnMP4Homography.place( x=self.x + 206, y=self.y, width=48, height=28 )

    self.btnSaveHomographyDB = tk.Button( self.root, text="Save Homography", font=( "Arial", 10 ), command=self.saveHomographyFunc, state=tk.DISABLED )
    self.btnSaveHomographyDB.place( x=self.x + 254, y=self.y, width=108, height=28 )

  def cmdPlayHomography( self ):
    self.playFunc( None )

  def cmdGIFHomography( self ):
    self.playFunc( GifEncoder() )

  def cmdMP4Homography( self ):
    self.playFunc( Mp4Encoder() )

  def setEnableStatus( self, hasHomography, hasTracks, hasClip=False ):
    self.btnSaveHomographyDB.config( state=tk.NORMAL if hasHomography and hasClip else tk.DISABLED )
    self.btnPlayHomography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
    self.btnGIFHomography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
    self.btnMP4Homography.config( state=tk.NORMAL if hasTracks else tk.DISABLED )
