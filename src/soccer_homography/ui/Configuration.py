import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from soccer_homography.appState import AppState
from soccer_homography.log import logging
from soccer_homography.ui.config import ImageOptionsUI, ImagePreview, Tracks, homographyData


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


class Log:

  def __init__( self, tab: ttk.Frame ) -> None:
    self.tab = tab

  def setup( self ):
    self.txtLog = ScrolledText( self.tab, width=80, height=20, state="normal" )
    self.txtLog.pack( fill="both", expand=True )


class Configuration:

  def __init__( self, parent: tk.Tk, state: AppState, on_change=None ):

    self.parent = parent
    self.appState: AppState = state

    # Build UI
    self.createLayout( on_change )

    # Set options based on current config
    self.tabImageOptions.stateToUI()

    # Setup logging handler so anything written gets put in the UI
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

    self.nbControl = ttk.Notebook( self.root, width=640, height=240 )
    self.tabImagePreview = ImagePreview( self.createTab( "Image Preview" ) )
    self.tabHomographyData = homographyData( self.appState, self.createTab( "Homography Data" ) )
    self.tabImageOptions = ImageOptionsUI( self.appState, self.createTab( "Image Options" ), on_change )
    self.tabLog = Log( self.createTab( "Log" ) )
    self.tabTracks = Tracks( self.appState, self.createTab( "Tracks" ) )
    #self.tabCams = Cameras( self.appState, self.createTab( "Cameras" ) )
    #self.tabVideos = Videos( self.appState, self.createTab( "Videos" ) )
    #self.tabMatches = Matches( self.appState, self.createTab( "Matches" ) )
    #self.tabPeople = People( self.appState, self.createTab( "People" ) )
    #self.tabClips = Clips( self.appState, self.createTab( "Matches" ) )
    self.nbControl.pack( expand=1, fill='both' )
    self.allTabs = [ self.tabHomographyData, self.tabImageOptions, self.tabImagePreview, self.tabLog, self.tabTracks ]

    for tab in self.allTabs:
      tab.setup()

  def createTab( self, text ):
    newTab = ttk.Frame( self.nbControl )
    self.nbControl.add( newTab, text=text )
    return newTab
