import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from soccer_homography.appState import AppState
from soccer_homography.db import listClipParticipants
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
    self.text_widget.yview_moveto( 1.0 )


class Log:

  def __init__( self, tab: ttk.Frame ) -> None:
    self.tab = tab

  def setup( self ):
    self.txtLog = ScrolledText( self.tab, width=150, height=12, state="normal" )
    self.txtLog.pack( anchor="nw", padx=4, pady=4 )


class ClipParticipants:

  def __init__( self, state: AppState, tab: ttk.Frame ) -> None:
    self.state = state
    self.tab = tab

  def setup( self ):
    self.participantTree = ttk.Treeview(
        self.tab,
        columns=( "shirt", "first", "last", "role" ),
        show="headings",
    )
    for column, heading, width in (
        ( "shirt", "Shirt", 70 ),
        ( "first", "First name", 180 ),
        ( "last", "Last name", 220 ),
        ( "role", "Role", 180 ),
    ):
      self.participantTree.heading( column, text=heading )
      self.participantTree.column( column, width=width, anchor="w" )
    self.participantTree.place( x=0, y=24, width=600, height=160 )
    scrollbar = ttk.Scrollbar( self.tab, orient="vertical", command=self.participantTree.yview )
    scrollbar.place( x=600, y=24, height=160 )
    self.participantTree.configure( yscrollcommand=scrollbar.set )
    self.refresh()

  def refresh( self ):
    self.participantTree.delete( *self.participantTree.get_children() )
    if self.state.db is None or self.state.curClipID <= 0:
      return

    for participant in listClipParticipants( self.state.db, self.state.curClipID ):
      self.participantTree.insert(
          "",
          "end",
          iid=str( participant.person_id.id ),
          values=(
              participant.shirt_number if participant.shirt_number is not None else "",
              participant.person_id.first_name,
              participant.person_id.last_name,
              participant.role.replace( "_", " " ),
          ),
      )


class Configuration:

  def __init__( self, parent: tk.Tk, state: AppState, on_change=None, crops_frame: ttk.LabelFrame | None = None ):

    self.parent = parent
    self.appState: AppState = state
    self.crops_frame = crops_frame or ttk.LabelFrame( parent, text="Crops" )

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

    #style = ttk.Style()
    #style.configure( "Red.TFrame", background="red" )
    self.root = ttk.LabelFrame( master=self.parent, borderwidth=5, relief='groove', text="Config" ) #, style="Red.TFrame" )
    self.root.place( x=50, y=690 + 56 )

    self.nbControl = ttk.Notebook( self.root, width=600, height=190 )
    self.tabImagePreview = ImagePreview( self.createTab( "Image Preview" ) )
    self.tabHomographyData = homographyData( self.appState, self.createTab( "Homography Data" ) )
    self.tabImageOptions = ImageOptionsUI( self.appState, self.createTab( "Image Options" ), on_change )
    self.tabLog = Log( self.createTab( "Log" ) )
    self.tabClipParticipants = ClipParticipants( self.appState, self.createTab( "Clip Participants" ) )
    self.tabTracks = Tracks( self.appState, self.createTab( "Tracks" ), self.crops_frame )
    self.nbControl.pack( expand=1, fill='both' )
    self.allTabs = [ self.tabHomographyData, self.tabImageOptions, self.tabImagePreview, self.tabLog, self.tabClipParticipants, self.tabTracks ]
    self.nbControl.bind( "<<NotebookTabChanged>>", self.onTabChanged )

    for tab in self.allTabs:
      tab.setup()

  def onTabChanged( self, _event=None ):
    if self.nbControl.select() == str( self.tabClipParticipants.tab ):
      self.tabClipParticipants.refresh()

  def createTab( self, text ):
    newTab = ttk.Frame( self.nbControl )
    self.nbControl.add( newTab, text=text )
    return newTab
