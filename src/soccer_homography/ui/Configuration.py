import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import duckdb

from soccer_homography.appState import AppState
from soccer_homography.data import ParticipationRole, roles
from soccer_homography.db import (
    PersonParticipation,
    PersonParticipationDB,
    listClipParticipants,
    upsertPersonParticipation,
)
from soccer_homography.log import logging
from soccer_homography.ui.config import (
    ImageOptionsUI,
    ImagePreview,
    Tracks,
    homographyData,
)
from soccer_homography.ui.config.modeloptions import ModelOptions


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

  def __init__( self, state: AppState, tab: ttk.Frame, on_role_changed=None ) -> None:
    self.state = state
    self.tab = tab
    self.on_role_changed = on_role_changed
    self.participants: dict[ int, PersonParticipation ] = {}
    self.editor: ttk.Combobox | None = None

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
    self.participantTree.bind( "<Double-1>", self.editRole )
    self.refresh()

  def refresh( self ):
    self.participantTree.delete( *self.participantTree.get_children() )
    self.participants.clear()
    if self.state.db is None or self.state.curClipID <= 0:
      return

    for participant in listClipParticipants( self.state.db, self.state.curClipID ):
      self.participants[ participant.person_id.id ] = participant
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

  def editRole( self, event ):
    row_id = self.participantTree.identify_row( event.y )
    column = self.participantTree.identify_column( event.x )
    if not row_id or column != "#4":
      return None
    bounds = self.participantTree.bbox( row_id, column )
    if not bounds:
      return "break"
    participant = self.participants.get( int( row_id ) )
    if participant is None:
      return "break"

    if self.editor is not None:
      self.editor.destroy()
    self.editor = ttk.Combobox(
        self.tab,
        state="readonly",
        values=tuple( role.replace( "_", " " ) for role in roles ),
    )
    x, y, width, height = bounds
    self.editor.place(
        x=self.participantTree.winfo_x() + x,
        y=self.participantTree.winfo_y() + y,
        width=width,
        height=height,
    )
    self.editor.set( participant.role.replace( "_", " " ) )

    def save_role( _event=None ) -> None:
      if self.editor is None:
        return
      selected_role = self.editor.get()
      self.editor.destroy()
      self.editor = None
      role: ParticipationRole = next(
          role for role in roles if role.replace( "_", " " ) == selected_role
      )
      if self.state.db is None:
        messagebox.showerror( "Participant update failed", "The database connection is unavailable.", parent=self.tab )
        return
      try:
        upsertPersonParticipation(
            self.state.db,
            PersonParticipationDB(
                match_id=participant.match_id.id,
                person_id=participant.person_id.id,
                shirt_number=participant.shirt_number,
                role=role,
                is_placeholder=participant.is_placeholder,
            ),
        )
      except ( duckdb.Error, RuntimeError, ValueError ) as error:
        messagebox.showerror(
            "Participant update failed",
            f"Could not update role for person {participant.person_id.id}: {error}",
            parent=self.tab,
        )
        return
      participant.role = role
      self.refresh()
      if self.on_role_changed is not None:
        self.on_role_changed()

    self.editor.bind( "<<ComboboxSelected>>", save_role )
    self.editor.bind( "<FocusOut>", lambda _event: self.closeEditor() )
    self.editor.focus_set()
    return "break"

  def closeEditor( self ) -> None:
    if self.editor is not None:
      self.editor.destroy()
      self.editor = None


class Configuration:

  def __init__( self, parent: tk.Tk, state: AppState, on_change=None, crops_frame: ttk.LabelFrame | None = None, on_frame_select=None, on_role_changed=None, on_track_changed=None ):

    self.parent = parent
    self.appState: AppState = state
    self.crops_frame = crops_frame or ttk.LabelFrame( parent, text="Crops" )

    # Build UI
    self.createLayout( on_change, on_frame_select, on_role_changed, on_track_changed )

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
  def createLayout( self, on_change, on_frame_select, on_role_changed, on_track_changed ):

    #style = ttk.Style()
    #style.configure( "Red.TFrame", background="red" )
    self.root = ttk.LabelFrame( master=self.parent, borderwidth=5, relief='groove', text="Config" )  #, style="Red.TFrame" )
    self.root.place( x=50, y=690 + 56 )

    self.nbControl = ttk.Notebook( self.root, width=600, height=190 )
    self.tabImagePreview = ImagePreview( self.createTab( "Image Preview" ) )
    self.tabLog = Log( self.createTab( "Log" ) )
    self.tabImageOptions = ImageOptionsUI( self.appState, self.createTab( "Image Options" ), on_change )
    self.tabModelOptions = ModelOptions( self.appState, self.createTab( "Model Options" ) )
    self.tabHomographyData = homographyData( self.appState, self.createTab( "Homography Data" ) )
    self.tabClipParticipants = ClipParticipants(
        self.appState,
        self.createTab( "Clip Participants" ),
        on_role_changed,
    )
    self.tabTracks = Tracks(
        self.appState, self.createTab( "Tracks" ), self.crops_frame, on_frame_select, self.tabModelOptions.getIdentificationPrompt, self.tabModelOptions.savePrompt,
        self.tabModelOptions.getIdentificationModel, on_role_changed, on_track_changed,
        crop_count_provider=self.tabModelOptions.getCropsPerSegment,
    )
    self.nbControl.pack( expand=1, fill='both' )
    self.allTabs = [ self.tabHomographyData, self.tabImageOptions, self.tabImagePreview, self.tabLog, self.tabClipParticipants, self.tabTracks, self.tabModelOptions ]
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
