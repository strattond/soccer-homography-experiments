import queue
import tkinter as tk
from tkinter import messagebox, ttk

import duckdb
from PIL import Image, ImageTk

from soccer_homography.appState import AppState
from soccer_homography.dataTypes import ParticipationRole, Track
from soccer_homography.dataTypes import Person as TrackPerson
from soccer_homography.db import ClipTrackDB, PersonParticipation, listClipParticipants, listClipTracks, upsertClipTrack
from soccer_homography.log import logger
from soccer_homography.ui.config.crop_worker import CropCache, CropExtractionWorker, CropJobMessage, CropSet


class Tracks:

  roles: tuple[ ParticipationRole, ...] = (
      "home_player",
      "home_goalkeeper",
      "away_player",
      "away_goalkeeper",
      "referee",
      "unknown",
  )

  def __init__( self, appState: AppState, tab: ttk.Frame, crops_frame: ttk.LabelFrame ) -> None:
    self.appState = appState
    self.tab = tab
    self.cropsFrame = crops_frame
    self.peopleByLabel: dict[ str, PersonParticipation ] = {}
    self.personLabels: dict[ int, str ] = {}
    self.person_options: tuple[ str, ...] = ( "<Unknown>",)
    self.editor: ttk.Combobox | None = None
    self.cropImages: list[ ImageTk.PhotoImage ] = []
    self.cropLabels: list[ ttk.Label ] = []
    self.cropResults: queue.Queue[ CropJobMessage ] = queue.Queue()
    self.cropGeneration = 0
    self.selTrackID: int | None = None
    self.refreshing = False
    self.cropCache: CropCache = {}
    self.cropCacheClipID: int | None = None

  def setup( self ) -> None:
    colNames = [ "Track ID", "Num Frames", "Person", "Role" ]
    colWidths = [ 90, 110, 260, 120 ]
    self.tblTrackData = ttk.Treeview(
        self.tab,
        columns=colNames,
        show="headings",
    )
    self.tblTrackData.place( x=0, y=24, width=580, height=160 )
    scrollbar = ttk.Scrollbar( self.tab, orient="vertical", command=self.tblTrackData.yview )
    scrollbar.place( x=580, y=24, height=160 )
    self.tblTrackData.configure( yscrollcommand=scrollbar.set )

    for i, ( col, width ) in enumerate( zip( colNames, colWidths ) ):
      self.tblTrackData.heading( col, text=col )
      self.tblTrackData.column( col, width=width, anchor="w" )

    self.tblTrackData.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblTrackData.tag_configure( "evenrow", background="#993333", foreground="white" )
    self.tblTrackData.bind( "<Double-1>", self.editCell )
    self.tblTrackData.bind( "<<TreeviewSelect>>", self.onTrackSelected )

    self.cropStatus = ttk.Label( self.cropsFrame, text="Click Crops to collect samples for unknown tracks." )
    self.cropStatus.place( x=8, y=26 )
    self.cropProgress = ttk.Progressbar( self.cropsFrame, mode="determinate", length=180 )
    self.cropProgress.place( x=8, y=50 )
    self.cropProgress.place_forget()
    self.cropWorker: CropExtractionWorker | None = None
    self.refresh()
    self.tab.after( 50, self.pollCropResults )

  def refreshPeople( self ) -> None:
    self.peopleByLabel.clear()
    self.personLabels.clear()
    if self.appState.db is None or self.appState.curClipID <= 0:
      self.person_options = ( "<Unknown>",)
      return

    for participant in listClipParticipants( self.appState.db, self.appState.curClipID ):
      person = participant.person_id
      label = f"{person.first_name} {person.last_name} (#{participant.shirt_number})"
      self.peopleByLabel[ label ] = participant
      self.personLabels[ person.id ] = label
    self.person_options = ( "<Unknown>", *self.peopleByLabel.keys() )

  def loadClipTrackAssignments( self ) -> None:
    if self.appState.db is None or self.appState.curClipID <= 0:
      return
    associations = listClipTracks( self.appState.db, self.appState.curClipID )
    for track_id, track in self.appState.tracks.items():
      association = associations.get( track_id )
      if association is None:
        continue
      track.role = association.role
      if association.person_id is None:
        track.person = None
      else:
        participant = next(
            ( item for item in self.peopleByLabel.values() if item.person_id.id == association.person_id ),
            None,
        )
        if participant is None:
          track.person = association.person_id
        else:
          person = participant.person_id
          track.person = TrackPerson( id=person.id, name=f"{person.first_name} {person.last_name}" )

  def personLabel( self, track: Track ) -> str:
    person = track.person
    if isinstance( person, TrackPerson ):
      return self.personLabels.get( person.id, person.name )
    if isinstance( person, int ):
      return self.personLabels.get( person, f"Person #{person}" )
    return "<Unknown>"

  def hasKnownPerson( self, track: Track ) -> bool:
    if isinstance( track.person, TrackPerson ):
      return True
    return isinstance( track.person, int ) and track.person in self.personLabels

  def refresh( self ) -> None:
    self.refreshPeople()
    self.loadClipTrackAssignments()
    selected = self.tblTrackData.selection()
    selected_id = selected[ 0 ] if selected else None
    self.refreshing = True
    self.tblTrackData.delete( *self.tblTrackData.get_children() )

    for i, ( key, track ) in enumerate( self.appState.tracks.items() ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      self.tblTrackData.insert(
          "",
          tk.END,
          iid=str( key ),
          values=( str( key ), str( len( track.boxes ) ), self.personLabel( track ), track.role.replace( "_", " " ) ),
          tags=( tag,),
      )
    if selected_id is not None and self.tblTrackData.exists( selected_id ):
      self.tblTrackData.selection_set( selected_id )
      self.tblTrackData.focus( selected_id )
    self.refreshing = False
    if selected_id is None or not self.tblTrackData.exists( selected_id ):
      self.selectionChanged( None )

  def editCell( self, event ):
    row_id = self.tblTrackData.identify_row( event.y )
    column = self.tblTrackData.identify_column( event.x )
    if not row_id or column not in ( "#3", "#4" ):
      return None
    bbox = self.tblTrackData.bbox( row_id, column )
    if not bbox:
      return "break"

    track = self.appState.tracks.get( int( row_id ) )
    if track is None:
      return "break"

    self.tblTrackData.selection_set( row_id )
    self.tblTrackData.focus( row_id )
    if self.editor is not None:
      self.editor.destroy()

    options = self.person_options if column == "#3" else tuple( role.replace( "_", " " ) for role in self.roles )
    self.editor = ttk.Combobox( self.tab, state="readonly", values=options )
    x, y, width, height = bbox
    self.editor.place( x=self.tblTrackData.winfo_x() + x, y=self.tblTrackData.winfo_y() + y, width=width, height=height )
    current = self.personLabel( track ) if column == "#3" else track.role.replace( "_", " " )
    self.editor.set( current if current in options else options[ 0 ] )

    def apply_selection( _event=None ) -> None:
      if self.editor is None:
        return
      selection = self.editor.get()
      self.editor.destroy()
      self.editor = None
      person_id: int | None
      role: ParticipationRole = track.role
      person: TrackPerson | int | None = track.person
      if column == "#3":
        participant = self.peopleByLabel.get( selection )
        if participant is None:
          person = None
          person_id = None
        else:
          db_person = participant.person_id
          person = TrackPerson( id=db_person.id, name=f"{db_person.first_name} {db_person.last_name}" )
          person_id = db_person.id
          role = participant.role
      else:
        role = next( role for role in self.roles if role.replace( "_", " " ) == selection )
        person_id = track.numId()

      if not self.persistAssignment( track.id, person_id, role ):
        self.refresh()
        return
      track.person = person
      track.role = role
      self.refresh()
      self.tblTrackData.selection_set( str( track.id ) )
      self.renderSelectedCrops( track )

    self.editor.bind( "<<ComboboxSelected>>", apply_selection )
    self.editor.bind( "<FocusOut>", lambda _event: self.closeEditor() )
    self.editor.focus_set()
    return "break"

  def persistAssignment( self, track_id: int, person_id: int | None, role: ParticipationRole ) -> bool:
    if self.appState.db is None or self.appState.curClipID <= 0:
      messagebox.showerror( "Track update failed", "Load a registered clip before assigning its tracks.", parent=self.tab )
      return False
    try:
      upsertClipTrack(
          self.appState.db,
          ClipTrackDB(
              clip_id=self.appState.curClipID,
              track_id=track_id,
              person_id=person_id,
              role=role,
          ),
      )
    except ( duckdb.Error, RuntimeError, ValueError ) as error:
      messagebox.showerror( "Track update failed", f"Could not save track {track_id}: {error}", parent=self.tab )
      logger.error( f"Could not save track {track_id} association: {error}" )
      return False
    return True

  def closeEditor( self ) -> None:
    if self.editor is not None:
      self.editor.destroy()
      self.editor = None

  def onTrackSelected( self, _event=None ) -> None:
    if self.refreshing:
      return
    selected = self.tblTrackData.selection()
    track = self.appState.tracks.get( int( selected[ 0 ] ) ) if selected else None
    self.selectionChanged( track )

  def selectionChanged( self, track: Track | None ) -> None:
    track_id = track.id if track is not None else None
    if track_id == self.selTrackID:
      return
    self.selTrackID = track_id
    self.renderSelectedCrops( track )

  def renderSelectedCrops( self, track: Track | None ) -> None:
    self.clearCropImages()
    if track is None or self.hasKnownPerson( track ):
      if self.cropWorker is None:
        self.cropStatus.config( text="" if track is None else "Person assigned; crops remain cached." )
      return
    if self.cropCacheClipID != self.appState.curClipID:
      if self.cropWorker is None:
        self.cropStatus.config( text="Press Crops to collect samples for unknown tracks." )
      return
    crops = self.cropCache.get( track.id )
    if crops is None:
      if self.cropWorker is None:
        self.cropStatus.config( text="Press Crops to collect samples for unknown tracks." )
      return
    if not crops:
      self.cropStatus.config( text="No valid crops found for this track." )
      return
    self.cropStatus.config( text=f"Track {track.id}: {len(crops)} cached crops" )
    self.displayCrops( crops )

  def collectCrops( self ) -> None:
    if self.appState.curClipID <= 0 or not self.appState.videoFile:
      messagebox.showerror( "Crops unavailable", "Load a clip before collecting crops.", parent=self.tab )
      return
    self.refreshPeople()
    self.loadClipTrackAssignments()
    unknown_tracks = [ ( track.id, sorted( track.boxes, key=lambda box: box.frame ) ) for track in self.appState.tracks.values() if not self.hasKnownPerson( track ) and track.boxes ]
    if not unknown_tracks:
      self.cropCache.clear()
      self.renderSelectedCrops( None )
      self.cropStatus.config( text="No unknown tracks with bounding boxes." )
      return

    self.cancelCropJob()
    self.cropCache.clear()
    self.clearCropImages()
    self.cropCacheClipID = self.appState.curClipID
    self.cropGeneration += 1
    generation = self.cropGeneration
    self.cropProgress.configure( maximum=len( unknown_tracks ), value=0 )
    self.cropProgress.place( x=8, y=50 )
    self.cropStatus.config( text=f"Collecting crops for {len(unknown_tracks)} unknown tracks..." )
    self.cropCacheClipID = self.appState.curClipID
    self.cropWorker = CropExtractionWorker(
        generation,
        self.appState.videoFile,
        unknown_tracks,
        self.cropResults,
    )
    self.cropWorker.start()

  def cancelCropJob( self ) -> None:
    if self.cropWorker is not None:
      self.cropWorker.cancel()
      self.cropWorker = None

  def onClipLoaded( self ) -> None:
    self.cancelCropJob()
    self.cropGeneration += 1
    self.cropCache.clear()
    self.cropCacheClipID = self.appState.curClipID if self.appState.curClipID > 0 else None
    self.selTrackID = None
    self.cropProgress.stop()
    self.cropProgress.place_forget()
    self.cropStatus.config( text="Click Crops to collect samples for unknown tracks." )
    self.clearCropImages()
    self.refresh()

  def shutdown( self ) -> None:
    self.cancelCropJob()
    self.cropGeneration += 1

  def pollCropResults( self ) -> None:
    if not self.tab.winfo_exists():
      return
    while True:
      try:
        message = self.cropResults.get_nowait()
      except queue.Empty:
        break
      if message.generation != self.cropGeneration:
        continue
      if message.kind == "progress":
        if message.total > 0:
          self.cropProgress.configure( maximum=message.total, value=message.completed )
        status = "Planning crop frames..." if message.stage == "planned frames" else "Collecting crops"
        if message.stage != "planned frames":
          status += f": {message.completed} / {message.total} frames"
        self.cropStatus.config( text=status )
      elif message.kind == "error":
        self.cropWorker = None
        self.cropProgress.place_forget()
        self.cropStatus.config( text="Could not collect track crops." )
        if message.error is not None:
          logger.error( f"Could not collect track crops: {message.error}" )
          messagebox.showerror( "Crop collection failed", str( message.error ), parent=self.tab )
      elif message.kind == "done":
        self.cropWorker = None
        self.cropProgress.configure( value=message.completed )
        self.cropProgress.place_forget()
        self.cropCache = message.crops or {}
        self.cropStatus.config( text=f"Crops cached for {len(self.cropCache)} unknown tracks." )
        selected = self.appState.tracks.get( self.selTrackID ) if self.selTrackID is not None else None
        self.renderSelectedCrops( selected )
    self.tab.after( 50, self.pollCropResults )

  def displayCrops( self, crops: CropSet ) -> None:
    for slot, ( frame_number, crop ) in enumerate( crops ):
      photo = ImageTk.PhotoImage( Image.fromarray( crop ) )
      self.cropImages.append( photo )
      label = ttk.Label( self.cropsFrame, image=photo, text=f"Frame {frame_number}", compound="top" )
      label.grid( row=0, column=slot, padx=2, pady=( 58, 0 ), sticky="n" )
      self.cropLabels.append( label )

  def clearCropImages( self ) -> None:
    for label in self.cropLabels:
      label.destroy()
    self.cropLabels.clear()
    self.cropImages.clear()
