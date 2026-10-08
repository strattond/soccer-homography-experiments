import queue
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import duckdb
from PIL import Image, ImageTk

from soccer_homography.appState import AppState
from soccer_homography.data import ParticipationRole, Track, roles
from soccer_homography.data import Person as TrackPerson
from soccer_homography.db import (
    ClipTrackDB,
    PersonParticipation,
    listClipParticipants,
    listClipTracks,
    upsertClipTrack,
)
from soccer_homography.inference.abstractions import (
    AbstractInferenceModel,
    IdentificationImageResult,
)
from soccer_homography.inference.clip_model import ClipImageResult, ClipRoleClassifier
from soccer_homography.inference.crop_inference import (
    DEFAULT_IDENTIFICATION_PROMPT,
    CropInferenceJobMessage,
    CropInferenceWorker,
    mostLikelyRole,
    roleVoteCounts,
)
from soccer_homography.inference.vlm_model import MoondreamVLM
from soccer_homography.log import logger
from soccer_homography.ui.config.crop_worker import (
    CropCache,
    CropExtractionWorker,
    CropJobMessage,
    CropSet,
)


class Tracks:

  def __init__(
      self,
      appState: AppState,
      tab: ttk.Frame,
      crops_frame: ttk.LabelFrame,
      on_frame_select: Callable[ [ int ], None ] | None = None,
      prompt_provider: Callable[ [], str ] | None = None,
      prompt_saver: Callable[ [ str ], bool ] | None = None,
      model_provider: Callable[ [], str ] | None = None,
      role_change: Callable[ [], None ] | None = None,
      on_track_changed: Callable[ [ int | None ], None ] | None = None
  ) -> None:
    self.appState = appState
    self.tab = tab
    self.cropsFrame = crops_frame
    self.peopleByLabel: dict[ str, PersonParticipation ] = {}
    self.personLabels: dict[ int, str ] = {}
    self.person_options: tuple[ str, ...] = ( "<Unknown>",)
    self.editor: ttk.Combobox | None = None
    self.cropPreviewImage: ImageTk.PhotoImage | None = None
    self.cropIdentificationResults: dict[ int, dict[ int, IdentificationImageResult ] ] = {}
    self.cropResults: queue.Queue[ CropJobMessage ] = queue.Queue()
    self.cropGeneration = 0
    self.selTrackID: int | None = None
    self.refreshing = False
    self.cropCache: CropCache = {}
    self.cropCacheClipID: int | None = None
    self.frameSelectCallback: Callable[ [ int ], None ] | None = on_frame_select
    self.promptProvider = prompt_provider
    self.promptSaver = prompt_saver
    self.modelProvider = model_provider
    self.on_role_change = role_change
    self.on_track_changed = on_track_changed

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
    self.cropStatus.place( x=8, y=6 )
    self.cropProgress = ttk.Progressbar( self.cropsFrame, mode="determinate", length=180 )
    self.cropProgress.place( x=8, y=30 )
    self.cropProgress.place_forget()
    self.cropTable = ttk.Treeview(
        self.cropsFrame,
        columns=( "frame", "guess" ),
        show="headings",
        selectmode="browse",
    )
    self.cropTable.heading( "frame", text="Frame" )
    self.cropTable.heading( "guess", text="Identified guess" )
    self.cropTable.column( "frame", width=75, anchor="w" )
    self.cropTable.column( "guess", width=245, anchor="w" )
    self.cropTable.place( x=8, y=54, width=330, height=158 )
    cropScrollbar = ttk.Scrollbar( self.cropsFrame, orient="vertical", command=self.cropTable.yview )
    cropScrollbar.place( x=338, y=54, height=158 )
    self.cropTable.configure( yscrollcommand=cropScrollbar.set )
    self.cropTable.bind( "<<TreeviewSelect>>", self.onCropSelected )
    self.cropPreview = ttk.Label( self.cropsFrame, text="Select a crop", anchor="center", relief="sunken" )
    self.cropPreview.place( x=350, y=54, width=286, height=158 )
    self.cropPreviewCaption = ttk.Label( self.cropsFrame, text="", anchor="center" )
    self.cropPreviewCaption.place( x=350, y=214, width=286, height=20 )
    self.cropWorker: CropExtractionWorker | None = None
    self.cropInferenceWorker: CropInferenceWorker | None = None
    self.inferenceModelVLM: AbstractInferenceModel = MoondreamVLM()
    self.inferenceModelClip: AbstractInferenceModel = ClipRoleClassifier()
    self.inferenceResults: queue.Queue[ CropInferenceJobMessage ] = queue.Queue()
    self.inferenceGeneration = 0
    self.vlmButton: tk.Button | None = None
    self.cropsButtonEnabled = False
    self.refresh()
    self.tab.after( 50, self.pollCropResults )
    self.tab.after( 100, self.pollVLMResults )

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

    # Ensure we sort it by Track ID rather than order of creation
    trackData = sorted( self.appState.tracks.items(), key=lambda frame: ( frame[ 0 ] ) )

    for i, ( key, track ) in enumerate( trackData ):
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

    options = self.person_options if column == "#3" else tuple( role.replace( "_", " " ) for role in roles )
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
          # Use person's role if available, but if it's unknown, use the track role.  This happens after
          # a quick VLM look and assignment, followed by allocating a person
          role = participant.role if participant.role != "unknown" else track.role
      else:
        role = next( role for role in roles if role.replace( "_", " " ) == selection )
        person_id = track.numId()

      if not self.persistAssignment( track.id, person_id, role ):
        self.refresh()
        return
      track.person = person
      track.role = role
      self.refresh()
      self.tblTrackData.selection_set( str( track.id ) )
      self.renderSelectedCrops( track )
      if self.on_role_change is not None:
        self.on_role_change()

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
    self.updateVLMButtonState()
    self.renderSelectedCrops( track )
    if self.on_track_changed is not None:
      self.on_track_changed( track_id )

  def renderSelectedCrops( self, track: Track | None ) -> None:
    self.clearCropImages()
    if track is None:
      if self.cropWorker is None:
        self.cropStatus.config( text="" )
      return
    if self.cropCacheClipID != self.appState.curClipID:
      if self.cropWorker is None:
        self.cropStatus.config( text="Person assigned; crops are not cached for this track." if self.hasKnownPerson( track ) else "Press Crops to collect samples for unknown tracks." )
      return
    crops = self.cropCache.get( track.id )
    if crops is None:
      if self.cropWorker is None:
        self.cropStatus.config( text="Person assigned; crops are not cached for this track." if self.hasKnownPerson( track ) else "Press Crops to collect samples for unknown tracks." )
      return
    if not crops:
      self.cropStatus.config( text="No valid crops found for this track." )
      return
    self.cropStatus.config( text=f"Track {track.id}: {len(crops)} cached crops" )
    self.updateVLMButtonState()
    self.displayCrops( track.id, crops )

  def collectCrops( self ) -> None:
    if self.appState.curClipID <= 0 or not self.appState.videoFile:
      messagebox.showerror( "Crops unavailable", "Load a clip before collecting crops.", parent=self.tab )
      return
    self.refreshPeople()
    self.loadClipTrackAssignments()
    self.cancelCropJob()
    self.cancelVLMJob()
    self.cropCache.clear()
    self.cropIdentificationResults.clear()
    self.updateVLMButtonState()
    unknown_tracks = [ ( track.id, sorted( track.boxes, key=lambda box: box.frame ) ) for track in self.appState.tracks.values() if not self.hasKnownPerson( track ) and track.boxes ]
    if not unknown_tracks:
      self.renderSelectedCrops( None )
      self.cropStatus.config( text="No unknown tracks with bounding boxes." )
      return

    self.clearCropImages()
    self.cropCacheClipID = self.appState.curClipID
    self.cropGeneration += 1
    generation = self.cropGeneration
    self.cropProgress.configure( maximum=len( unknown_tracks ), value=0 )
    self.cropProgress.place( x=8, y=30 )
    self.cropStatus.config( text=f"Collecting crops for {len(unknown_tracks)} unknown tracks..." )
    self.cropCacheClipID = self.appState.curClipID
    self.cropWorker = CropExtractionWorker(
        generation,
        self.appState.videoFile,
        unknown_tracks,
        self.cropResults,
        clip_id=self.appState.curClipID,
    )
    self.cropWorker.start()

  def cancelCropJob( self ) -> None:
    if self.cropWorker is not None:
      self.cropWorker.cancel()
      self.cropWorker = None

  def setVLMButton( self, button: tk.Button ) -> None:
    self.vlmButton = button
    self.updateVLMButtonState()

  def updateVLMButtonState( self, crops_enabled: bool | None = None ) -> None:
    if crops_enabled is not None:
      self.cropsButtonEnabled = crops_enabled
    has_crops = ( self.selTrackID is not None and self.cropCacheClipID == self.appState.curClipID and bool( self.cropCache.get( self.selTrackID, [] ) ) )
    enabled = self.cropsButtonEnabled and has_crops and self.cropInferenceWorker is None
    if self.vlmButton is not None:
      self.vlmButton.config( state=tk.NORMAL if enabled else tk.DISABLED )

  def setPromptProvider(
      self,
      provider: Callable[ [], str ],
      saver: Callable[ [ str ], bool ],
  ) -> None:
    self.promptProvider = provider
    self.promptSaver = saver

  def runVLM( self ) -> None:
    if self.selTrackID is None:
      messagebox.showinfo( "VLM role suggestion", "Select a track with cached crops first.", parent=self.tab )
      return
    crops = self.cropCache.get( self.selTrackID, [] )
    if self.cropCacheClipID != self.appState.curClipID or not crops:
      messagebox.showinfo( "VLM role suggestion", "Collect crops before requesting a role suggestion.", parent=self.tab )
      self.updateVLMButtonState()
      return
    model_name = self.modelProvider() if self.modelProvider is not None else "VLM"
    if model_name == "Clip":
      prompt = ""
      model = self.inferenceModelClip
    elif model_name == "VLM":
      prompt = self.promptProvider() if self.promptProvider is not None else DEFAULT_IDENTIFICATION_PROMPT
      if not prompt:
        messagebox.showerror( "VLM role prompt", "Enter a prompt in the Image Options tab before running the VLM.", parent=self.tab )
        return
      if self.promptSaver is not None and not self.promptSaver( prompt ):
        return
      model = self.inferenceModelVLM
    else:
      messagebox.showerror( "Identification model unavailable", f"Unsupported identification model: {model_name}", parent=self.tab )
      return

    self.inferenceGeneration += 1
    self.cropInferenceWorker = CropInferenceWorker(
        self.inferenceGeneration,
        self.appState.curClipID,
        self.selTrackID,
        list( crops ),
        prompt,
        model,
        self.inferenceResults,
    )
    self.activeIdentifier = model_name
    self.cropStatus.config( text=f"Analyzing Track {self.selTrackID} crops with {model_name}..." )
    self.updateVLMButtonState()
    self.cropInferenceWorker.start()

  def onClipLoaded( self ) -> None:
    self.cancelCropJob()
    self.cancelVLMJob()
    self.cropGeneration += 1
    self.cropCache.clear()
    self.cropIdentificationResults.clear()
    self.cropCacheClipID = self.appState.curClipID if self.appState.curClipID > 0 else None
    self.selTrackID = None
    self.cropProgress.stop()
    self.cropProgress.place_forget()
    self.cropStatus.config( text="Click Crops to collect samples for unknown tracks." )
    self.clearCropImages()
    self.updateVLMButtonState()
    self.refresh()

  def shutdown( self ) -> None:
    self.cancelCropJob()
    self.cancelVLMJob()
    self.cropGeneration += 1

  def cancelVLMJob( self ) -> None:
    if self.cropInferenceWorker is not None:
      self.cropInferenceWorker.cancel()
      self.cropInferenceWorker = None
    self.inferenceGeneration += 1
    self.updateVLMButtonState()

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
        self.updateVLMButtonState()
        self.cropStatus.config( text=f"Crops cached for {len(self.cropCache)} unknown tracks." )
        selected = self.appState.tracks.get( self.selTrackID ) if self.selTrackID is not None else None
        self.renderSelectedCrops( selected )
    self.tab.after( 50, self.pollCropResults )

  def pollVLMResults( self ) -> None:
    if not self.tab.winfo_exists():
      return
    while True:
      try:
        message = self.inferenceResults.get_nowait()
      except queue.Empty:
        break
      if message.generation != self.inferenceGeneration or message.clip_id != self.appState.curClipID:
        continue
      if message.kind == "progress":
        self.cropStatus.config( text=f"Analyzing Track {message.track_id} with {self.activeIdentifier}: "
                                f"crop {message.completed} / {message.total}" )
      elif message.kind == "answer" and message.image_result is not None:
        self.updateCropResultLabel( message.track_id, message.image_result )
        self.cropStatus.config( text=f"Track {message.track_id}, frame {message.image_result.frame_number}: "
                                f"{self.formatIdentificationGuess( message.image_result )}" )
      elif message.kind == "error":
        self.cropInferenceWorker = None
        self.updateVLMButtonState()
        self.cropStatus.config( text=f"{self.activeIdentifier} role analysis failed." )
        if message.error is not None:
          logger.error( f"{self.activeIdentifier} role analysis failed: {message.error}" )
          messagebox.showerror( f"{self.activeIdentifier} role analysis failed", str( message.error ), parent=self.tab )
      elif message.kind == "done":
        self.cropInferenceWorker = None
        self.updateVLMButtonState()
        answers = message.answers or []
        vote_counts = message.vote_counts or roleVoteCounts( answers )
        role = message.role if message.vote_counts is not None else mostLikelyRole( vote_counts )
        report = self.formatIdentificationReport( role, vote_counts, answers )
        if role is None:
          messagebox.showinfo(
              f"{self.activeIdentifier} role votes",
              report,
              parent=self.tab,
          )
          self.cropStatus.config( text=f"{self.activeIdentifier} did not produce a unique role for Track {message.track_id}." )
        else:
          apply_role = messagebox.askyesno(
              f"{self.activeIdentifier} role votes",
              f"Track {message.track_id}\n\n{report}\n\nApply and save the most likely role?",
              parent=self.tab,
          )
          if apply_role:
            track = self.appState.tracks.get( message.track_id )
            if track is None:
              continue
            if self.persistAssignment( track.id, track.numId(), role ):
              track.role = role
              self.refresh()
              if self.tblTrackData.exists( str( track.id ) ):
                self.tblTrackData.selection_set( str( track.id ) )
              self.cropStatus.config( text=f"Saved {self.activeIdentifier} role {role.replace( '_', ' ' )} for Track {track.id}." )
          else:
            self.cropStatus.config( text=f"{self.activeIdentifier} most likely role: {role.replace( '_', ' ' )} "
                                    f"for Track {message.track_id}." )
    self.tab.after( 100, self.pollVLMResults )

  @staticmethod
  def formatIdentificationGuess( result: IdentificationImageResult ) -> str:
    guess = result.role.replace( "_", " " ) if result.role is not None else "no role guess"
    if isinstance( result, ClipImageResult ):
      return f"{guess} ({result.confidence:.0%} confidence)"
    return guess

  @staticmethod
  def formatIdentificationReport(
      role: ParticipationRole | None,
      counts: dict[ ParticipationRole, int ],
      answers: list[ IdentificationImageResult ],
  ) -> str:
    max_count = max( counts.values(), default=0 )
    if role is not None:
      result = f"Most likely: {role.replace( '_', ' ' )} ({max_count} vote(s))"
    elif max_count:
      tied_roles = [ name.replace( "_", " " ) for name, count in counts.items() if count == max_count ]
      result = f"Most likely: tie ({max_count} votes each for {', '.join( tied_roles )})"
    else:
      result = "Most likely: no role could be identified"
    votes = "\n".join( f"{name.replace( '_', ' ' )}: {counts.get( name, 0 )}" for name in roles )
    clip_confidences = [ answer.confidence for answer in answers if isinstance( answer, ClipImageResult ) and answer.role == role and answer.confidence is not None ]
    if role is not None and clip_confidences:
      confidence = sum( clip_confidences ) / len( clip_confidences )
      result += f"\nMean CLIP confidence for most likely role: {confidence:.0%}"
    crop_results = "\n".join( f"Frame {answer.frame_number}: {Tracks.formatIdentificationGuess( answer )}" for answer in answers )
    return f"{result}\n\nVotes:\n{votes}\n\nCrop results:\n{crop_results}"

  def updateCropResultLabel( self, track_id: int, result: IdentificationImageResult ) -> None:
    self.cropIdentificationResults.setdefault( track_id, {} )[ result.frame_number ] = result
    if track_id != self.selTrackID or not self.cropTable.exists( str( result.frame_number ) ):
      return
    self.cropTable.item(
        str( result.frame_number ),
        values=( result.frame_number, self.formatIdentificationGuess( result ) ),
    )
    if self.cropTable.selection() == ( str( result.frame_number ),):
      self.cropPreviewCaption.config( text=f"Frame {result.frame_number} - {self.formatIdentificationGuess( result )}" )

  def displayCrops( self, track_id: int, crops: CropSet ) -> None:
    results = self.cropIdentificationResults.get( track_id, {} )
    for frame_number, _ in crops:
      result = results.get( frame_number )
      guess = self.formatIdentificationGuess( result ) if result is not None else "Not analyzed"
      self.cropTable.insert(
          "",
          tk.END,
          iid=str( frame_number ),
          values=( frame_number, guess ),
      )

  def onCropSelected( self, _event=None ) -> None:
    selected = self.cropTable.selection()
    if not selected or self.selTrackID is None:
      return
    frame_number = int( selected[ 0 ] )
    crop = next(
        ( image for frame, image in self.cropCache.get( self.selTrackID, [] ) if frame == frame_number ),
        None,
    )
    if crop is None:
      return
    self.cropPreviewImage = ImageTk.PhotoImage( Image.fromarray( crop ) )
    self.cropPreview.config( image=self.cropPreviewImage, text="" )
    result = self.cropIdentificationResults.get( self.selTrackID, {} ).get( frame_number )
    guess = self.formatIdentificationGuess( result ) if result is not None else "Not analyzed"
    self.cropPreviewCaption.config( text=f"Frame {frame_number} - {guess}" )
    self.selectCropFrame( frame_number )

  def selectCropFrame( self, frame_number: int ) -> None:
    if self.frameSelectCallback is not None:
      self.frameSelectCallback( frame_number )

  def clearCropImages( self ) -> None:
    self.cropTable.delete( *self.cropTable.get_children() )
    self.cropPreviewImage = None
    self.cropPreview.config( image="", text="Select a crop" )
    self.cropPreviewCaption.config( text="" )
