import queue
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

import duckdb
from PIL import Image, ImageTk

from soccer_homography.appState import AppState
from soccer_homography.data import ParticipationRole, Track, TrackSegment, roles
from soccer_homography.db import (
  PersonParticipation,
  PersonParticipationDB,
  TrackSegmentDB,
  listClipParticipants,
  replaceTrackSegments,
  upsertPersonParticipation,
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
      on_track_changed: Callable[ [ int | None ], None ] | None = None,
      crop_count_provider: Callable[ [], int ] | None = None,
  ) -> None:
    self.appState = appState
    self.tab = tab
    self.cropsFrame = crops_frame
    self.peopleByLabel: dict[ str, PersonParticipation ] = {}
    self.personLabels: dict[ int, str ] = {}
    self.person_options: tuple[ str, ...] = ( "<Unknown>",)
    self.current_frame = 0
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
    self.cropCountProvider = crop_count_provider

  def setup( self ) -> None:
    colNames = [ "Track ID", "Num Frames", "Person" ]
    colWidths = [ 90, 110, 380 ]
    self.tblTrackData = ttk.Treeview(
        self.tab,
        columns=colNames,
        show="headings",
    )
    self.tblTrackData.place( x=0, y=0, width=600, height=184 )
    scrollbar = ttk.Scrollbar( self.tab, orient="vertical", command=self.tblTrackData.yview )
    scrollbar.place( x=600, y=0, height=184 )
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
      label = f"{person.first_name} {person.last_name}".strip()
      if participant.shirt_number is not None:
        label += f" (#{participant.shirt_number})"
      self.peopleByLabel[ label ] = participant
      self.personLabels[ person.id ] = label
    self.person_options = ( "<Unknown>", *self.peopleByLabel.keys() )

  def personLabel( self, track: Track, frame: int | None = None ) -> str:
    segment = track.segmentAt( self.current_frame if frame is None else frame )
    if segment is None:
      return "<No segment>"
    if segment.person_id is None:
      return "<Unknown>"
    return self.personLabels.get( segment.person_id, f"Person #{segment.person_id}" )

  def hasKnownPerson( self, track: Track, frame: int | None = None ) -> bool:
    segment = track.segmentAt( self.current_frame if frame is None else frame )
    return segment is not None and segment.person_id in self.personLabels

  def refresh( self ) -> None:
    self.refreshPeople()
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
          values=( str( key ), str( len( track.boxes ) ), self.personLabel( track ) ),
          tags=( tag,),
      )
    if selected_id is not None and self.tblTrackData.exists( selected_id ):
      self.tblTrackData.selection_set( selected_id )
      self.tblTrackData.focus( selected_id )
    self.refreshing = False
    if selected_id is None or not self.tblTrackData.exists( selected_id ):
      self.selectionChanged( None )

  def updateCurrentFrame( self, frame: int ) -> None:
    self.current_frame = frame
    if not hasattr( self, "tblTrackData" ):
      return
    for item_id in self.tblTrackData.get_children():
      track = self.appState.tracks.get( int( item_id ) )
      if track is None:
        continue
      values = list( self.tblTrackData.item( item_id, "values" ) )
      if len( values ) == 3:
        values[ 2 ] = self.personLabel( track )
        self.tblTrackData.item( item_id, values=values )
    if self.selTrackID is not None:
      self.updateVLMButtonState()

  def selectTrack( self, track_id: int ) -> None:
    track = self.appState.tracks.get( track_id )
    if track is None:
      return
    item_id = str( track_id )
    if not self.tblTrackData.exists( item_id ):
      self.refresh()
    if self.tblTrackData.exists( item_id ):
      self.tblTrackData.selection_set( item_id )
      self.tblTrackData.focus( item_id )
      self.tblTrackData.see( item_id )
      self.selectionChanged( track )

  def editCell( self, event ):
    row_id = self.tblTrackData.identify_row( event.y )
    column = self.tblTrackData.identify_column( event.x )
    if not row_id or column != "#3":
      return None
    bbox = self.tblTrackData.bbox( row_id, column )
    if not bbox:
      return "break"

    track = self.appState.tracks.get( int( row_id ) )
    if track is None:
      return "break"
    segment = track.segmentAt( self.current_frame )
    if segment is None:
      return "break"

    self.tblTrackData.selection_set( row_id )
    self.tblTrackData.focus( row_id )
    if self.editor is not None:
      self.editor.destroy()

    options = self.person_options
    self.editor = ttk.Combobox( self.tab, state="normal", values=options )
    x, y, width, height = bbox
    self.editor.place( x=self.tblTrackData.winfo_x() + x, y=self.tblTrackData.winfo_y() + y, width=width, height=height )
    current = self.personLabel( track )
    self.editor.set( current if current in options else options[ 0 ] )

    def apply_selection( _event=None ) -> None:
      if self.editor is None:
        return
      selection = self.editor.get()
      participant = self.peopleByLabel.get( selection )
      if participant is None:
        matches = [
            label for label in self.peopleByLabel
            if selection.casefold() in label.casefold()
        ]
        if len( matches ) == 1:
          selection = matches[ 0 ]
          participant = self.peopleByLabel[ selection ]
      if participant is None and selection != "<Unknown>":
        return
      self.editor.destroy()
      self.editor = None
      if participant is None:
        person_id = None
      else:
        person_id = participant.person_id.id
      revised_segments = [
          TrackSegment( item.frame_start, item.frame_end, person_id if item is segment else item.person_id )
          for item in track.segments
      ]
      if not self.persistSegments( track.id, revised_segments ):
        self.refresh()
        return
      track.segments = revised_segments

      self.refresh()
      self.tblTrackData.selection_set( str( track.id ) )
      self.renderSelectedCrops( track )
      if self.on_role_change is not None:
        self.on_role_change()

    self.editor.bind( "<<ComboboxSelected>>", apply_selection )
    self.editor.bind( "<Return>", apply_selection )
    self.editor.bind( "<KeyRelease>", self.filterPersonOptions )
    self.editor.bind( "<FocusOut>", lambda _event: self.closeEditor() )
    self.editor.focus_set()
    return "break"

  def filterPersonOptions( self, event ) -> None:
    if self.editor is None or event.keysym in ( "Up", "Down", "Left", "Right", "Return", "Escape" ):
      return
    text = self.editor.get()
    matches = tuple(
        label for label in self.person_options
        if text.casefold() in label.casefold()
    )
    self.editor.configure( values=matches or self.person_options )

  def updateParticipantRole( self, participant: PersonParticipation, role: ParticipationRole ) -> bool:
    person_id = participant.person_id.id
    if self.appState.db is None:
      messagebox.showerror( "Track update failed", "The database connection is unavailable.", parent=self.tab )
      return False
    try:
      upsertPersonParticipation(
          self.appState.db,
          PersonParticipationDB(
              match_id=participant.match_id.id,
              person_id=person_id,
              shirt_number=participant.shirt_number,
              role=role,
              is_placeholder=participant.is_placeholder,
          ),
      )
    except ( duckdb.Error, RuntimeError, ValueError ) as error:
      messagebox.showerror( "Track update failed", f"Could not update role for person {person_id}: {error}", parent=self.tab )
      logger.error( f"Could not update participation role for person {person_id}: {error}" )
      return False
    participant.role = role
    return True

  def persistSegments( self, track_id: int, segments: list[ TrackSegment ] ) -> bool:
    if self.appState.db is None or self.appState.curClipID <= 0:
      messagebox.showerror( "Track update failed", "Load a registered clip before updating track segments.", parent=self.tab )
      return False
    try:
      replaceTrackSegments(
          self.appState.db,
          self.appState.curClipID,
          track_id,
          [
              TrackSegmentDB(
                  clip_id=self.appState.curClipID,
                  track_id=track_id,
                  person_id=segment.person_id,
                  frame_start=segment.frame_start,
                  frame_end=segment.frame_end,
              )
              for segment in segments
          ],
      )
    except ( duckdb.Error, RuntimeError, ValueError ) as error:
      messagebox.showerror( "Track update failed", f"Could not save segments for track {track_id}: {error}", parent=self.tab )
      logger.error( f"Could not save segments for track {track_id}: {error}" )
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
        self.cropStatus.config( text="Assigned segment; crops are not cached." if self.hasKnownPerson( track ) else "Press Crops to collect samples for unassigned segments." )
      return
    crops = self.cropsForTrack( track.id )
    if not crops:
      if self.cropWorker is None:
        self.cropStatus.config( text="No crops cached for this track's segments. Press Crops to collect unassigned segments." )
      return
    self.cropStatus.config( text=f"Track {track.id}: {len(crops)} cached segment crops" )
    self.updateVLMButtonState()
    self.displayCrops( track.id, crops )

  def cropsForTrack( self, track_id: int ) -> CropSet:
    return sorted(
        (
            crop
            for ( cached_track_id, _segment_start ), crops in self.cropCache.items()
            if cached_track_id == track_id
            for crop in crops
        ),
        key=lambda crop: crop[ 0 ],
    )

  def cropsForSegment( self, track_id: int, segment_start: int ) -> CropSet:
    return self.cropCache.get( ( track_id, segment_start ), [] )

  def collectCrops( self ) -> None:
    if self.appState.curClipID <= 0 or not self.appState.videoFile:
      messagebox.showerror( "Crops unavailable", "Load a clip before collecting crops.", parent=self.tab )
      return
    self.refreshPeople()
    self.cancelCropJob()
    self.cancelVLMJob()
    self.cropCache.clear()
    self.cropIdentificationResults.clear()
    self.updateVLMButtonState()
    unknown_segments = []
    for track in self.appState.tracks.values():
      for segment in track.segments:
        if segment.person_id is not None:
          continue
        boxes = sorted(
            (
                box for box in track.boxes
                if segment.frame_start <= box.frame <= segment.frame_end
            ),
            key=lambda box: box.frame,
        )
        if boxes:
          unknown_segments.append( ( track.id, segment.frame_start, boxes ) )
    if not unknown_segments:
      self.renderSelectedCrops( None )
      self.cropStatus.config( text="No unassigned track segments with bounding boxes." )
      return

    self.clearCropImages()
    self.cropCacheClipID = self.appState.curClipID
    self.cropGeneration += 1
    generation = self.cropGeneration
    self.cropProgress.configure( maximum=len( unknown_segments ), value=0 )
    self.cropProgress.place( x=8, y=30 )
    self.cropStatus.config( text=f"Collecting crops for {len(unknown_segments)} unassigned track segments..." )
    self.cropCacheClipID = self.appState.curClipID
    self.cropWorker = CropExtractionWorker(
        generation,
        self.appState.videoFile,
      unknown_segments,
        self.cropResults,
        clip_id=self.appState.curClipID,
        max_crops=self.cropCountProvider() if self.cropCountProvider is not None else 6,
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
    track = self.appState.tracks.get( self.selTrackID ) if self.selTrackID is not None else None
    segment = track.segmentAt( self.current_frame ) if track is not None else None
    has_crops = (
        track is not None
        and segment is not None
        and self.cropCacheClipID == self.appState.curClipID
        and bool( self.cropsForSegment( track.id, segment.frame_start ) )
    )
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
    track = self.appState.tracks.get( self.selTrackID )
    segment = track.segmentAt( self.current_frame ) if track is not None else None
    crops = self.cropsForSegment( self.selTrackID, segment.frame_start ) if segment is not None else []
    if self.cropCacheClipID != self.appState.curClipID or not crops:
      messagebox.showinfo( "VLM role suggestion", "Collect crops for the selected track segment before requesting a role suggestion.", parent=self.tab )
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
        self.cropStatus.config( text=f"Crops cached for {len(self.cropCache)} unassigned track segments." )
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
            segment = track.segmentAt( self.current_frame )
            person_id = segment.person_id if segment is not None else None
            participant = next(
                ( item for item in self.peopleByLabel.values() if item.person_id.id == person_id ),
                None,
            )
            if participant is None:
              messagebox.showinfo(
                  "Participant required",
                  "Assign a match participant or placeholder before saving a role.",
                  parent=self.tab,
              )
              continue
            if self.updateParticipantRole( participant, role ):
              self.refresh()
              if self.tblTrackData.exists( str( track.id ) ):
                self.tblTrackData.selection_set( str( track.id ) )
              if self.on_role_change is not None:
                self.on_role_change()
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
        ( image for frame, image in self.cropsForTrack( self.selTrackID ) if frame == frame_number ),
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
