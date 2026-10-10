import json
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import duckdb
import pyarrow as pa
from PIL import Image, ImageTk

from soccer_homography.appState import AppState
from soccer_homography.data import (
    CHUNK_SIZE,
    BoundingBox,
    Homography,
    SelectionPoint,
    Track,
    TrackData,
    VideoData,
)
from soccer_homography.db import (
    addGenericPeopleToMatch,
    deleteClipTracks,
    deleteTrackSegments,
    deleteTrackingChunks,
    getCameraByID,
    getClipByID,
    getClipHomography,
    getMatchByID,
    getVideoByID,
    initDB,
    listClips,
    readDetectionChunks,
    readTrackingChunks,
    saveClipHomography,
    writeBatchDetections,
    writeBatchTracking,
)
from soccer_homography.db.chunk_writer import AsyncChunkWriter
from soccer_homography.encoder import BaseVideoEncoder
from soccer_homography.log import logger, logging
from soccer_homography.SportsTracker import (
    Command,
    CommandType,
    Output,
    OutputType,
    SportsTracker,
)
from soccer_homography.ui import (
    Configuration,
    DataMaintenance,
    FrameMinimap,
    HomographyUI,
    LabelledSpinBox,
    LivePreview,
    MainCanvasController,
    ProgressBarETA,
    RadarCanvas,
    Slider,
)
from soccer_homography.ui.config.crop_worker import deleteTrackCrops
from soccer_homography.ui.frameminimap import TrackingType


class App:

  def __init__( self, root: tk.Tk, appState: AppState ):
    self.root = root
    self.curTrackID: int | None = None
    self.root.title( "Homography Mapper" )
    self.root.geometry( "1920x1080" )
    self.root.resizable( True, True )
    self.appState: AppState = appState
    self.appState.db = initDB()
    self.tracking: SportsTracker | None = None
    self.ui_queue = queue.Queue()
    self.detection_writer = AsyncChunkWriter(
        "detection",
        writeBatchDetections,
        lambda delay, callback: self.root.after( delay, callback ),
        lambda callback_id: self.root.after_cancel( callback_id ),
        lambda chunk_id, error: self.on_chunk_write_error( "detection", chunk_id, error ),
        lambda records: sum( len( boxes ) for boxes in records.values() ),
    )
    self.tracking_writer = AsyncChunkWriter(
        "tracking",
        writeBatchTracking,
        lambda delay, callback: self.root.after( delay, callback ),
        lambda callback_id: self.root.after_cancel( callback_id ),
        lambda chunk_id, error: self.on_chunk_write_error( "tracking", chunk_id, error ),
        lambda records: sum( len( track.boxes ) for track in records ),
    )
    self.pendingDetectionChunks: set[ int ] = set()
    self.pendingTrackingChunks: set[ int ] = set()

    # Initialize variables

    # Create widgets
    self.createWidgets()
    root.protocol( "WM_DELETE_WINDOW", self.on_close )

  def on_close( self ):
    if self.tracking is not None and self.tracking.thread is not None:
      self.tracking.in_queue.put( Command( CommandType.STOP ) )
      self.tracking.thread.join( timeout=30 )
      if self.tracking is not None and self.tracking.thread.is_alive():
        print( "Forcibly terminating" )
    self.detection_writer.shutdown()
    self.tracking_writer.shutdown()
    self.tabData.tabTracks.shutdown()
    if self.appState.db is not None:
      self.appState.db.close()

    self.root.destroy()

  def createWidgets( self ):
    """Create and place all widgets"""

    # imagePreview
    self.imagePreview = tk.Canvas( self.root, bg="#ffffff", highlightthickness=1, highlightbackground="#d1d5db" )
    self.imagePreview.place( x=50, y=20, width=1280, height=720 )

    # Tabular data + line detection options
    self.crops = ttk.LabelFrame( self.root, text="Crops" )
    self.crops.place( x=680, y=690 + 56, width=650, height=240 )
    self.tabData = Configuration(
        parent=self.root,
        state=self.appState,
        on_change=self.onOptionsChange,
        crops_frame=self.crops,
        on_frame_select=self.onFrameSelect,
        on_role_changed=self.onRoleChanged,
        on_track_changed=self.onTrackChanged
    )

    # radarMap
    self.radarMap = tk.Canvas( self.root, bg="#dfdfdf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.radarMap.place( x=1460, y=50, width=420 + 20, height=272 + 20 )

    # lblRadar
    self.lblRadar = tk.Label( self.root, text="Bird's eye view (point matcher)", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblRadar.place( x=1460, y=20, width=250, height=24 )

    self.uiHomography = HomographyUI( self.root, self.appState, 1460, 700, self.playIt, self.homoReplace, self.saveClipHomography, self.loadHomographyFromDisk )

    self.createWidgetsDetection( 1460, 740 )
    self.createWidgetsTrack( 1460, 780 )
    self.createWidgetsSource( 1460, 820 )
    self.createWidgetsFrameControl( 1460, 900 )
    self.createWidgetsMisc( 1460, 900 )

  def openDataMaintenance( self ):
    if self.appState.db is None:
      self.appState.db = initDB()
    DataMaintenance( self.root, self.appState.db )

  def addGenericPeopleToCurrentClip( self ) -> None:
    clip_id = self.appState.curClipID
    conn = self.appState.db
    if clip_id <= 0 or conn is None:
      messagebox.showerror( "Add generic people", "Load a registered clip first.", parent=self.root )
      return

    try:
      clip = getClipByID( conn, clip_id )
      if clip is None:
        messagebox.showerror( "Add generic people", f"Clip {clip_id} was not found in the database.", parent=self.root )
        return
      participations = addGenericPeopleToMatch( conn, clip.match_id )
    except ( duckdb.Error, RuntimeError, ValueError ) as error:
      logger.exception( f"Could not add generic people to clip {clip_id}." )
      messagebox.showerror( "Could not add generic people", str( error ), parent=self.root )
      return

    self.tabData.tabClipParticipants.refresh()
    messagebox.showinfo(
        "Generic people added",
        f"Assigned {len( participations )} generic people to the current clip.",
        parent=self.root,
    )

  def createWidgetsDetection( self, left: int, top: int ):
    # lblDetectAction
    self.lblDetectAction = tk.Label( self.root, text="Detection", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblDetectAction.place( x=left, y=top, width=100, height=24 )

    # btnRunYoloDetection
    self.btnYoloOneFrame = tk.Button( self.root, text="Frame", font=( "Arial", 10 ), command=self.cmdYoloOneFrame, state=tk.DISABLED )
    self.btnYoloOneFrame.place( x=left + 110, y=top, width=52, height=28 )

    # btnRunYoloVidDetection
    self.btnYoloRange = tk.Button( self.root, text="Range", font=( "Arial", 10 ), command=self.cmdYoloRange, state=tk.DISABLED )
    self.btnYoloRange.place( x=left + 162, y=top, width=52, height=28 )

  def createWidgetsTrack( self, left: int, top: int ):
    # lblDetectAction
    self.lblTrackAction = tk.Label( self.root, text="Tracking", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblTrackAction.place( x=left, y=top, width=100, height=24 )

    # btnRunYoloVidDetection
    self.btnTrackRange = tk.Button( self.root, text="Range", font=( "Arial", 10 ), command=self.cmdTrackRange, state=tk.DISABLED )
    self.btnTrackRange.place( x=left + 110, y=top, width=52, height=28 )
    self.btnCrops = tk.Button( self.root, text="Crops", font=( "Arial", 10 ), command=self.tabData.tabTracks.collectCrops, state=tk.DISABLED )
    self.btnCrops.place( x=left + 162, y=top, width=52, height=28 )
    self.btnVLM = tk.Button( self.root, text="VLM", font=( "Arial", 10 ), command=self.tabData.tabTracks.runVLM, state=tk.DISABLED )
    self.btnVLM.place( x=left + 214, y=top, width=52, height=28 )
    self.btnHeatmap = tk.Button( self.root, text="Heatmap", font=( "Arial", 10 ), command=self.runHeatmap, state=tk.DISABLED )
    self.btnHeatmap.place( x=left + 266, y=top, width=78, height=28 )
    self.btnDeleteTracks = tk.Button( self.root, text="Delete", font=( "Arial", 10 ), command=self.deleteTracksForCurrentClip, state=tk.DISABLED )
    self.btnDeleteTracks.place( x=left + 344, y=top, width=60, height=28 )
    self.tabData.tabTracks.setVLMButton( self.btnVLM )

  def createWidgetsSource( self, left: int, top: int ):
    # lblDetectAction
    self.lblSourceAction = tk.Label( self.root, text="Source", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblSourceAction.place( x=left, y=top, width=100, height=24 )

    # Load Video from file or database
    self.btnSourceDB = tk.Button( self.root, text="Clip", font=( "Arial", 10 ), command=self.cmdSourceClip )
    self.btnSourceDB.place( x=left + 110, y=top, width=52, height=28 )
    tk.Button( self.root, text="Data Maintenance", font=( "Arial", 10 ), command=self.openDataMaintenance ).place( x=left + 162, y=top, width=120, height=28 )
    tk.Button( self.root, text="Add Generic", font=( "Arial", 10 ), command=self.addGenericPeopleToCurrentClip ).place( x=left + 286, y=top, width=90, height=28 )

  def createWidgetsFrameControl( self, left: int, top: int ):

    self.minimap = FrameMinimap( master=self.root, totalFrames=0, on_frame_select=self.onFrameSelect )
    self.minimap.place( x=1390, y=20, width=30, height=720 )
    # sliderVideoFrame
    self.sldVideoFrame = Slider( from_=0, to=100, command=self.cmdUpdateVideoFrame, root=self.root, x=left + 110, y=top, width=300, height=24 )

    # lblVideoFrameSlider
    self.lblVideoFrameSlider = tk.Label( self.root, text="Video Frame", fg="#000000", font=( "Arial", 10 ), anchor="center" )
    self.lblVideoFrameSlider.place( x=left, y=top, width=100, height=24 )

    self.minFrame = LabelledSpinBox( root=self.root, from_=0, to=100, x=left + 90, y=top + 40, width=100, height=24, offset=80, label="Start" )
    self.maxFrame = LabelledSpinBox( root=self.root, from_=0, to=100, x=left + 90, y=top + 68, width=100, height=24, offset=80, label="Finish", initValue=100 )

    self.btnResetZoom = tk.Button( self.root, text="Reset Zoom", font=( "Arial", 10 ), command=self.cmdResetZoom )
    self.btnResetZoom.place( x=left + 210, y=top + 40, width=88, height=28 )
    self.btnResetPan = tk.Button( self.root, text="Reset Pan", font=( "Arial", 10 ), command=self.cmdResetPan )
    self.btnResetPan.place( x=left + 298, y=top + 40, width=88, height=28 )

  def createWidgetsMisc( self, left: int, top: int ):

    self.radarMapController = RadarCanvas(
        self.radarMap,
        ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ),
        self.appState.cfg,
        self.appState.pitch,
        self.on_radar_click,
        self.on_radar_hover,
        self.on_radar_selection_move,
    )

    self.mainImageController = MainCanvasController(
        self.imagePreview,
        self.appState,
        self.on_main_click,
        self.on_main_hover,
        self.on_main_view_change,
        self.on_main_selection_move,
    )
    self.root.bind( "<Escape>", self.clearPendingMapping )
    self.livePreviewController = LivePreview( self.root, ( 1460, 400 ), ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ), self.appState, self.bumpIt )
    self.livePreviewController.setTrackSelectCallback( self.tabData.tabTracks.selectTrack )

    self.prgDetection = ProgressBarETA( root=self.root, x=left - 125, y=20, width=24, height=720 )
    self.prgHomography = ProgressBarETA( root=self.root, x=left - 100, y=20, width=24, height=720 )

  # ==========================================
  # Event Handlers - Implement your logic here
  # ==========================================

  def homoReplace( self ):
    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )
    self.redisplayHomographyData()
    self.checkButtonState()

  def playIt( self, encoder: BaseVideoEncoder | None ):
    min = self.minFrame.get()
    max = self.maxFrame.get()
    self.prgHomography.setRange( min, max )
    self.livePreviewController.play( min, max, encoder )

  def cmdSourceClip( self ):
    conn = self.appState.db
    if conn is None:
      conn = initDB()
      self.appState.db = conn
    clips = listClips( conn )
    if not clips:
      messagebox.showinfo( "No clips", "There are no clips registered in the database.", parent=self.root )
      return

    window = tk.Toplevel( self.root )
    window.title( "Select clip" )
    window.transient( self.root )
    window.grab_set()
    window.geometry( "800x400" )
    content = ttk.Frame( window )
    content.pack( fill="both", expand=True, padx=8, pady=8 )
    tree = ttk.Treeview( content, columns=( "id", "match", "camera", "sequence", "file" ), show="headings", selectmode="browse" )
    for column, heading, width in (
        ( "id", "Clip ID", 70 ),
        ( "match", "Match", 220 ),
        ( "camera", "Camera", 120 ),
        ( "sequence", "Sequence", 80 ),
        ( "file", "Video", 300 ),
    ):
      tree.heading( column, text=heading )
      tree.column( column, width=width, anchor="w" )
    vertical_scrollbar = ttk.Scrollbar( content, orient="vertical", command=tree.yview )
    horizontal_scrollbar = ttk.Scrollbar( content, orient="horizontal", command=tree.xview )
    tree.configure( yscrollcommand=vertical_scrollbar.set, xscrollcommand=horizontal_scrollbar.set )
    tree.grid( row=0, column=0, sticky="nsew" )
    vertical_scrollbar.grid( row=0, column=1, sticky="ns" )
    horizontal_scrollbar.grid( row=1, column=0, sticky="ew" )
    content.rowconfigure( 0, weight=1 )
    content.columnconfigure( 0, weight=1 )
    clips_by_id = { clip.id: clip for clip in clips }
    for clip in clips:
      video = getVideoByID( conn, clip.video_id )
      match = getMatchByID( conn, clip.match_id )
      camera = getCameraByID( conn, clip.camera_id )
      if video is None or match is None or camera is None:
        logger.warning( f"Skipping clip {clip.id}: its video, match, or camera record is missing." )
        continue
      date = match.date.strftime( "%Y-%m-%d %H:%M" ) if match.date is not None else ""
      tree.insert(
          "",
          "end",
          iid=str( clip.id ),
          values=( clip.id, f"{date} {match.home} v {match.away}", camera.name, clip.sequence, video.file ),
      )

    def load_selected( _event=None ):
      selected = tree.selection()
      if not selected:
        return
      clip = clips_by_id.get( int( selected[ 0 ] ) )
      video = getVideoByID( conn, clip.video_id ) if clip is not None else None
      if clip is not None and video is not None and self.loadSourceVideo( video.file, clip.id ):
        window.destroy()

    tree.bind( "<Double-1>", load_selected )
    ttk.Button( window, text="Load selected clip", command=load_selected ).pack( pady=( 0, 8 ) )

  def loadSourceVideo( self, filename: str, clip_id: int ) -> bool:
    capture = cv2.VideoCapture( filename )
    if not capture.isOpened():
      capture.release()
      messagebox.showerror( "Unable to open video", f"OpenCV could not open:\n{filename}", parent=self.root )
      return False
    had_loaded_clip = self.appState.curClipID > 0
    if self.appState.cap is not None:
      self.appState.cap.release()
    vidData = VideoData( capture )
    self.appState.videoFile = filename
    self.appState.cap = capture
    self.appState.curClipID = clip_id
    self.appState.curHomographyID = None
    self.appState.boxes.clear()
    self.appState.tracks.clear()
    self.appState.framesProcessed = 0
    self.appState.detectChunk = 0
    self.appState.trackChunk = 0
    self.appState.frameRate = int( vidData.fps )
    self.pendingDetectionChunks.clear()
    self.pendingTrackingChunks.clear()
    if had_loaded_clip:
      self.appState.data = Homography()
    if self.appState.db is not None:
      try:
        stored_homography = getClipHomography( self.appState.db, clip_id )
      except ( duckdb.Error, ValueError ) as error:
        stored_homography = None
        messagebox.showerror( "Load Homography failed", str( error ), parent=self.root )
      if stored_homography is not None:
        self.appState.data.load_dict( stored_homography.payload )
        self.appState.curHomographyID = stored_homography.homography_id
    try:
      self.appState.boxes.update( readDetectionChunks( clip_id ) )
      self.appState.tracks.update( readTrackingChunks( clip_id ) )
    except ( OSError, ValueError, pa.ArrowException ) as error:
      logger.exception( f"Could not load tracking data for clip {clip_id}." )
      messagebox.showerror( "Load tracking data failed", str( error ), parent=self.root )
      self.appState.boxes.clear()
      self.appState.tracks.clear()
    self.tabData.tabTracks.onClipLoaded()
    vidData = VideoData( capture )
    self.sldVideoFrame.setMax( max( 0, vidData.frames - 1 ) )
    self.minFrame.setMax( max( 0, vidData.frames - 1 ) )
    self.maxFrame.setMax( max( 0, vidData.frames - 1 ) )
    self.mainImageController.load( capture, vidData )
    self.mainImageController.setFrame( 0, self.curTrackID )
    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )
    self.minimap.updateTotalFrames( vidData.frames )
    processed_frames = set( self.appState.boxes ) | { box.frame for track in self.appState.tracks.values() for box in track.boxes }
    if ( len( processed_frames ) > 0 ):
      self.minimap.markFramesAsDone( list( processed_frames ) )
    self.minimap.setCurrentFrame( 0 )
    if self.appState.tracks and self.appState.data.hom4k is not None:
      self.refreshHomographyData( 0 )
    self.tabData.tabClipParticipants.refresh()
    self.checkButtonState()
    return True

  def hasHomography( self ) -> bool:
    return len( self.appState.data.world_pts ) >= 4

  def cmdUpdateVideoFrame( self, value ):
    frame = int( value )
    self.mainImageController.setFrame( frame, self.curTrackID )
    self.minimap.setCurrentFrame( frame )
    self.livePreviewController.updateMappings( self.appState.tracks, frame )

  def onFrameSelect( self, frame: int ) -> None:
    self.sldVideoFrame.setValue( frame )

  def onRoleChanged( self ) -> None:
    self.tabData.tabTracks.refreshPeople()
    self.tabData.tabTracks.loadClipTrackAssignments()
    self.livePreviewController.updateMappings( self.appState.tracks, self.mainImageController.frame_num )

  def onTrackChanged( self, trackID: int | None ) -> None:
    if trackID is None:
      self.minimap.clear( TrackingType.CUR_TRACK )
    else:
      trackData = self.appState.tracks.get( trackID, None )
      if trackData is not None:
        frames = [ box.frame for box in trackData.boxes ]
        self.minimap.clearFrames( TrackingType.CUR_TRACK )
        self.minimap.markFramesAsDone( frames, TrackingType.CUR_TRACK )
    self.minimap.redraw()
    self.curTrackID = trackID
    self.mainImageController.updateTracks( self.appState.tracks, self.mainImageController.frame_num, self.curTrackID )

  def checkButtonState( self ):
    cappable = self.appState.cap is not None and self.appState.cap.isOpened()
    homoable = self.hasHomography() and len( self.appState.tracks.items() ) > 0
    trackable = len( self.appState.boxes.items() ) > 0
    self.btnYoloOneFrame.config( state=tk.NORMAL if cappable else tk.DISABLED )
    self.btnYoloRange.config( state=tk.NORMAL if cappable else tk.DISABLED )
    self.btnTrackRange.config( state=tk.NORMAL if trackable else tk.DISABLED )
    self.btnDeleteTracks.config( state=tk.NORMAL if self.appState.curClipID > 0 else tk.DISABLED )
    self.btnCrops.config( state=tk.NORMAL if self.appState.tracks else tk.DISABLED )
    self.btnHeatmap.config( state=tk.NORMAL if self.appState.tracks else tk.DISABLED )
    self.tabData.tabTracks.updateVLMButtonState( bool( self.appState.tracks ) )
    self.uiHomography.setEnableStatus( self.hasHomography(), homoable, self.appState.curClipID > 0 )
    self.sldVideoFrame.setEnabled( cappable )

  def loadHomographyFromDisk( self ) -> None:
    if self.appState.curClipID <= 0:
      messagebox.showerror( "Load Homography", "Load a registered clip before loading its homography.", parent=self.root )
      return

    path = filedialog.askopenfilename( title="Select homography file" )
    if not path:
      return

    try:
      with open( path, "r" ) as f:
        data = json.load( f )
      self.appState.data.load_dict( data )
      self.appState.data.compute()
    except ( OSError, ValueError, TypeError, KeyError ) as error:
      logger.exception( f"Could not load homography from {path}." )
      messagebox.showerror( "Load Homography failed", str( error ), parent=self.root )
      return

    self.saveClipHomography()
    logger.info( f"Loaded and saved homography {self.appState.curHomographyID} for clip {self.appState.curClipID}" )
    self.homoReplace()

  def saveClipHomography( self ) -> None:
    if self.appState.db is None or self.appState.curClipID <= 0:
      messagebox.showerror( "Save Homography", "Load a registered clip before saving its homography.", parent=self.root )
      return
    try:
      saved = saveClipHomography(
          self.appState.db,
          self.appState.curClipID,
          self.appState.data.to_dict(),
          homography_id=self.appState.curHomographyID,
      )
    except ( duckdb.Error, RuntimeError, ValueError ) as error:
      logger.exception( "Could not save homography for the active clip." )
      messagebox.showerror( "Save Homography failed", str( error ), parent=self.root )
      return
    self.appState.curHomographyID = saved.homography_id
    logger.info( f"Saved homography {saved.homography_id} for clip {saved.clip_id}" )

  def setProgRange( self, prog: ttk.Progressbar, val: int, max: int ):
    prog[ 'value' ] = val
    prog[ 'maximum' ] = max

  def runYolo( self, minFrame, maxFrame ):
    # Step 1 - load the model
    self.allocateModelTracking()
    if self.tracking is not None:
      # Step 2 - do it
      # But clear out existing bounding data...
      for f in range( minFrame, maxFrame + 1 ):
        if f in self.appState.boxes:
          self.pendingDetectionChunks.add( f // CHUNK_SIZE )
          del self.appState.boxes[ f ]
      logger.info( f"Identifying frames {minFrame} to {maxFrame}" )
      self.prgDetection.setRange( 0, ( maxFrame-minFrame ) + 1 )
      self.prgHomography.setRange( 0, 0 )
      self.tracking.in_queue.put( Command( CommandType.PAUSE ) )
      self.tracking.in_queue.put( Command( CommandType.SET_CLIP_ID, payload=self.appState.curClipID ) )
      self.tracking.in_queue.put( Command( CommandType.RUN_BBOX, minFrame, maxFrame ) )
      self.tracking.in_queue.put( Command( CommandType.RESUME ) )
      self.prgDetection.start()
      self.root.after( 100, self.pollForUI )

  def runTracking( self, minFrame, maxFrame ):
    # Step 1 - load the model
    self.allocateModelTracking()
    if self.tracking is not None:
      # Step 2 - do it
      # But clear out existing homography data...
      for value in self.appState.tracks.values():
        retained = [ box for box in value.boxes if not minFrame <= box.frame <= maxFrame ]
        if len( retained ) != len( value.boxes ):
          self.pendingTrackingChunks.update( box.frame // CHUNK_SIZE for box in value.boxes if minFrame <= box.frame <= maxFrame )
          value.boxes = retained
        value.clearHomography()
      self.appState.tracks = { track_id: track for track_id, track in self.appState.tracks.items() if track.boxes }
      logger.info( f"Tracking frames {minFrame} to {maxFrame}" )
      self.prgDetection.setRange( 0, ( maxFrame-minFrame ) + 1 )
      self.prgHomography.setRange( 0, 0 )
      self.tracking.in_queue.put( Command( CommandType.PAUSE ) )
      sliced = dict[ int, list[ BoundingBox ] ]( ( k, v ) for k, v in self.appState.boxes.items() if minFrame <= k <= maxFrame )
      self.tracking.in_queue.put( Command( CommandType.SET_CLIP_ID, payload=self.appState.curClipID ) )
      self.tracking.in_queue.put( Command( CommandType.FEED_BBOX, payload=sliced ) )
      self.tracking.in_queue.put( Command( CommandType.RUN_TRACK, minFrame, maxFrame ) )
      self.tracking.in_queue.put( Command( CommandType.RESUME ) )
      self.prgDetection.start()
      self.root.after( 100, self.pollForUI )

  def cmdYoloOneFrame( self ):
    self.runYolo( self.mainImageController.frame_num, self.mainImageController.frame_num )

  def pollForUI( self ):
    data: Output | None = None
    pollDelay: int = 50
    try:
      if self.tracking is not None:
        data = self.tracking.out_queue.get_nowait()
      else:
        data = self.ui_queue.get_nowait()
    except queue.Empty:
      data = None

    if data is not None:
      if data.type == OutputType.BBOX and data.data is not None and isinstance( data.data, BoundingBox ):
        bbox = data.data
        if bbox.frame not in self.appState.boxes:
          self.appState.boxes[ bbox.frame ] = []
        self.appState.boxes[ bbox.frame ].append( bbox )
        self.pendingDetectionChunks.add( bbox.frame // CHUNK_SIZE )
        pollDelay = 0
      if data.type == OutputType.TRACK and data.data is not None and isinstance( data.data, TrackData ):
        track = data.data
        if track.tid not in self.appState.tracks:
          self.appState.tracks[ track.tid ] = Track( track.clip, track.tid )
        self.appState.tracks[ track.tid ].boxes.append( track.data )
        self.pendingTrackingChunks.add( track.data.frame // CHUNK_SIZE )
        pollDelay = 0
      elif data.type == OutputType.NEW_FRAME:
        self.appState.framesProcessed += 1
        self.prgDetection.tick()
        if self.tracking is not None:
          tMode = self.tracking.curMode
          if tMode == CommandType.RUN_BBOX and isinstance( data.data, int ):
            self.minimap.markFramesAsDone( [ data.data ] )
            self.minimap.redraw()
          if tMode == CommandType.RUN_BBOX and self.appState.framesProcessed % CHUNK_SIZE == 0:
            # Write out the saved data
            self.chunkDetections()
            logger.info( f"Writing chunk {self.appState.detectChunk}" )
            self.appState.detectChunk += 1
          elif tMode == CommandType.RUN_TRACK and self.appState.framesProcessed % CHUNK_SIZE == 0:
            # Write out the saved data
            self.chunkTracking()
            logger.info( f"Writing chunk {self.appState.trackChunk}" )
            self.appState.trackChunk += 1
        pollDelay = 0
      elif data.type == OutputType.COMPLETED:
        self.prgDetection.stop()
        self.chunkDetections()
        self.chunkTracking()
        self.refreshHomographyData( self.mainImageController.frame_num )
        self.mainImageController.updateBoundingBoxes( self.appState.boxes, self.mainImageController.frame_num )
        self.mainImageController.updateTracks( self.appState.tracks, self.mainImageController.frame_num, self.curTrackID )
        self.livePreviewController.updateMappings( self.appState.tracks, self.mainImageController.frame_num )
        self.checkButtonState()
        self.tabData.tabTracks.refresh()
        self.minimap.redraw()
        return
      elif data.type == OutputType.STOP:
        self.prgDetection.stop()
        return

    self.root.after( pollDelay, self.pollForUI )

  def cmdYoloRange( self ):
    self.runYolo( self.minFrame.get(), self.maxFrame.get() )

  def cmdTrackRange( self ):
    self.runTracking( self.minFrame.get(), self.maxFrame.get() )

  def runHeatmap( self ):
    # Reset the heat maps
    for h in self.appState.heatmaps.values():
      h.reset()
    # Now accumulate based on the track data
    frameRate = 1 / self.appState.frameRate if self.appState.frameRate > 0 else 0.2
    self.prgDetection.setRange( 0, len( self.appState.tracks ) )
    self.prgDetection.start()
    self.root.after( 100, self.pollForUI )
    logger.info( f"Generating heatmaps for {len(self.appState.tracks)} tracks" )
    for track in self.appState.tracks.values():
      for homog, box in zip( track.homog_smooth, track.boxes ):
        if homog is not None:
          self.appState.heatmaps[ track.role ].accumulate( int( homog.x ), int( homog.y ), frameRate )
      self.prgDetection.tick()
    for r, h in self.appState.heatmaps.items():
      heatmap = h.get_display_image( label=r )
      self.livePreviewController.updateHeatmap( r, heatmap )
    self.ui_queue.put( Output( OutputType.STOP ) )

  def deleteTracksForCurrentClip( self ) -> None:
    clip_id = self.appState.curClipID
    if clip_id <= 0 or self.appState.db is None:
      messagebox.showerror( "Delete tracks", "Load a registered clip before deleting its tracks.", parent=self.root )
      return
    tracker = self.tracking
    if tracker is not None and tracker.thread is not None:
      tracking_active = (
          tracker.thread.is_alive()
          and not tracker.stopped
          and tracker.curMode in ( CommandType.RUN_TRACK, CommandType.RUN_BBOX )
      )
      if tracking_active or not tracker.out_queue.empty():
        messagebox.showwarning( "Delete tracks", "Wait for tracking to finish before deleting tracks.", parent=self.root )
        return
    if not messagebox.askyesno(
        "Delete tracks",
        f"Delete all tracks for clip {clip_id} from memory, DuckDB, tracking parquet chunks, "
        "track segments, and cached crops?",
        parent=self.root,
    ):
      return

    try:
      self.tracking_writer.waitForPending()
      deleted_chunks = deleteTrackingChunks( clip_id )
      deleted_crops = deleteTrackCrops( clip_id )
      deleteClipTracks( self.appState.db, clip_id )
      deleteTrackSegments( self.appState.db, clip_id )
    except ( duckdb.Error, OSError, RuntimeError, ValueError ) as error:
      logger.exception( f"Could not delete tracks for clip {clip_id}." )
      messagebox.showerror( "Delete tracks failed", str( error ), parent=self.root )
      return

    self.appState.tracks.clear()
    self.pendingTrackingChunks.clear()
    self.appState.trackChunk = 0
    self.tabData.tabTracks.refresh()
    frame = self.mainImageController.frame_num
    self.mainImageController.updateTracks( self.appState.tracks, frame, self.curTrackID )
    self.livePreviewController.updateMappings( self.appState.tracks, frame )
    self.minimap.clear( TrackingType.CUR_TRACK )
    self.checkButtonState()
    logger.info(
        f"Deleted tracking data for clip {clip_id}, {deleted_chunks} parquet chunks, "
        f"and {deleted_crops} cached crops."
    )

  def chunkDetections( self ):
    for chunk_id in sorted( self.pendingDetectionChunks ):
      lo_frame = chunk_id * CHUNK_SIZE
      hi_frame = lo_frame + CHUNK_SIZE
      export = { frame: list( boxes ) for frame, boxes in self.appState.boxes.items() if lo_frame <= frame < hi_frame }
      self.detection_writer.submit( self.appState.curClipID, chunk_id, export )
    self.pendingDetectionChunks.clear()

  def chunkTracking( self ):
    for chunk_id in sorted( self.pendingTrackingChunks ):
      lo_frame = chunk_id * CHUNK_SIZE
      hi_frame = lo_frame + CHUNK_SIZE
      export = [ partial for track in self.appState.tracks.values() if ( partial := track.forExport( lo_frame, hi_frame ) ).boxes ]
      self.tracking_writer.submit( self.appState.curClipID, chunk_id, export )
    self.pendingTrackingChunks.clear()

  def on_chunk_write_error( self, data_type: str, chunk_id: int, error: Exception ) -> None:
    messagebox.showerror(
        f"{data_type.title()} write failed",
        f"Could not write {data_type} chunk {chunk_id}: {error}",
        parent=self.root,
    )

  def allocateModelTracking( self ):
    if self.tracking is not None and self.tracking.thread is not None and self.tracking.thread.is_alive():
      return
    self.tracking = SportsTracker( self.appState.mdlOpts, self.appState.videoFile )
    self.tracking.start()

  def mapping_check( self ):
    if self.appState.last_image_click is None or self.appState.sel_world_point is None:
      return

    # We have a map!  Construct a mapping pair
    self.appState.data.world_pts.append( self.appState.sel_world_point )
    self.appState.data.img_pts_4k.append( self.appState.last_image_click )
    self.appState.last_image_click = None
    self.appState.sel_world_point = None
    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )
    self.redisplayHomographyData()

  def on_main_click( self, x: int, y: int, point: SelectionPoint ):
    self.appState.last_image_click = point
    self.mapping_check()

  def on_main_hover( self, x: int, y: int ):
    pass

  def on_main_selection_move( self ):
    self.redisplayHomographyData()

  def cmdResetZoom( self ):
    self.mainImageController.resetZoom()

  def cmdResetPan( self ):
    self.mainImageController.resetTranslation()

  def on_main_view_change( self ):
    xf = self.mainImageController.transform

    self.tabData.tabImagePreview.refresh( xf )

    self.mainImageController.updateBoundingBoxes( self.appState.boxes, self.mainImageController.frame_num )
    self.mainImageController.updateTracks( self.appState.tracks, self.mainImageController.frame_num, self.curTrackID )
    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )

  def on_radar_click( self, x: int, y: int, point: SelectionPoint ):
    self.appState.sel_world_point = point
    self.mapping_check()

  def on_radar_hover( self, x: int, y: int ):
    pass

  def on_radar_selection_move( self ):
    self.redisplayHomographyData()

  def clearPendingMapping( self, _event=None ):
    self.appState.last_image_click = None
    self.appState.sel_world_point = None
    self.mainImageController.clearPendingMapping()
    self.radarMapController.clearPendingMapping()

  def onOptionsChange( self ):
    self.mainImageController.refreshHough()

  def redisplayHomographyData( self ):
    self.tabData.tabHomographyData.refresh()
    self.appState.data.compute()
    logger.info( "Clearing existing homography calculations" )
    for value in self.appState.tracks.values():
      value.clearHomography()
    self.refreshHomographyData()
    self.checkButtonState()

  def bumpIt( self ):
    self.prgHomography.tick()

  def refreshHomographyData( self, index: int = 0 ):
    # Now recalculate
    logger.info( f"Refreshing homography calculations for {len(self.appState.tracks.items())} tracks" )
    self.prgHomography.setRange( 0, len( self.appState.tracks.items() ) )
    for value in self.appState.tracks.values():
      value.refreshHomography( self.appState.data )
      self.root.after( 0, self.bumpIt )
    self.livePreviewController.updateMappings( self.appState.tracks, index )


def main() -> None:
  root = tk.Tk()
  appState = AppState()
  appState.pitch.padding = 10
  appState.pitch.makeEmptyPitch()
  logger.setLevel( logging.INFO )
  app = App( root, appState )
  root.mainloop()
