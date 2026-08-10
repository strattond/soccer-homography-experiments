import queue
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from appState import AppState
from constants import CHUNK_SIZE
from dataTypes import RawTrackData, SelectionPoint, Track, VideoData
from db import writeBatch
from encoder import BaseVideoEncoder
from log import logger, logging
from SportsTracker import Command, CommandType, Output, OutputType, SportsTracker
from ui import (
  Configuration,
  HomographyUI,
  LabelledSpinBox,
  LivePreview,
  MainCanvasController,
  ProgressBarETA,
  RadarCanvas,
  Slider,
)


class App:

  def __init__( self, root: tk.Tk, appState: AppState ):
    self.root = root
    self.root.title( "Homography Mapper" )
    self.root.geometry( "1920x1080" )
    self.root.resizable( True, True )
    self.appState: AppState = appState
    self.tracking: SportsTracker | None = None

    # Initialize variables

    # Create widgets
    self._create_widgets()
    root.protocol( "WM_DELETE_WINDOW", self.on_close )

  def on_close( self ):
    if self.tracking is not None and self.tracking.thread is not None:
      self.tracking.in_queue.put( Command( CommandType.STOP ) )
      self.tracking.thread.join( timeout=30 )
      if self.tracking is not None and self.tracking.thread.is_alive():
        print( "Forcibly terminating" )

    self.root.destroy()

  def _create_widgets( self ):
    """Create and place all widgets"""

    # imagePreview
    self.imagePreview = tk.Canvas( self.root, bg="#ffffff", highlightthickness=1, highlightbackground="#d1d5db" )
    self.imagePreview.place( x=50, y=50, width=1280, height=720 )

    # Tabular data + line detection options
    self.tabData = Configuration( parent=self.root, state=self.appState, on_change=self.on_options_change )

    # radarMap
    self.radarMap = tk.Canvas( self.root, bg="#dfdfdf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.radarMap.place( x=1460, y=50, width=420 + 20, height=272 + 20 )

    # livePreview
    self.livePreview = tk.Canvas( self.root, bg="#bfbfbf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.livePreview.place( x=1460, y=400, width=420 + 20, height=272 + 20 )

    # lblRadar
    self.lblRadar = tk.Label( self.root, text="Bird's eye view (point matcher)", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblRadar.place( x=1460, y=20, width=250, height=24 )

    # lblLivePreview
    self.lblLivePreview = tk.Label( self.root, text="Live Preview", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblLivePreview.place( x=1460, y=350, width=100, height=24 )

    # lblHomography
    self.lblHomography = tk.Label( self.root, text="Homography Point Matcher", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblHomography.place( x=50, y=20, width=200, height=24 )

    self.uiHomography = HomographyUI( self.root, self.appState, 1460, 700, self.playIt, self.homoReplace )

    # lblDataAction
    self.lblDataAction = tk.Label( self.root, text="Data", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblDataAction.place( x=1460, y=740, width=100, height=24 )

    # btnDataLoad
    self.btnDataLoad = tk.Button( self.root, text="Load", font=( "Arial", 12 ), command=self.cmdDataLoad )
    self.btnDataLoad.place( x=1600, y=740, width=60, height=36 )

    # btnDataSave
    self.btnDataSave = tk.Button( self.root, text="Save", font=( "Arial", 12 ), command=self.cmdDataSave, state=tk.DISABLED )
    self.btnDataSave.place( x=1660, y=740, width=60, height=36 )

    # lblDetectAction
    self.lblDetectAction = tk.Label( self.root, text="Detection", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblDetectAction.place( x=1460, y=780, width=100, height=24 )

    # btnRunYoloDetection
    self.btnYoloOneFrame = tk.Button( self.root, text="Frame", font=( "Arial", 12 ), command=self.cmdYoloOneFrame, state=tk.DISABLED )
    self.btnYoloOneFrame.place( x=1600, y=780, width=60, height=36 )

    # btnRunYoloVidDetection
    self.btnYoloRange = tk.Button( self.root, text="Range", font=( "Arial", 12 ), command=self.cmdYoloRange, state=tk.DISABLED )
    self.btnYoloRange.place( x=1660, y=780, width=60, height=36 )

    # lblDetectAction
    self.lblSourceAction = tk.Label( self.root, text="Source", fg="#000000", font=( "Arial", 12 ), anchor="w" )
    self.lblSourceAction.place( x=1460, y=820, width=100, height=24 )

    # btnLoadVideo
    self.btnSourceVideo = tk.Button( self.root, text="Video", font=( "Arial", 12 ), command=self.cmdSourceVideo )
    self.btnSourceVideo.place( x=1660, y=820, width=60, height=36 )

    # sliderVideoFrame
    self.sldVideoFrame = Slider( from_=0, to=100, command=self.cmdUpdateVideoFrame, root=self.root, x=1650, y=900, width=200, height=24 )

    # lblVideoFrameSlider
    self.lblVideoFrameSlider = tk.Label( self.root, text="Video Frame", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblVideoFrameSlider.place( x=1460, y=900, width=100, height=24 )

    self.radarMapController = RadarCanvas(
        self.radarMap, ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ), self.appState.cfg, self.appState.pitch, self.on_radar_click, self.on_radar_hover
    )

    self.mainImageController = MainCanvasController( self.imagePreview, self.appState, self.on_main_click, self.on_main_hover, self.on_main_view_change )
    self.livePreviewController = LivePreview( self.livePreview, ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ), self.appState, self.bumpIt )

    self.prgDetection = ProgressBarETA( root=self.root, x=1350, y=50, width=24, height=720 )
    self.prgHomography = ProgressBarETA( root=self.root, x=1375, y=50, width=24, height=720 )

    self.minFrame = LabelledSpinBox( root=self.root, from_=0, to=100, x=1650, y=940, width=200, height=24, offset=190, label="Start" )
    self.maxFrame = LabelledSpinBox( root=self.root, from_=0, to=100, x=1650, y=965, width=200, height=24, offset=190, label="Finish", initValue=100 )

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

  def cmdDataLoad( self ):
    """
    Handle cmdLoadBB event
    TODO: Implement your logic here
    """
    pass

  def cmdDataSave( self ):
    filetypes = ( ( 'Saved match data', '*.json' ),)

    filename = filedialog.asksaveasfilename( title='Save Match Data', initialdir='.', filetypes=filetypes )
    if filename is not None:
      self.appState.save( filename )

  def cmdSourceVideo( self ):
    filetypes = ( ( 'Video files', [ '*.mp4', '*.mkv' ] ),)

    filename = filedialog.askopenfilename( title='Open video', initialdir='.', filetypes=filetypes )
    if filename is not None:
      if self.appState.cap is not None:
        self.appState.cap.release()

      self.appState.videoFile = filename
      self.appState.cap = cv2.VideoCapture( filename )
      if self.appState.cap.isOpened():
        vidData = VideoData( self.appState.cap )
        self.sldVideoFrame.setMax( vidData.frames - 1 )
        self.minFrame.setMax( vidData.frames - 1 )
        self.maxFrame.setMax( vidData.frames - 1 )
        self.mainImageController.load( self.appState.cap, vidData )
        self.mainImageController.setFrame( 0 )
        self.checkButtonState()

  def hasHomography( self ) -> bool:
    return len( self.appState.data.world_pts ) >= 4

  def cmdUpdateVideoFrame( self, value ):
    self.mainImageController.setFrame( int( value ) )
    self.livePreviewController.updateMappings( self.appState.tracks, int( value ) )

  def checkButtonState( self ):
    cappable = self.appState.cap is not None and self.appState.cap.isOpened()
    homoable = self.hasHomography() and len( self.appState.tracks.items() ) > 0
    self.btnYoloOneFrame.config( state=tk.NORMAL if cappable else tk.DISABLED )
    self.btnYoloRange.config( state=tk.NORMAL if cappable else tk.DISABLED )
    self.uiHomography.setEnableStatus( self.hasHomography(), homoable )
    self.btnDataSave.config( state=tk.NORMAL if homoable else tk.DISABLED )
    self.sldVideoFrame.setEnabled( cappable )

  def setProgRange( self, prog: ttk.Progressbar, val: int, max: int ):
    prog[ 'value' ] = val
    prog[ 'maximum' ] = max

  def runYolo( self, minFrame, maxFrame ):
    # Step 1 - load the model
    self.allocateModelTracking()
    if self.tracking is not None:
      # Step 2 - do it
      # But clear out existing homography data...
      for value in self.appState.tracks.values():
        value.clearHomography()
      logger.info( f"Tracking frames {minFrame} to {maxFrame}" )
      self.prgDetection.setRange( 0, ( maxFrame-minFrame ) + 1 )
      self.prgHomography.setRange( 0, 0 )
      self.tracking.in_queue.put( Command( CommandType.PAUSE ) )
      self.tracking.in_queue.put( Command( CommandType.RUN_FRAMES, minFrame, maxFrame ) )
      self.tracking.in_queue.put( Command( CommandType.RESUME ) )
      self.prgDetection.start()
      self.root.after( 100, self.pollForUI )

  def cmdYoloOneFrame( self ):
    self.runYolo( self.mainImageController.frame_num, self.mainImageController.frame_num )

  def pollForUI( self ):
    data: Output | None = None
    try:
      if self.tracking is not None:
        data = self.tracking.out_queue.get_nowait()
    except queue.Empty:
      data = None

    if data is not None:
      if data.type == OutputType.BBOX and data.data is not None and isinstance( data.data, RawTrackData ):
        if data.data.tid not in self.appState.tracks:
          self.appState.tracks[ data.data.tid ] = Track( data.data.tid )
        self.appState.tracks[ data.data.tid ].boxes.append( data.data.data )
      elif data.type == OutputType.NEW_FRAME:
        self.prgDetection.tick()
        self.appState.framesProcessed += 1
        if self.appState.framesProcessed % CHUNK_SIZE == 0:
          # Write out the saved data
          self.chunkIt()
          self.appState.chunk += 1
      elif data.type == OutputType.COMPLETED:
        self.prgDetection.stop()
        self.refreshHomographyData( self.mainImageController.frame_num )
        self.mainImageController.updateBoundingBoxes( self.appState.tracks, self.mainImageController.frame_num )
        self.livePreviewController.updateMappings( self.appState.tracks, self.mainImageController.frame_num )
        self.checkButtonState()
        self.tabData.tabTracks.refresh()
        return

    self.root.after( 50, self.pollForUI )

  def cmdYoloRange( self ):
    self.runYolo( self.minFrame.get(), self.maxFrame.get() )

  def chunkIt( self ):
    loTrack = self.appState.chunk * CHUNK_SIZE
    hiTrack = ( self.appState.chunk + 1 ) * CHUNK_SIZE
    export: list[ Track ] = []
    for value in self.appState.tracks.values():
      toAdd = value.forExport( loTrack, hiTrack )
      if len( toAdd.boxes ) > 0:
        export.append( toAdd )

    writeBatch( 1, self.appState.chunk, export )

  def allocateModelTracking( self ):
    if self.tracking is not None and self.tracking.thread is not None and self.tracking.thread.is_alive():
      return
    print( "Creating SportsTracker" )
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

  def on_main_view_change( self ):
    xf = self.mainImageController.transform

    self.tabData.tabImagePreview.refresh( xf )

    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )

  def on_radar_click( self, x: int, y: int, point: SelectionPoint ):
    self.appState.sel_world_point = point
    self.mapping_check()

  def on_radar_hover( self, x: int, y: int ):
    pass

  def on_options_change( self ):
    self.mainImageController.refreshHough()

  def redisplayHomographyData( self ):
    self.tabData.tabHomographyData.refresh()
    logger.info( "Clearing existing homography calculations" )
    for value in self.appState.tracks.values():
      value.clearHomography()
    self.refreshHomographyData()

  def bumpIt( self ):
    self.prgHomography.tick()

  def refreshHomographyData( self, index: int = 0 ):
    # Now recalculate
    logger.info( f"Refreshing homography calculations for {len(self.appState.tracks.items())} tracks" )
    self.prgHomography.setRange( 0, len( self.appState.tracks.items() ) )
    for ( i, value ) in self.appState.tracks.items():
      print( f"Refreshing homography for track {i}" )
      value.refreshHomography( self.appState.data )
      self.root.after( 0, self.bumpIt )
    self.livePreviewController.updateMappings( self.appState.tracks, index )


if __name__ == "__main__":
  root = tk.Tk()
  appState = AppState()
  appState.pitch.padding = 10
  appState.pitch.makeEmptyPitch()
  logger.setLevel( logging.INFO )
  app = App( root, appState )
  root.mainloop()
