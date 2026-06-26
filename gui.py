import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

from Configuration import Configuration
from LivePreview import LivePreview
from MainCanvas import MainCanvasController
from RadarCanvas import RadarCanvas
from SportsTracker import SportsTracker
from appState import AppState
from components import Slider
from dataTypes import Homography, SelectionPoint, VideoData


class App:

  def __init__( self, root, appState: AppState ):
    self.root = root
    self.root.title( "Homography Mapper" )
    self.root.geometry( "1920x1080" )
    self.root.resizable( True, True )
    self.appState: AppState = appState
    self.tracking: SportsTracker | None = None

    # Initialize variables

    # Create widgets
    self._create_widgets()

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

    # btnLoadHomography
    self.btnLoadHomography = tk.Button( self.root, text="Load Homography", font=( "Arial", 12 ), command=self.cmdLoadHomography )
    self.btnLoadHomography.place( x=1460, y=700, width=180, height=36 )

    # btnSaveHomography
    self.btnSaveHomography = tk.Button( self.root, text="Save Homography", font=( "Arial", 12 ), command=self.cmdSaveHomography, state=tk.DISABLED )
    self.btnSaveHomography.place( x=1650, y=700, width=180, height=36 )

    # btnLoadBoundingBoxes
    self.btnLoadBoundingBoxes = tk.Button( self.root, text="Load Bounding Boxes", font=( "Arial", 12 ), command=self.cmdLoadBB )
    self.btnLoadBoundingBoxes.place( x=1460, y=750, width=180, height=36 )

    # btnRunYoloDetection
    self.btnRunYoloDetection = tk.Button( self.root, text="Run Detection", font=( "Arial", 12 ), command=self.cmdRunYoloDetection, state=tk.DISABLED )
    self.btnRunYoloDetection.place( x=1650, y=750, width=180, height=36 )

    # btnRunYoloVidDetection
    self.btnRunYoloVidDetection = tk.Button( self.root, text="Run Detection (video)", font=( "Arial", 12 ), command=self.cmdRunYoloVidDetection, state=tk.DISABLED )
    self.btnRunYoloVidDetection.place( x=1650, y=800, width=180, height=36 )

    # btnLoadVideo
    self.btnLoadVideo = tk.Button( self.root, text="Load Video", font=( "Arial", 12 ), command=self.cmdLoadVideo )
    self.btnLoadVideo.place( x=1460, y=800, width=180, height=36 )

    # sliderVideoFrame
    #self.sldVideoFrame = tk.Scale( self.root, from_=0, to=100, orient='horizontal', command=self.cmdUpdateVideoFrame )
    #self.sldVideoFrame.place( x=1650, y=860, width=200, height=240 )
    self.sldVideoFrame = Slider( from_=0, to=100, command=self.cmdUpdateVideoFrame, root=self.root, x=1650, y=860, width=200, height=240 )

    # lblVideoFrameSlider
    self.lblVideoFrameSlider = tk.Label( self.root, text="Video Frame", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblVideoFrameSlider.place( x=1460, y=860, width=100, height=24 )

    self.radarMapController = RadarCanvas(
        self.radarMap, ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ), self.appState.cfg, self.appState.pitch, self.on_radar_click, self.on_radar_hover
    )

    self.mainImageController = MainCanvasController( self.imagePreview, self.appState, self.on_main_click, self.on_main_hover, self.on_main_view_change )
    self.livePreviewController = LivePreview( self.livePreview, ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ), self.appState )

  # ==========================================
  # Event Handlers - Implement your logic here
  # ==========================================

  def cmdLoadHomography( self ):
    filetypes = ( ( 'Homography files', '*.json' ),)

    filename = filedialog.askopenfilename( title='Open homography', initialdir='.', filetypes=filetypes )
    if filename is not None:
      # Reset the homography
      self.appState.data = Homography()
      # And then load it
      self.appState.data.load( filename )
      self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
      self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )
      self.refreshHomographyData()

  def cmdSaveHomography( self ):
    filetypes = ( ( 'Homography files', '*.json' ),)
    filename = filedialog.asksaveasfilename( title='Save homography', initialdir='.', filetypes=filetypes )
    if filename is not None:
      # And then load it
      self.appState.data.save( filename )

  def cmdLoadBB( self ):
    """
    Handle cmdLoadBB event
    TODO: Implement your logic here
    """
    pass

  def cmdLoadVideo( self ):
    filetypes = ( ( 'Video files', [ '*.mp4', '*.mkv' ] ),)

    filename = filedialog.askopenfilename( title='Open video', initialdir='.', filetypes=filetypes )
    if filename is not None:
      if self.appState.cap is not None:
        self.appState.cap.release()

      self.appState.cap = cv2.VideoCapture( filename )
      if self.appState.cap.isOpened():
        vidData = VideoData( self.appState.cap )
        self.sldVideoFrame.setMax( vidData.frames )
        self.mainImageController.load( self.appState.cap, vidData )
        self.mainImageController.setFrame( 0 )
        self.checkButtonState()

  def cmdUpdateVideoFrame( self, value ):

    self.mainImageController.setFrame( int( value ) )

  def checkButtonState( self ):
    cappable = self.appState.cap is not None and self.appState.cap.isOpened()
    homoable = len( self.appState.data.world_pts ) >= 4
    self.btnRunYoloDetection.config( state=tk.NORMAL if cappable else tk.DISABLED )
    self.btnRunYoloVidDetection.config( state=tk.NORMAL if cappable and homoable else tk.DISABLED )
    self.btnSaveHomography.config( state=tk.NORMAL if homoable else tk.DISABLED )
    self.sldVideoFrame.setEnabled( cappable )

  def cmdRunYoloDetection( self ):
    """
    Handle cmdRunYoloDetection event
    TODO: Implement your logic here
    """
    # Step 1 - load the model
    self.allocateModelTracking()
    if self.tracking is not None:
      image = np.array( self.mainImageController.pil_image.convert( "RGB" ) )
      result_rgb = cv2.cvtColor( image, cv2.COLOR_RGB2BGR )

      self.tracking.track( result_rgb, self.mainImageController.frame_num )

  def cmdRunYoloVidDetection( self ):
    """
    Handle cmdRunYoloDetection event
    TODO: Implement your logic here
    """
    self.allocateModelTracking()
    pass

  def allocateModelTracking( self ):
    if self.tracking is not None:
      # Already allocated
      return
    self.tracking = SportsTracker( self.appState.mdlOpts )

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
    self.refreshHomographyData()

  def on_main_click( self, x: int, y: int, point: SelectionPoint ):
    self.appState.last_image_click = point
    self.mapping_check()

  def on_main_hover( self, x: int, y: int ):
    pass

  def on_main_view_change( self ):
    xf = self.mainImageController.transform

    self.tabData.refreshImagePreview( xf )

    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )

  def on_radar_click( self, x: int, y: int, point: SelectionPoint ):
    self.appState.sel_world_point = point
    self.mapping_check()

  def on_radar_hover( self, x: int, y: int ):
    pass

  def on_options_change( self ):
    self.mainImageController.refreshHough()

  def refreshHomographyData( self ):
    self.tabData.refreshHomographyData()


if __name__ == "__main__":
  root = tk.Tk()
  appState = AppState()
  appState.pitch.padding = 10
  appState.pitch.makeEmptyPitch()
  app = App( root, appState )
  root.mainloop()
