import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

from LivePreview import LivePreview
from MainCanvas import MainCanvasController
from RadarCanvas import RadarCanvas
from appState import AppState
from dataTypes import Homography, SelectionPoint, VideoData


class App:

  def __init__( self, root, appState: AppState ):
    self.root = root
    self.root.title( "Homography Mapper" )
    self.root.geometry( "1920x1080" )
    self.root.resizable( True, True )
    self.appState: AppState = appState

    # Initialize variables

    # Create widgets
    self._create_widgets()

  def _create_widgets( self ):
    """Create and place all widgets"""

    # imagePreview
    self.imagePreview = tk.Canvas( self.root, bg="#ffffff", highlightthickness=1, highlightbackground="#d1d5db" )
    self.imagePreview.place( x=50, y=50, width=1280, height=720 )

    # lblImagePreview
    self.lblImagePreview = tk.Label( self.root, text="Image Preview", fg="#000000", font=( "Arial", 12 ) )
    self.lblImagePreview.place( x=50, y=720 + 56, width=600, height=24 )

    self.tblPreview = ttk.Treeview( root, columns=( "Property", "Value" ), show="headings" )
    self.tblPreview.place( x=50, y=720 + 80, width=600, height=160 )

    self.tblPreview.heading( "Property", text="Property" )
    self.tblPreview.heading( "Value", text="Value" )

    initialData = [ ( "Zoom", "" ), ( "Offset X", "" ), ( "Offset Y", "" ) ]
    self.tblPreview.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblPreview.tag_configure( "evenrow", background="#993333", foreground="white" )

    self.constantRowIDs = []
    for i, row in enumerate( initialData ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      iid = self.tblPreview.insert( "", tk.END, values=row, tags=( tag,) )
      self.constantRowIDs.append( iid )

    # lblHomographyData
    self.lblHmographyData = tk.Label( self.root, text="Homography Data", fg="#000000", font=( "Arial", 12 ) )
    self.lblHmographyData.place( x=730, y=720 + 56, width=600, height=24 )

    self.tblHomographyData = ttk.Treeview(
        root, columns=( "Field Point", "World Position", "Image Position" ), show="headings"
    )
    self.tblHomographyData.place( x=730, y=720 + 80, width=600, height=160 )

    self.tblHomographyData.heading( "Field Point", text="Field Point" )
    self.tblHomographyData.heading( "World Position", text="World Position" )
    self.tblHomographyData.heading( "Image Position", text="Image Position" )

    self.tblHomographyData.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblHomographyData.tag_configure( "evenrow", background="#993333", foreground="white" )

    # radarMap
    self.radarMap = tk.Canvas( self.root, bg="#dfdfdf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.radarMap.place( x=1460, y=50, width=420 + 20, height=272 + 20 )

    # livePreview
    self.livePreview = tk.Canvas( self.root, bg="#bfbfbf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.livePreview.place( x=1460, y=400, width=420 + 20, height=272 + 20 )

    # lblRadar
    self.lblRadar = tk.Label(
        self.root, text="Bird's eye view (point matcher)", fg="#000000", font=( "Arial", 12 ), anchor="center"
    )
    self.lblRadar.place( x=1460, y=20, width=250, height=24 )

    # lblLivePreview
    self.lblLivePreview = tk.Label( self.root, text="Live Preview", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblLivePreview.place( x=1460, y=350, width=100, height=24 )

    # lblHomography
    self.lblHomography = tk.Label(
        self.root, text="Homography Point Matcher", fg="#000000", font=( "Arial", 12 ), anchor="center"
    )
    self.lblHomography.place( x=50, y=20, width=200, height=24 )

    # btnLoadHomography
    self.btnLoadHomography = tk.Button(
        self.root, text="Load Homography", font=( "Arial", 12 ), command=self.cmdLoadHomography
    )
    self.btnLoadHomography.place( x=1460, y=700, width=180, height=36 )

    # btnSaveHomography
    self.btnSaveHomography = tk.Button(
        self.root, text="Save Homography", font=( "Arial", 12 ), command=self.cmdSaveHomography
    )
    self.btnSaveHomography.place( x=1650, y=700, width=180, height=36 )

    # btnLoadBoundingBoxes
    self.btnLoadBoundingBoxes = tk.Button( self.root, text="Load Bounding Boxes", font=( "Arial", 12 ), command=self.cmdLoadBB )
    self.btnLoadBoundingBoxes.place( x=1460, y=750, width=180, height=36 )

    # btnRunYoloDetection
    self.btnRunYoloDetection = tk.Button(
        self.root, text="Run Detection", font=( "Arial", 12 ), command=self.cmdRunYoloDetection
    )
    self.btnRunYoloDetection.place( x=1650, y=750, width=180, height=36 )

    # btnLoadVideo
    self.btnLoadVideo = tk.Button( self.root, text="Load Video", font=( "Arial", 12 ), command=self.cmdLoadVideo )
    self.btnLoadVideo.place( x=1460, y=800, width=180, height=36 )

    # sliderVideoFrame
    self.sldVideoFrame = tk.Scale( self.root, from_=0, to=100, orient='horizontal' )
    self.sldVideoFrame.place( x=1650, y=860, width=200, height=240 )

    # lblVideoFrameSlider
    self.lblVideoFrameSlider = tk.Label( self.root, text="Video Frame", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblVideoFrameSlider.place( x=1460, y=860, width=100, height=24 )

    self.radarMapController = RadarCanvas(
        self.radarMap, ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ), self.appState.cfg,
        self.appState.pitch, self.on_radar_click, self.on_radar_hover
    )

    self.mainImageController = MainCanvasController(
        self.imagePreview, self.appState.cfg, self.appState.pitch, self.on_main_click, self.on_main_hover,
        self.on_main_view_change
    )

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
        self.sldVideoFrame.config( to=vidData.frames )
        self.mainImageController.load( self.appState.cap, vidData )
        self.mainImageController.setFrame( 0 )

  def cmdRunYoloDetection( self ):
    """
    Handle cmdRunYoloDetection event
    TODO: Implement your logic here
    """
    # Step 1 - load the model
    self.appState.model = YOLO( r"runs\detect\train16\weights\best.pt", verbose=False )
    pass

  def mapping_check( self ):
    if self.appState.last_image_click is None or self.appState.sel_world_point is None:
      return

    # We have a map!  Construct a mapping pair
    print( f"Adding a world point {self.appState.sel_world_point} at {self.appState.last_image_click}" )
    self.appState.data.world_pts.append( self.appState.sel_world_point )
    self.appState.data.img_pts_4k.append( self.appState.last_image_click )
    self.appState.last_image_click = None
    self.appState.sel_world_point = None
    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )
    self.refreshHomographyData()

  def on_main_click( self, x: int, y: int, point: SelectionPoint ):
    print( "Main  Click ", str( x ), "x", str( y ) )
    self.appState.last_image_click = point
    self.mapping_check()

  def on_main_hover( self, x: int, y: int ):
    pass

  def on_main_view_change( self ):
    print( "View Change " )
    xf = self.mainImageController.transform
    updates = [ ( "Zoom", f"{xf.scale:.3f}" ), ( "Offset X", f"{xf.offset.x:.3f}" ), ( "Offset Y", f"{xf.offset.y:.3f}" ) ]
    for iid, new_values in zip( self.constantRowIDs, updates ):
      self.tblPreview.item( iid, values=new_values )

    self.radarMapController.updateSelectionMarkers( self.appState.data.world_pts )
    self.mainImageController.updateSelectionMarkers( self.appState.data.img_pts_4k )

  def on_radar_click( self, x: int, y: int, point: SelectionPoint ):
    print( "Radar Click ", str( x ), "x", str( y ) )
    self.appState.sel_world_point = point
    self.mapping_check()

  def on_radar_hover( self, x: int, y: int ):
    pass

  def refreshHomographyData( self ):
    for iid in self.tblHomographyData.get_children():
      self.tblHomographyData.delete( iid )

    for i, ( sel, world ) in enumerate( zip( self.appState.data.img_pts_4k, self.appState.data.world_pts ) ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      row = (
          str( self.appState.cfg.labels[ world.index ] ) if ( self.appState.cfg and world.index is not None ) else "",
          f"{world.coords.x:.3f},{world.coords.y:.3f}", f"{sel.coords.x},{sel.coords.y}"
      )
      self.tblHomographyData.insert( "", tk.END, values=row, tags=( tag,) )


if __name__ == "__main__":
  root = tk.Tk()
  appState = AppState()
  appState.pitch.padding = 10
  appState.pitch.makeEmptyPitch()
  app = App( root, appState )
  root.mainloop()
