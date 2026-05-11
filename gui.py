import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

from RadarCanvas import RadarCanvas
from appState import AppState


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

    # selector
    self.selector = tk.Canvas( self.root, bg="#ffffff", highlightthickness=1, highlightbackground="#d1d5db" )
    self.selector.place( x=0, y=50, width=1440, height=810 )

    # radarMap
    self.radarMap = tk.Canvas( self.root, bg="#dfdfdf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.radarMap.place( x=1460, y=50, width=420 + 20, height=272 + 20 )
    self.radarMap.bind( "<Motion>", self.on_radar_hover )

    # livePreview
    self.livePreview = tk.Canvas( self.root, bg="#bfbfbf", highlightthickness=1, highlightbackground="#d1d5db" )
    self.livePreview.place( x=1460, y=400, width=420, height=272 )

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
        self.root, text="Save Homography", font=( "Arial", 12 ), command=self.btnSaveHomography
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
    # TODO: Implement slider widget
    self.sldVideoFrame = tk.Scale( self.root, from_=0, to=100, orient='horizontal' )
    self.sldVideoFrame.place( x=1650, y=860, width=200, height=240 )

    # lblVideoFrameSlider
    self.lblVideoFrameSlider = tk.Label( self.root, text="Video Frame", fg="#000000", font=( "Arial", 12 ), anchor="center" )
    self.lblVideoFrameSlider.place( x=1460, y=860, width=100, height=24 )

    self.radarMapController = RadarCanvas(
        self.radarMap, ImageTk.PhotoImage( Image.fromarray( self.appState.pitch.empty ) ), self.appState.cfg,
        self.appState.pitch
    )

  # ==========================================
  # Event Handlers - Implement your logic here
  # ==========================================

  def cmdLoadHomography( self ):
    """
    Handle cmdLoadHomography event
    TODO: Implement your logic here
    """
    pass

  def btnSaveHomography( self ):
    """
    Handle btnSaveHomography event
    TODO: Implement your logic here
    """
    pass

  def cmdLoadBB( self ):
    """
    Handle cmdLoadBB event
    TODO: Implement your logic here
    """
    pass

  def cmdLoadVideo( self ):
    """
    Handle cmdLoadVideo event
    TODO: Implement your logic here
    """
    pass

  def cmdRunYoloDetection( self ):
    """
    Handle cmdRunYoloDetection event
    TODO: Implement your logic here
    """
    pass

  #def renderFieldPoints( self ):
  #  copy = self.appState.pitch.empty.copy()
  #  pitch.draw_key_points( active, x0, y0, data.img_pts_4k )
  #  pitch.draw_sel_points( active, data.img_pts_4k, scale=0.5 )
  #  pass

  def on_radar_hover( self, event ):
    rx, ry = event.x, event.y
    nearest = self.appState.pitch.nearest_field_point( rx, ry )
    x, y = self.appState.pitch.calc_point_offset( nearest )
    self.radarMap.coords( self.hover_marker, x - 6, y - 6, x + 6, y + 6 )
    self.radarMap.itemconfig( self.hover_marker, state="normal" )
    #self.update_radar_hover(rx, ry)


if __name__ == "__main__":
  root = tk.Tk()
  appState = AppState()
  appState.pitch.padding = 10
  appState.pitch.draw_empty_pitch()
  app = App( root, appState )
  root.mainloop()
