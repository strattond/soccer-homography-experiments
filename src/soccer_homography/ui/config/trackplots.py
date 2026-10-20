import tkinter as tk
from tkinter import ttk

from soccer_homography.data import Track


class TrackPlots( ttk.Frame ):

  def __init__( self, master ):
    super().__init__( master )
    self.track: Track | None = None
    self.frame_rate = 50.0
    self.canvas = tk.Canvas( self, background="white", highlightthickness=0 )
    self.canvas.pack( fill="both", expand=True )
    self.canvas.bind( "<Configure>", lambda _event: self.draw() )
    self.min_frame = 0
    self.max_frame = 3000

  def setTrack( self, track: Track | None, frame_rate: float = 50.0, min_frame: int = 0, max_frame: int = 3000 ) -> None:
    if frame_rate <= 0:
      raise ValueError( "Frame rate must be positive to plot track velocity" )
    self.track = track
    self.frame_rate = frame_rate
    self.min_frame = min_frame
    self.max_frame = max_frame
    self.draw()

  def draw( self ) -> None:
    canvas = self.canvas
    canvas.delete( "all" )
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    if width < 120 or height < 100:
      return

    track = self.track
    if track is None:
      canvas.create_text( width / 2, height / 2, text="Select a track with homography data to view plots", fill="#555555" )
      return
    if self.min_frame > self.max_frame:
      canvas.create_text( width / 2, height / 2, text="Start frame must not exceed finish frame", fill="#555555" )
      return
    point_count = min( len( track.boxes ), len( track.homog ), len( track.homog_smooth ), len( track.homog_velocity ) )
    if point_count == 0:
      canvas.create_text( width / 2, height / 2, text="Select a track with homography data to view plots", fill="#555555" )
      return

    all_frames = [ box.frame for box in track.boxes[ :point_count ] ]
    all_raw = track.homog[ :point_count ]
    all_smoothed = track.homog_smooth[ :point_count ]
    all_velocity = track.homog_velocity[ :point_count ]
    visible_indices = [
        index for index, frame in enumerate( all_frames )
        if self.min_frame <= frame <= self.max_frame
    ]
    if not visible_indices:
      canvas.create_text(
          width / 2,
          height / 2,
          text=f"No track data between frames {self.min_frame} and {self.max_frame}",
          fill="#555555",
      )
      return

    frames = [ all_frames[ index ] for index in visible_indices ]
    raw = [ all_raw[ index ] for index in visible_indices ]
    smoothed = [ all_smoothed[ index ] for index in visible_indices ]
    velocity = [ all_velocity[ index ] for index in visible_indices ]
    measured_speeds = []
    for index in visible_indices:
      if index == 0:
        measured_speeds.append( 0.0 )
        continue
      dt = ( all_frames[ index ] - all_frames[ index - 1 ] ) / self.frame_rate
      if dt <= 0:
        raise ValueError( "Track frames must be strictly increasing to plot measured velocity" )
      measured_speeds.append(
          ( ( all_raw[ index ].x - all_raw[ index - 1 ].x )**2 + ( all_raw[ index ].y - all_raw[ index - 1 ].y )**2 )**0.5 / dt
      )
    position_series = (
        ( "Raw X", [ point.x for point in raw ], "#d38b00", True ),
        ( "Smoothed X", [ point.x for point in smoothed ], "#d38b00", False ),
        ( "Raw Y", [ point.y for point in raw ], "#2864a5", True ),
        ( "Smoothed Y", [ point.y for point in smoothed ], "#2864a5", False ),
    )
    speeds = [ ( point.x**2 + point.y**2 )**0.5 for point in velocity ]
    velocity_series = (
        ( "Vx", [ point.x for point in velocity ], "#d38b00", False ),
        ( "Vy", [ point.y for point in velocity ], "#2864a5", False ),
        ( "Speed", speeds, "#333333", False ),
        ( "Measured", measured_speeds, "#c62828", True ),
    )
    gap = 12
    chart_height = ( height-gap ) / 2
    frame_bounds = ( self.min_frame, self.max_frame )
    self.drawChart( 8, 4, width - 16, chart_height, "Position (m)", frames, position_series, frame_bounds=frame_bounds )
    self.drawChart(
        8,
        chart_height + gap,
        width - 16,
        chart_height,
        "Velocity (m/s)",
        frames,
        velocity_series,
        speed_outliers=measured_speeds,
        frame_bounds=frame_bounds,
    )

  def drawChart( self, left, top, width, height, title, frames, series, speed_outliers=None, frame_bounds=None ) -> None:
    canvas = self.canvas
    plot_left = left + 48
    plot_right = left + width - 8
    plot_top = top + 20
    plot_bottom = top + height - 20
    canvas.create_text( left + 4, top + 2, text=title, anchor="nw", font=( "Arial", 9, "bold" ) )
    if plot_right <= plot_left or plot_bottom <= plot_top:
      return

    values = [ value for _, line, _, _ in series for value in line ]
    if speed_outliers is not None:
      values.extend( ( 10.0, *speed_outliers ) )
    value_min = min( values, default=0.0 )
    value_max = max( values, default=1.0 )
    if speed_outliers is not None:
      value_min = min( value_min, -10.0 )
      value_max = max( value_max, 10.0 )
    if value_min == value_max:
      value_min -= 0.5
      value_max += 0.5
    padding = ( value_max-value_min ) * 0.08
    value_min -= padding
    value_max += padding
    frame_min, frame_max = frame_bounds if frame_bounds is not None else ( min( frames ), max( frames ) )
    if frame_min == frame_max:
      frame_min -= 0.5
      frame_max += 0.5

    def x_coord( frame ):
      return plot_left + ( frame-frame_min ) / ( frame_max-frame_min ) * ( plot_right-plot_left )

    def y_coord( value ):
      return plot_bottom - ( value-value_min ) / ( value_max-value_min ) * ( plot_bottom-plot_top )

    canvas.create_line( plot_left, plot_top, plot_left, plot_bottom, fill="#777777" )
    canvas.create_line( plot_left, plot_bottom, plot_right, plot_bottom, fill="#777777" )
    for fraction in ( 0.0, 0.25, 0.5, 0.75, 1.0 ):
      y = plot_bottom - fraction * ( plot_bottom-plot_top )
      value = value_min + fraction * ( value_max-value_min )
      canvas.create_line( plot_left, y, plot_right, y, fill="#e5e5e5" )
      canvas.create_text( plot_left - 4, y, text=f"{value:.1f}", anchor="e", fill="#555555", font=( "Arial", 7 ) )
    canvas.create_text( plot_left, plot_bottom + 3, text=str( self.min_frame if frame_bounds is not None else frames[ 0 ] ), anchor="nw", fill="#555555", font=( "Arial", 7 ) )
    canvas.create_text( plot_right, plot_bottom + 3, text=str( self.max_frame if frame_bounds is not None else frames[ -1 ] ), anchor="ne", fill="#555555", font=( "Arial", 7 ) )
    canvas.create_text( ( plot_left+plot_right ) / 2, plot_bottom + 3, text="Frame", anchor="n", fill="#555555", font=( "Arial", 7 ) )

    if speed_outliers is not None:
      threshold_y = y_coord( 10.0 )
      canvas.create_line( plot_left, threshold_y, plot_right, threshold_y, fill="#c62828", dash=( 4, 3 ) )
      canvas.create_text( plot_right - 2, threshold_y - 2, text="10 m/s", anchor="se", fill="#c62828", font=( "Arial", 7 ) )

    legend_x = plot_left + 4
    legend_y = plot_top + 2
    for label, line, color, dashed in series:
      points = [ coordinate for frame, value in zip( frames, line ) for coordinate in ( x_coord( frame ), y_coord( value ) ) ]
      if len( points ) >= 4:
        options = { "fill": color, "width": 1 if dashed else 2}
        if dashed:
          options[ "dash" ] = ( 3, 2 )
        canvas.create_line( *points, **options )
      canvas.create_line( legend_x, legend_y, legend_x + 12, legend_y, fill=color, dash=( 3, 2 ) if dashed else () )
      canvas.create_text( legend_x + 15, legend_y, text=label, anchor="w", fill="#333333", font=( "Arial", 7 ) )
      legend_x += 58 if len( label ) < 8 else 80

    if speed_outliers is not None:
      for frame, speed in zip( frames, speed_outliers ):
        if speed > 10:
          canvas.create_oval(
              x_coord( frame ) - 3,
              y_coord( speed ) - 3,
              x_coord( frame ) + 3,
              y_coord( speed ) + 3,
              fill="#c62828",
              outline="white",
          )
