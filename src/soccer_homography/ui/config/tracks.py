import tkinter as tk
from tkinter import ttk

from soccer_homography.appState import AppState


class Tracks:

  def __init__( self, appState: AppState, tab: ttk.Frame ) -> None:
    self.appState = appState
    self.tab = tab

  def setup( self ):
    self.tblTrackData = ttk.Treeview( self.tab, columns=( "Track ID", "Num Frames", "Person" ), show="headings" )
    self.tblTrackData.place( x=0, y=24, width=600, height=160 )

    self.tblTrackData.heading( "Track ID", text="Track ID" )
    self.tblTrackData.heading( "Num Frames", text="Num Frames" )
    self.tblTrackData.heading( "Person", text="Person" )

    self.tblTrackData.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblTrackData.tag_configure( "evenrow", background="#993333", foreground="white" )

  def refresh( self ):
    for iid in self.tblTrackData.get_children():
      self.tblTrackData.delete( iid )

    for i, ( key, track ) in enumerate( self.appState.tracks.items() ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      row = ( str( key ), str( len( track.boxes ) ), track.person.name if track.person is not None else "<Unknown>" )
      self.tblTrackData.insert( "", tk.END, values=row, tags=( tag,) )
