import tkinter as tk
from tkinter import ttk

from soccer_homography.appState import AppState


class homographyData:

  def __init__( self, appState: AppState, tab: ttk.Frame ) -> None:
    self.appState = appState
    self.tab = tab

  def setup( self ):
    self.tblHomographyData = ttk.Treeview( self.tab, columns=( "Field Point", "World Position", "Image Position" ), show="headings" )
    self.tblHomographyData.place( x=0, y=24, width=600, height=160 )

    self.tblHomographyData.heading( "Field Point", text="Field Point" )
    self.tblHomographyData.heading( "World Position", text="World Position" )
    self.tblHomographyData.heading( "Image Position", text="Image Position" )

    self.tblHomographyData.tag_configure( "oddrow", background="#661111", foreground="white" )
    self.tblHomographyData.tag_configure( "evenrow", background="#993333", foreground="white" )

  def refresh( self ):
    for iid in self.tblHomographyData.get_children():
      self.tblHomographyData.delete( iid )

    for i, ( sel, world ) in enumerate( zip( self.appState.data.img_pts_4k, self.appState.data.world_pts ) ):
      tag = "evenrow" if i % 2 == 0 else "oddrow"
      row = (
          str( self.appState.cfg.labels[ world.index ] ) if
          ( self.appState.cfg and world.index is not None ) else "", f"{world.coords.x:.3f},{world.coords.y:.3f}", f"{sel.coords.x},{sel.coords.y}"
      )
      self.tblHomographyData.insert( "", tk.END, values=row, tags=( tag,) )
