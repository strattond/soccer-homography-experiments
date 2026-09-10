import tkinter as tk
from tkinter import ttk

from soccer_homography.dataTypes import ViewTransform


class ImagePreview:

  def __init__( self, tab: ttk.Frame ) -> None:
    self.tab = tab

  def setup( self ):
    self.tblPreview = ttk.Treeview( self.tab, columns=( "Property", "Value" ), show="headings" )
    self.tblPreview.place( x=0, y=24, width=600, height=160 )

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

  def refresh( self, xf: ViewTransform ):
    updates = [ ( "Zoom", f"{xf.scale:.3f}" ), ( "Offset X", f"{xf.offset.x:.3f}" ), ( "Offset Y", f"{xf.offset.y:.3f}" ) ]
    for iid, new_values in zip( self.constantRowIDs, updates ):
      self.tblPreview.item( iid, values=new_values )
