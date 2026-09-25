from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import duckdb

from soccer_homography.db import (
    Camera,
    ClipDB,
    Match,
    Video,
    deleteVideo,
    listCameras,
    listClips,
    listMatches,
    listPersons,
    listVideos,
    reorderClips,
    upsertCamera,
    upsertClip,
    upsertMatch,
    upsertVideo,
)


class DataMaintenance:

  def __init__( self, parent: tk.Misc, conn: duckdb.DuckDBPyConnection ):
    self.parent = parent
    self.conn = conn
    self.window = tk.Toplevel( parent )
    self.window.title( "Data Maintenance" )
    self.window.geometry( "1100x700" )
    self.window.transient( parent )
    self.window.protocol( "WM_DELETE_WINDOW", self.close )
    self.build()

  def close( self ):
    self.window.destroy()

  def build( self ):
    self.notebook = ttk.Notebook( self.window )
    self.notebook.pack( fill="both", expand=True, padx=8, pady=8 )
    self.buildVideos()
    self.buildMatches()
    self.buildCameras()
    self.buildClips()
    self.buildPersons()

  def makeTree( self, tab: ttk.Frame, columns: tuple[ str, ...], headings: tuple[ str, ...] ) -> ttk.Treeview:
    tree = ttk.Treeview( tab, columns=columns, show="headings", selectmode="browse" )
    for column, heading in zip( columns, headings ):
      tree.heading( column, text=heading )
      tree.column( column, width=140, anchor="w" )
    tree.pack( side="left", fill="both", expand=True )
    scrollbar = ttk.Scrollbar( tab, orient="vertical", command=tree.yview )
    scrollbar.pack( side="right", fill="y" )
    tree.configure( yscrollcommand=scrollbar.set )
    return tree

  def buildVideos( self ):
    tab = ttk.Frame( self.notebook )
    self.notebook.add( tab, text="Videos" )
    self.videoTree = self.makeTree( tab, ( "id", "file" ), ( "ID", "File" ) )
    controls = ttk.Frame( tab )
    controls.pack( fill="x", side="bottom", pady=6 )
    ttk.Button( controls, text="Browse...", command=self.addVideo ).pack( side="left" )
    ttk.Button( controls, text="Remove", command=self.removeVideo ).pack( side="left", padx=6 )
    ttk.Button( controls, text="Refresh", command=self.refreshVideos ).pack( side="left" )
    self.refreshVideos()

  def refreshVideos( self ):
    self.videoTree.delete( *self.videoTree.get_children() )
    for video in listVideos( self.conn ):
      self.videoTree.insert( "", "end", iid=str( video.id ), values=( video.id, video.file ) )
    self.refreshClipVideoList()

  def addVideo( self ):
    path = filedialog.askopenfilename( parent=self.window, title="Select video" )
    if not path:
      return
    capture = cv2.VideoCapture( path )
    opened = capture.isOpened()
    capture.release()
    if not opened:
      messagebox.showerror( "Unsupported video", "OpenCV could not open the selected file.", parent=self.window )
      return
    try:
      upsertVideo( self.conn, Video( id=0, file=path ) )
      self.conn.commit()
    except duckdb.ConstraintException:
      messagebox.showinfo( "Already added", "That video already exists in the database.", parent=self.window )
    except duckdb.Error as error:
      messagebox.showerror( "Database error", str( error ), parent=self.window )
    self.refreshVideos()

  def removeVideo( self ):
    selected = self.videoTree.selection()
    if not selected or not messagebox.askyesno( "Remove video", "Remove the selected video?", parent=self.window ):
      return
    try:
      deleteVideo( self.conn, int( selected[ 0 ] ) )
      self.conn.commit()
    except duckdb.ConstraintException:
      messagebox.showerror( "Video is in use", "Remove its Clips before removing this video.", parent=self.window )
    self.refreshVideos()

  def buildMatches( self ):
    tab = ttk.Frame( self.notebook )
    self.notebook.add( tab, text="Matches" )
    self.matchTree = self.makeTree( tab, ( "id", "date", "home", "away", "division" ), ( "ID", "Date", "Home", "Away", "Division" ) )
    form = ttk.Frame( tab )
    form.pack( fill="x", side="bottom", pady=6 )
    self.matchVars = [ tk.StringVar() for _ in range( 4 ) ]
    for index, label in enumerate( ( "Date", "Home", "Away", "Division" ) ):
      ttk.Label( form, text=label ).grid( row=index, column=0, sticky="w", pady=2 )
      ttk.Entry( form, textvariable=self.matchVars[ index ], width=36 ).grid( row=index, column=1, padx=4, sticky="ew" )
    buttons = ttk.Frame( form )
    buttons.grid( row=4, column=0, columnspan=2, sticky="w", pady=( 6, 0 ) )
    ttk.Button( buttons, text="Save", command=self.saveMatch ).pack( side="left" )
    ttk.Button( buttons, text="Reset", command=lambda: self.clearForm( self.matchVars ) ).pack( side="left", padx=4 )
    form.columnconfigure( 1, weight=1 )
    self.matchTree.bind( "<<TreeviewSelect>>", self.selectMatch )
    self.refreshMatches()

  def refreshMatches( self ):
    self.matchTree.delete( *self.matchTree.get_children() )
    for match in listMatches( self.conn ):
      self.matchTree.insert( "", "end", iid=str( match.id ), values=( match.id, match.date, match.home, match.away, match.division ) )
    self.refreshClipFilters()

  def selectMatch( self, _event=None ):
    selected = self.matchTree.selection()
    if selected:
      values = self.matchTree.item( selected[ 0 ], "values" )
      for variable, value in zip( self.matchVars, values[ 1: ] ):
        variable.set( value )

  def saveMatch( self ):
    selected = self.matchTree.selection()
    match = Match( int( selected[ 0 ] ) if selected else 0, *( variable.get().strip() for variable in self.matchVars ) )
    try:
      upsertMatch( self.conn, match )
      self.conn.commit()
      self.refreshMatches()
    except duckdb.Error as error:
      messagebox.showerror( "Database error", str( error ), parent=self.window )

  def buildCameras( self ):
    tab = ttk.Frame( self.notebook )
    self.notebook.add( tab, text="Cameras" )
    self.cameraTree = self.makeTree( tab, ( "id", "name" ), ( "ID", "Name" ) )
    form = ttk.Frame( tab )
    form.pack( fill="x", side="bottom", pady=6 )
    self.cameraName = tk.StringVar()
    ttk.Label( form, text="Name" ).grid( row=0, column=0, sticky="w" )
    ttk.Entry( form, textvariable=self.cameraName, width=36 ).grid( row=0, column=1, padx=4, sticky="ew" )
    buttons = ttk.Frame( form )
    buttons.grid( row=1, column=0, columnspan=2, sticky="w", pady=( 6, 0 ) )
    ttk.Button( buttons, text="Save selected/new", command=self.saveCamera ).pack( side="left" )
    ttk.Button( buttons, text="New", command=lambda: self.cameraName.set( "" ) ).pack( side="left", padx=4 )
    form.columnconfigure( 1, weight=1 )
    self.cameraTree.bind( "<<TreeviewSelect>>", self.selectCamera )
    self.refreshCameras()

  def refreshCameras( self ):
    self.cameraTree.delete( *self.cameraTree.get_children() )
    for camera in listCameras( self.conn ):
      self.cameraTree.insert( "", "end", iid=str( camera.id ), values=( camera.id, camera.name ) )
    self.refreshClipFilters()

  def selectCamera( self, _event=None ):
    selected = self.cameraTree.selection()
    if selected:
      self.cameraName.set( self.cameraTree.item( selected[ 0 ], "values" )[ 1 ] )

  def saveCamera( self ):
    selected = self.cameraTree.selection()
    try:
      upsertCamera( self.conn, Camera( int( selected[ 0 ] ) if selected else 0, self.cameraName.get().strip() ) )
      self.conn.commit()
      self.refreshCameras()
    except duckdb.Error as error:
      messagebox.showerror( "Database error", str( error ), parent=self.window )

  def buildClips( self ):
    tab = ttk.Frame( self.notebook )
    self.notebook.add( tab, text="Clips" )
    filters = ttk.Frame( tab )
    filters.pack( fill="x", padx=8, pady=6 )
    filters.columnconfigure( 1, weight=1 )
    self.clipCamera = ttk.Combobox( filters, state="readonly", width=24 )
    self.clipMatch = ttk.Combobox( filters, state="readonly", width=24 )
    ttk.Label( filters, text="Camera" ).grid( row=0, column=0, sticky="w", padx=( 0, 8 ), pady=2 )
    self.clipCamera.grid( row=0, column=1, sticky="ew", pady=2 )
    ttk.Label( filters, text="Match" ).grid( row=1, column=0, sticky="w", padx=( 0, 8 ), pady=2 )
    self.clipMatch.grid( row=1, column=1, sticky="ew", pady=2 )
    self.clipCamera.bind( "<<ComboboxSelected>>", lambda _event: self.refreshClipOrder() )
    self.clipMatch.bind( "<<ComboboxSelected>>", lambda _event: self.refreshClipOrder() )
    body = ttk.Frame( tab )
    body.pack( fill="both", expand=True )
    ttk.Label( body, text="Videos to add" ).pack( anchor="w" )
    self.clipVideos = tk.Listbox( body, selectmode="extended", exportselection=False, height=8 )
    self.clipVideos.pack( fill="x" )
    self.clipOrderTree = self.makeTree( body, ( "sequence", "video", "id" ), ( "Sequence", "Video", "Clip ID" ) )
    self.clipOrderTree.bind( "<ButtonPress-1>", self.startClipDrag )
    self.clipOrderTree.bind( "<B1-Motion>", self.dragClip )
    self.clipOrderTree.bind( "<ButtonRelease-1>", self.finishClipDrag )
    controls = ttk.Frame( tab )
    controls.pack( fill="x", pady=6 )
    ttk.Button( controls, text="Add selected videos", command=self.addClips ).pack( side="left" )
    ttk.Button( controls, text="Move up", command=lambda: self.moveClip( -1 ) ).pack( side="left", padx=4 )
    ttk.Button( controls, text="Move down", command=lambda: self.moveClip( 1 ) ).pack( side="left" )
    ttk.Button( controls, text="Save order", command=self.saveClipOrder ).pack( side="left", padx=4 )
    self.refreshClipVideoList()
    self.refreshClipFilters()

  def refreshClipVideoList( self ):
    if not hasattr( self, "clipVideos" ):
      return
    self.clipVideos.delete( 0, tk.END )
    for video in listVideos( self.conn ):
      self.clipVideos.insert( tk.END, f"{video.id}: {video.file}" )

  def refreshClipFilters( self ):
    if not hasattr( self, "clipCamera" ):
      return
    self.cameras = listCameras( self.conn )
    self.matches = listMatches( self.conn )
    self.clipCamera[ "values" ] = [ f"{item.id}: {item.name}" for item in self.cameras ]
    self.clipMatch[ "values" ] = [ f"{item.id}: {item.date} {item.home} v {item.away}" for item in self.matches ]
    self.refreshClipOrder()

  def selectedID( self, widget: ttk.Combobox ) -> int | None:
    value = widget.get().split( ":", 1 )[ 0 ]
    return int( value ) if value.isdigit() else None

  def refreshClipOrder( self ):
    if not hasattr( self, "clipOrderTree" ):
      return
    self.clipOrderTree.delete( *self.clipOrderTree.get_children() )
    camera_id = self.selectedID( self.clipCamera )
    match_id = self.selectedID( self.clipMatch )
    if camera_id is None or match_id is None:
      return
    videos = { video.id: video.file for video in listVideos( self.conn ) }
    for clip in listClips( self.conn, camera_id=camera_id, match_id=match_id ):
      self.clipOrderTree.insert( "", "end", iid=str( clip.id ), values=( clip.sequence, videos.get( clip.video_id, clip.video_id ), clip.id ) )

  def addClips( self ):
    camera_id = self.selectedID( self.clipCamera )
    match_id = self.selectedID( self.clipMatch )
    if camera_id is None or match_id is None:
      messagebox.showerror( "Missing selection", "Select a Camera and Match first.", parent=self.window )
      return
    existing = listClips( self.conn, camera_id=camera_id, match_id=match_id )
    next_sequence = max( ( clip.sequence for clip in existing ), default=-1 ) + 1
    existing_video_ids = { clip.video_id for clip in existing }
    videos = listVideos( self.conn )
    selected = set( self.clipVideos.curselection() )
    try:
      for index, video in enumerate( videos ):
        if index in selected and video.id not in existing_video_ids:
          upsertClip( self.conn, ClipDB( 0, video.id, match_id, camera_id, next_sequence ) )
          next_sequence += 1
      self.conn.commit()
      self.refreshClipOrder()
    except duckdb.Error as error:
      messagebox.showerror( "Database error", str( error ), parent=self.window )

  def moveClip( self, offset: int ):
    selected = self.clipOrderTree.selection()
    if not selected:
      return
    index = self.clipOrderTree.index( selected[ 0 ] )
    target = index + offset
    children = self.clipOrderTree.get_children()
    if target < 0 or target >= len( children ):
      return
    self.clipOrderTree.move( selected[ 0 ], "", target )
    self.clipOrderTree.selection_set( selected[ 0 ] )

  def startClipDrag( self, event ):
    item = self.clipOrderTree.identify_row( event.y )
    self.clipDragItem = item if item else None

  def dragClip( self, event ):
    if not getattr( self, "clipDragItem", None ):
      return
    target = self.clipOrderTree.identify_row( event.y )
    if target and target != self.clipDragItem:
      self.clipOrderTree.move( self.clipDragItem, "", self.clipOrderTree.index( target ) )
      self.clipOrderTree.selection_set( self.clipDragItem )

  def finishClipDrag( self, _event ):
    self.clipDragItem = None

  def saveClipOrder( self ):
    clips = []
    for sequence, item_id in enumerate( self.clipOrderTree.get_children() ):
      values = self.clipOrderTree.item( item_id, "values" )
      clips.append( ClipDB( int( values[ 2 ] ), 0, 0, 0, sequence ) )
    try:
      reorderClips( self.conn, clips )
      self.refreshClipOrder()
    except duckdb.Error as error:
      messagebox.showerror( "Database error", str( error ), parent=self.window )

  def buildPersons( self ):
    tab = ttk.Frame( self.notebook )
    self.notebook.add( tab, text="Persons" )
    self.personTree = self.makeTree( tab, ( "id", "first", "last" ), ( "ID", "First name", "Last name" ) )
    ttk.Button( tab, text="Refresh", command=self.refreshPersons ).pack( side="bottom", pady=6 )
    self.refreshPersons()

  def refreshPersons( self ):
    self.personTree.delete( *self.personTree.get_children() )
    for person in listPersons( self.conn ):
      self.personTree.insert( "", "end", values=( person.id, person.first_name, person.last_name ) )

  @staticmethod
  def clearForm( variables: list[ tk.StringVar ] ):
    for variable in variables:
      variable.set( "" )
