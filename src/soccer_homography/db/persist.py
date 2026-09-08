from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass( slots=True )
class Video:
  id: int
  file: str


@dataclass( slots=True )
class Match:
  id: int
  date: str
  home: str
  away: str
  division: str


@dataclass( slots=True )
class Camera:
  id: int
  name: str


@dataclass( slots=True )
class Clip:
  id: int
  video_id: int
  match_id: int
  camera_id: int
  sequence: int


def nextPK( conn: duckdb.DuckDBPyConnection, table_name: str ) -> int:
  row = conn.execute( f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table_name}" ).fetchone()
  if row is None:
    raise ValueError( f"Unable to determine next primary key for table {table_name}" )
  return int( row[ 0 ] )


def getConn( db_path: str | Path = "soccer_homography.db" ) -> duckdb.DuckDBPyConnection:
  return duckdb.connect( str( db_path ) )


def initDB( db_path: str | Path = "soccer_homography.db" ) -> duckdb.DuckDBPyConnection:
  conn = getConn( db_path )
  conn.execute(
      """
    CREATE TABLE IF NOT EXISTS videos (
      id INTEGER PRIMARY KEY,
      file VARCHAR NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS matches (
      id INTEGER PRIMARY KEY,
      date VARCHAR,
      home VARCHAR,
      away VARCHAR,
      division VARCHAR
    );

    CREATE TABLE IF NOT EXISTS cameras (
      id INTEGER PRIMARY KEY,
      name VARCHAR NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS clips (
      id INTEGER PRIMARY KEY,
      video_id INTEGER NOT NULL,
      match_id INTEGER NOT NULL,
      camera_id INTEGER NOT NULL,
      sequence INTEGER NOT NULL,
      UNIQUE(video_id, match_id, camera_id, sequence),
      FOREIGN KEY(video_id) REFERENCES videos(id),
      FOREIGN KEY(match_id) REFERENCES matches(id),
      FOREIGN KEY(camera_id) REFERENCES cameras(id)
    );
  """
  )
  return conn


def upsertVideo( conn: duckdb.DuckDBPyConnection, video: Video ) -> Video:
  if not video.file:
    raise ValueError( "Video file is required" )

  existing = conn.execute( "SELECT id FROM videos WHERE file = ?", [ video.file ] ).fetchone()
  if existing is not None:
    video.id = int( existing[ 0 ] )
    return video

  video.id = nextPK( conn, "videos" )
  conn.execute( "INSERT INTO videos(id, file) VALUES (?, ?)", [ video.id, video.file ] )
  return video


def upsertMatch( conn: duckdb.DuckDBPyConnection, match: Match ) -> Match:
  existing = conn.execute(
      "SELECT id FROM matches WHERE date = ? AND home = ? AND away = ? AND division = ?",
      [ match.date, match.home, match.away, match.division ],
  ).fetchone()
  if existing is not None:
    match.id = int( existing[ 0 ] )
    return match

  match.id = nextPK( conn, "matches" )
  conn.execute(
      "INSERT INTO matches(id, date, home, away, division) VALUES (?, ?, ?, ?, ?)",
      [ match.id, match.date, match.home, match.away, match.division ],
  )
  return match


def upsertCamera( conn: duckdb.DuckDBPyConnection, camera: Camera ) -> Camera:
  if not camera.name:
    raise ValueError( "Camera name is required" )

  existing = conn.execute( "SELECT id FROM cameras WHERE name = ?", [ camera.name ] ).fetchone()
  if existing is not None:
    camera.id = int( existing[ 0 ] )
    return camera

  camera.id = nextPK( conn, "cameras" )
  conn.execute( "INSERT INTO cameras(id, name) VALUES (?, ?)", [ camera.id, camera.name ] )
  return camera


def saveClip( conn: duckdb.DuckDBPyConnection, clip: Clip ) -> Clip:
  if clip.id <= 0:
    clip.id = nextPK( conn, "clips" )

  existing = conn.execute(
      "SELECT id FROM clips WHERE video_id = ? AND match_id = ? AND camera_id = ? AND sequence = ?",
      [ clip.video_id, clip.match_id, clip.camera_id, clip.sequence ],
  ).fetchone()

  if existing is not None:
    clip.id = int( existing[ 0 ] )
    conn.execute(
        "UPDATE clips SET video_id = ?, match_id = ?, camera_id = ?, sequence = ? WHERE id = ?",
        [ clip.video_id, clip.match_id, clip.camera_id, clip.sequence, clip.id ],
    )
    return clip

  conn.execute(
      "INSERT INTO clips(id, video_id, match_id, camera_id, sequence) VALUES (?, ?, ?, ?, ?)",
      [ clip.id, clip.video_id, clip.match_id, clip.camera_id, clip.sequence ],
  )
  return clip


def getClip( conn: duckdb.DuckDBPyConnection, video_id: int, match_id: int, camera_id: int, sequence: int ) -> Clip | None:
  row = conn.execute(
      "SELECT id, video_id, match_id, camera_id, sequence FROM clips WHERE video_id = ? AND match_id = ? AND camera_id = ? AND sequence = ?",
      [ video_id, match_id, camera_id, sequence ],
  ).fetchone()

  if row is None:
    return None

  return Clip( id=int( row[ 0 ] ), video_id=int( row[ 1 ] ), match_id=int( row[ 2 ] ), camera_id=int( row[ 3 ] ), sequence=int( row[ 4 ] ) )


def getClipByID( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> Clip | None:
  row = conn.execute(
      "SELECT id, video_id, match_id, camera_id, sequence FROM clips WHERE id = ?",
      [ clip_id ],
  ).fetchone()

  if row is None:
    return None

  return Clip( id=int( row[ 0 ] ), video_id=int( row[ 1 ] ), match_id=int( row[ 2 ] ), camera_id=int( row[ 3 ] ), sequence=int( row[ 4 ] ) )


def listClips( conn: duckdb.DuckDBPyConnection, *, video_id: int | None = None, match_id: int | None = None, camera_id: int | None = None ) -> list[ Clip ]:
  clauses: list[ str ] = []
  params: list[ int ] = []

  if video_id is not None:
    clauses.append( "video_id = ?" )
    params.append( video_id )
  if match_id is not None:
    clauses.append( "match_id = ?" )
    params.append( match_id )
  if camera_id is not None:
    clauses.append( "camera_id = ?" )
    params.append( camera_id )

  sql = "SELECT id, video_id, match_id, camera_id, sequence FROM clips"
  if clauses:
    sql += " WHERE " + " AND ".join( clauses )
  sql += " ORDER BY video_id, match_id, camera_id, sequence"

  rows = conn.execute( sql, params ).fetchall()
  return [ Clip( id=int( row[ 0 ] ), video_id=int( row[ 1 ] ), match_id=int( row[ 2 ] ), camera_id=int( row[ 3 ] ), sequence=int( row[ 4 ] ) ) for row in rows ]


def getVideoByID( conn: duckdb.DuckDBPyConnection, video_id: int ) -> Video | None:
  row = conn.execute( "SELECT id, file FROM videos WHERE id = ?", [ video_id ] ).fetchone()
  if row is None:
    return None
  return Video( id=int( row[ 0 ] ), file=str( row[ 1 ] ) )


def getMatchByID( conn: duckdb.DuckDBPyConnection, match_id: int ) -> Match | None:
  row = conn.execute( "SELECT id, date, home, away, division FROM matches WHERE id = ?", [ match_id ] ).fetchone()
  if row is None:
    return None
  return Match( id=int( row[ 0 ] ), date=str( row[ 1 ] ), home=str( row[ 2 ] ), away=str( row[ 3 ] ), division=str( row[ 4 ] ) )


def getCameraByID( conn: duckdb.DuckDBPyConnection, camera_id: int ) -> Camera | None:
  row = conn.execute( "SELECT id, name FROM cameras WHERE id = ?", [ camera_id ] ).fetchone()
  if row is None:
    return None
  return Camera( id=int( row[ 0 ] ), name=str( row[ 1 ] ) )


def readTrackingForClip( conn: duckdb.DuckDBPyConnection, parquet_path: str | Path, clip: Clip ) -> duckdb.DuckDBPyRelation:
  """Read the tracking rows for a clip by joining the compound clip identity to a parquet dataset."""
  result: duckdb.DuckDBPyRelation = conn.sql(
      """
      SELECT t.*
      FROM read_parquet(?) AS t
      WHERE t.video_id = ?
        AND t.match_id = ?
        AND t.camera_id = ?
        AND t.sequence = ?
      ORDER BY t.frame
      """,
      [ str( parquet_path ), clip.video_id, clip.match_id, clip.camera_id, clip.sequence ],
  )
  return result
