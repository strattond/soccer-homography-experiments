from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb


# Database representations
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
class ClipDB:
  id: int
  video_id: int
  match_id: int
  camera_id: int
  sequence: int


@dataclass( slots=True )
class Person:
  id: int
  first_name: str
  last_name: str


@dataclass( slots=True )
class PersonParticipationDB:
  match_id: int
  person_id: int
  shirt_number: int
  role: str  # Must be 'home', 'away', or 'referee'


# More business representations
@dataclass( slots=True )
class Clip:
  id: int
  video_id: Video
  match_id: Match
  camera_id: Camera
  sequence: int


@dataclass( slots=True )
class PersonParticipation:
  match_id: Match
  person_id: Person
  shirt_number: int
  role: str  # Must be 'home', 'away', or 'referee'


def getConn( db_path: str | Path = "soccer_homography.db" ) -> duckdb.DuckDBPyConnection:
  return duckdb.connect( str( db_path ) )


def initDB( db_path: str | Path = "soccer_homography.db" ) -> duckdb.DuckDBPyConnection:
  conn = getConn( db_path )
  conn.execute(
      """
    CREATE TABLE IF NOT EXISTS videos (
      id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      file VARCHAR NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS matches (
      id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      date VARCHAR,
      home VARCHAR,
      away VARCHAR,
      division VARCHAR
    );

    CREATE TABLE IF NOT EXISTS cameras (
      id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      name VARCHAR NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS clips (
      id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      video_id INTEGER NOT NULL,
      match_id INTEGER NOT NULL,
      camera_id INTEGER NOT NULL,
      sequence INTEGER NOT NULL,
      UNIQUE(video_id, match_id, camera_id, sequence),
      FOREIGN KEY(video_id) REFERENCES videos(id),
      FOREIGN KEY(match_id) REFERENCES matches(id),
      FOREIGN KEY(camera_id) REFERENCES cameras(id)
    );

    CREATE TABLE Person (
      person_id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
      first_name VARCHAR NOT NULL,
      last_name  VARCHAR NOT NULL
    );

    CREATE TABLE PersonParticipation (
      match_id INTEGER NOT NULL,
      person_id INTEGER NOT NULL,
      shirt_number INTEGER,
      role VARCHAR NOT NULL CHECK (role IN ('home', 'away', 'referee')),
      PRIMARY KEY (match_id, person_id),
      FOREIGN KEY (match_id) REFERENCES Match(match_id),
      FOREIGN KEY (person_id) REFERENCES Person(person_id)
    );

  """
  )
  return conn


def getLastInsertedID( conn: duckdb.DuckDBPyConnection ) -> int:
  row = conn.execute( "SELECT last_insert_rowid()" ).fetchone()
  if row is None:
    raise ValueError( "Unable to retrieve last inserted ID" )
  return int( row[ 0 ] )


def upsertVideo( conn: duckdb.DuckDBPyConnection, video: Video ) -> Video:
  existing = None
  if video.id > 0:
    existing = conn.execute( "SELECT file FROM videos WHERE id = ?", [ video.id ] ).fetchone()

  if existing is not None:
    conn.execute( "UPDATE videos SET file = ? WHERE id = ?", [ video.file, video.id ] )
  else:
    conn.execute( "INSERT INTO videos(file) VALUES (?)", [ video.file ] )
    video.id = getLastInsertedID( conn )
  return video


def upsertMatch( conn: duckdb.DuckDBPyConnection, match: Match ) -> Match:
  existing = None
  if match.id > 0:
    existing = conn.execute( "SELECT date, home, away FROM matches WHERE id = ?", [ match.id ] ).fetchone()
  if existing is not None:
    conn.execute( "UPDATE matches SET date = ?, home = ?, away = ?, division = ? WHERE id = ?", [ match.date, match.home, match.away, match.division, match.id ] )
  else:
    conn.execute(
        "INSERT INTO matches(date, home, away, division) VALUES (?, ?, ?, ?)",
        [ match.date, match.home, match.away, match.division ],
    )
    match.id = getLastInsertedID( conn )
  return match


def upsertCamera( conn: duckdb.DuckDBPyConnection, camera: Camera ) -> Camera:
  existing = None
  if camera.id > 0:
    existing = conn.execute( "SELECT id FROM cameras WHERE id = ?", [ camera.id ] ).fetchone()
  if existing is not None:
    conn.execute( "UPDATE cameras SET name = ? WHERE id = ?", [ camera.name, camera.id ] )
  else:
    conn.execute( "INSERT INTO cameras(name) VALUES (?)", [ camera.name ] )
    camera.id = getLastInsertedID( conn )
  return camera


def upsertClip( conn: duckdb.DuckDBPyConnection, clip: ClipDB ) -> ClipDB:
  existing = None
  if clip.id > 0:
    existing = conn.execute(
        "SELECT id FROM clips WHERE id = ?",
        [ clip.id ],
    ).fetchone()

  if existing is not None:
    conn.execute(
        "UPDATE clips SET video_id = ?, match_id = ?, camera_id = ?, sequence = ? WHERE id = ?",
        [ clip.video_id, clip.match_id, clip.camera_id, clip.sequence, clip.id ],
    )
  else:
    conn.execute(
        "INSERT INTO clips(video_id, match_id, camera_id, sequence) VALUES (?, ?, ?, ?)",
        [ clip.video_id, clip.match_id, clip.camera_id, clip.sequence ],
    )
    clip.id = getLastInsertedID( conn )
  return clip


def getClip( conn: duckdb.DuckDBPyConnection, video_id: int, match_id: int, camera_id: int, sequence: int ) -> ClipDB | None:
  row = conn.execute(
      "SELECT id, video_id, match_id, camera_id, sequence FROM clips WHERE video_id = ? AND match_id = ? AND camera_id = ? AND sequence = ?",
      [ video_id, match_id, camera_id, sequence ],
  ).fetchone()

  if row is None:
    return None

  return ClipDB( id=int( row[ 0 ] ), video_id=int( row[ 1 ] ), match_id=int( row[ 2 ] ), camera_id=int( row[ 3 ] ), sequence=int( row[ 4 ] ) )


def getClipByID( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> ClipDB | None:
  row = conn.execute(
      "SELECT id, video_id, match_id, camera_id, sequence FROM clips WHERE id = ?",
      [ clip_id ],
  ).fetchone()

  if row is None:
    return None

  return ClipDB( id=int( row[ 0 ] ), video_id=int( row[ 1 ] ), match_id=int( row[ 2 ] ), camera_id=int( row[ 3 ] ), sequence=int( row[ 4 ] ) )


def listClips( conn: duckdb.DuckDBPyConnection, *, video_id: int | None = None, match_id: int | None = None, camera_id: int | None = None ) -> list[ ClipDB ]:
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
  return [ ClipDB( id=int( row[ 0 ] ), video_id=int( row[ 1 ] ), match_id=int( row[ 2 ] ), camera_id=int( row[ 3 ] ), sequence=int( row[ 4 ] ) ) for row in rows ]


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


def readTrackingForClip( conn: duckdb.DuckDBPyConnection, parquet_path: str | Path, clip: ClipDB ) -> duckdb.DuckDBPyRelation:
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
      params=[ str( parquet_path ), clip.video_id, clip.match_id, clip.camera_id, clip.sequence ],
  )
  return result
