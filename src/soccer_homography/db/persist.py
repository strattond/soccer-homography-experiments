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
    CREATE SEQUENCE IF NOT EXISTS video_seq;
    CREATE SEQUENCE IF NOT EXISTS match_seq;
    CREATE SEQUENCE IF NOT EXISTS camera_seq;
    CREATE SEQUENCE IF NOT EXISTS clip_seq;
    CREATE SEQUENCE IF NOT EXISTS person_seq;
    
    CREATE TABLE IF NOT EXISTS videos (
      id INTEGER PRIMARY KEY DEFAULT nextval('video_seq'),
      file VARCHAR NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS matches (
      id INTEGER PRIMARY KEY DEFAULT nextval('match_seq'),
      date VARCHAR,
      home VARCHAR,
      away VARCHAR,
      division VARCHAR
    );

    CREATE TABLE IF NOT EXISTS cameras (
      id INTEGER PRIMARY KEY DEFAULT nextval('camera_seq'),
      name VARCHAR NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS clips (
      id INTEGER PRIMARY KEY DEFAULT nextval('clip_seq'),
      video_id INTEGER NOT NULL,
      match_id INTEGER NOT NULL,
      camera_id INTEGER NOT NULL,
      sequence INTEGER NOT NULL,
      UNIQUE(video_id, match_id, camera_id, sequence),
      FOREIGN KEY(video_id) REFERENCES videos(id),
      FOREIGN KEY(match_id) REFERENCES matches(id),
      FOREIGN KEY(camera_id) REFERENCES cameras(id)
    );

    CREATE TABLE IF NOT EXISTS Person (
      id INTEGER PRIMARY KEY DEFAULT nextval('person_seq'),
      first_name VARCHAR NOT NULL,
      last_name  VARCHAR NOT NULL
    );

    CREATE TABLE IF NOT EXISTS PersonParticipation (
      match_id INTEGER NOT NULL,
      person_id INTEGER NOT NULL,
      shirt_number INTEGER,
      role VARCHAR NOT NULL CHECK (role IN ('home', 'away', 'referee')),
      PRIMARY KEY (match_id, person_id),
      FOREIGN KEY (match_id) REFERENCES matches(id),
      FOREIGN KEY (person_id) REFERENCES Person(id)
    );

  """
  )
  return conn


def upsertVideo( conn: duckdb.DuckDBPyConnection, video: Video ) -> Video:
  existing = None
  if video.id > 0:
    existing = conn.execute( "SELECT file FROM videos WHERE id = ?", [ video.id ] ).fetchone()

  if existing is not None:
    conn.execute( "UPDATE videos SET file = ? WHERE id = ?", [ video.file, video.id ] )
  else:
    result = conn.execute(
        "INSERT INTO videos(file) VALUES (?) RETURNING id",
        [ video.file ],
    ).fetchone()
    if result is not None:
      video.id = int( result[ 0 ] )
  return video


def listVideos( conn: duckdb.DuckDBPyConnection ) -> list[ Video ]:
  rows = conn.execute( "SELECT id, file FROM videos ORDER BY file" ).fetchall()
  return [ Video( id=int( row[ 0 ] ), file=str( row[ 1 ] ) ) for row in rows ]


def deleteVideo( conn: duckdb.DuckDBPyConnection, video_id: int ) -> None:
  conn.execute( "DELETE FROM videos WHERE id = ?", [ video_id ] )


def upsertMatch( conn: duckdb.DuckDBPyConnection, match: Match ) -> Match:
  existing = None
  if match.id > 0:
    existing = conn.execute( "SELECT date, home, away FROM matches WHERE id = ?", [ match.id ] ).fetchone()
  if existing is not None:
    conn.execute( "UPDATE matches SET date = ?, home = ?, away = ?, division = ? WHERE id = ?", [ match.date, match.home, match.away, match.division, match.id ] )
  else:
    result = conn.execute(
        "INSERT INTO matches(date, home, away, division) VALUES (?, ?, ?, ?) RETURNING id",
        [ match.date, match.home, match.away, match.division ],
    ).fetchone()
    if not result is None:
      match.id = int( result[ 0 ] )
  return match


def listMatches( conn: duckdb.DuckDBPyConnection ) -> list[ Match ]:
  rows = conn.execute( "SELECT id, date, home, away, division FROM matches ORDER BY date, home, away" ).fetchall()
  return [ Match( id=int( row[ 0 ] ), date=str( row[ 1 ] or "" ), home=str( row[ 2 ] or "" ), away=str( row[ 3 ] or "" ), division=str( row[ 4 ] or "" ) ) for row in rows ]


def upsertCamera( conn: duckdb.DuckDBPyConnection, camera: Camera ) -> Camera:
  existing = None
  if camera.id > 0:
    existing = conn.execute( "SELECT id FROM cameras WHERE id = ?", [ camera.id ] ).fetchone()
  if existing is not None:
    conn.execute( "UPDATE cameras SET name = ? WHERE id = ?", [ camera.name, camera.id ] )
  else:
    result = conn.execute( "INSERT INTO cameras(name) VALUES (?) RETURNING id", [ camera.name ] ).fetchone()
    if not result is None:
      camera.id = int( result[ 0 ] )
  return camera


def listCameras( conn: duckdb.DuckDBPyConnection ) -> list[ Camera ]:
  rows = conn.execute( "SELECT id, name FROM cameras ORDER BY name" ).fetchall()
  return [ Camera( id=int( row[ 0 ] ), name=str( row[ 1 ] ) ) for row in rows ]


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
    result = conn.execute(
        "INSERT INTO clips(video_id, match_id, camera_id, sequence) VALUES (?, ?, ?, ?) RETURNING id",
        [ clip.video_id, clip.match_id, clip.camera_id, clip.sequence ],
    ).fetchone()
    if not result is None:
      clip.id = int( result[ 0 ] )
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
  sql += " ORDER BY match_id, camera_id, sequence, video_id"

  rows = conn.execute( sql, params ).fetchall()
  return [ ClipDB( id=int( row[ 0 ] ), video_id=int( row[ 1 ] ), match_id=int( row[ 2 ] ), camera_id=int( row[ 3 ] ), sequence=int( row[ 4 ] ) ) for row in rows ]


def listPersons( conn: duckdb.DuckDBPyConnection ) -> list[ Person ]:
  rows = conn.execute( "SELECT id, first_name, last_name FROM Person ORDER BY last_name, first_name" ).fetchall()
  return [ Person( id=int( row[ 0 ] ), first_name=str( row[ 1 ] ), last_name=str( row[ 2 ] ) ) for row in rows ]


def reorderClips( conn: duckdb.DuckDBPyConnection, clips: list[ ClipDB ] ) -> None:
  if not clips:
    return
  conn.execute( "BEGIN TRANSACTION" )
  try:
    for offset, clip in enumerate( clips ):
      conn.execute( "UPDATE clips SET sequence = ? WHERE id = ?", [ -( offset + 1 ), clip.id ] )
    for offset, clip in enumerate( clips ):
      conn.execute( "UPDATE clips SET sequence = ? WHERE id = ?", [ offset, clip.id ] )
    conn.execute( "COMMIT" )
  except Exception:
    conn.execute( "ROLLBACK" )
    raise


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
