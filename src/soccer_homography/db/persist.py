from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb
from squadi_data.fixed import DivisionData

from soccer_homography.data import ParticipationRole

fixedTZ = datetime.now().astimezone().tzinfo


# Database representations
@dataclass( slots=True )
class Video:
  id: int
  file: str


@dataclass( slots=True )
class Match:
  id: int
  date: datetime | None
  home: str
  away: str
  division: str
  squadi_id: int | None = None


@dataclass( slots=True )
class MatchImportResult:
  matches_created: int = 0
  persons_created: int = 0
  participations_created: int = 0
  participations_updated: int = 0
  errors: list[ str ] = field( default_factory=list )


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
class ClipTrackDB:
  clip_id: int
  track_id: int
  person_id: int | None


@dataclass( slots=True )
class TrackSegmentDB:
  clip_id: int
  track_id: int
  person_id: int
  frame_start: int
  frame_end: int


@dataclass( slots=True )
class ClipHomographyDB:
  clip_id: int
  homography_id: int
  payload: dict[ str, Any ]
  frame_start: int | None
  frame_end: int | None


@dataclass( slots=True )
class Person:
  id: int
  first_name: str
  last_name: str


@dataclass( slots=True )
class PersonParticipationDB:
  match_id: int
  person_id: int
  shirt_number: int | None
  role: ParticipationRole
  is_placeholder: bool = False


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
  shirt_number: int | None
  role: ParticipationRole
  is_placeholder: bool = False


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
    CREATE SEQUENCE IF NOT EXISTS homography_seq;
    
    CREATE TABLE IF NOT EXISTS videos (
      id INTEGER PRIMARY KEY DEFAULT nextval('video_seq'),
      file VARCHAR NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS matches (
      id INTEGER PRIMARY KEY DEFAULT nextval('match_seq'),
      date TIMESTAMP,
      home VARCHAR,
      away VARCHAR,
      division VARCHAR,
      squadi_id INTEGER UNIQUE
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
      UNIQUE( video_id, match_id, camera_id, sequence ),
      FOREIGN KEY( video_id ) REFERENCES videos( id ),
      FOREIGN KEY( match_id ) REFERENCES matches( id ),
      FOREIGN KEY( camera_id ) REFERENCES cameras( id )
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
      role VARCHAR NOT NULL CHECK (role IN ('home_player', 'home_goalkeeper', 'away_player', 'away_goalkeeper', 'referee', 'unknown')),
      is_placeholder BOOLEAN NOT NULL DEFAULT FALSE,
      PRIMARY KEY( match_id, person_id ),
      FOREIGN KEY( match_id ) REFERENCES matches( id ),
      FOREIGN KEY( person_id ) REFERENCES Person( id )
    );

    CREATE TABLE IF NOT EXISTS ClipTrack (
      clip_id INTEGER NOT NULL,
      track_id INTEGER NOT NULL,
      person_id INTEGER,
      PRIMARY KEY (clip_id, track_id),
      FOREIGN KEY (clip_id) REFERENCES clips( id ),
      FOREIGN KEY (person_id) REFERENCES Person( id )
    );

    CREATE TABLE IF NOT EXISTS TrackSegment (
      clip_id INTEGER NOT NULL,
      track_id INTEGER NOT NULL,
      person_id INTEGER NOT NULL,
      frame_start INTEGER NOT NULL,
      frame_end INTEGER NOT NULL,
      PRIMARY KEY (clip_id, track_id, frame_start),
      FOREIGN KEY (clip_id) REFERENCES clips( id ),
      FOREIGN KEY (person_id) REFERENCES Person( id ),
      CHECK (frame_start <= frame_end)
    );

    CREATE TABLE IF NOT EXISTS Homographies (
      id INTEGER PRIMARY KEY DEFAULT nextval('homography_seq'),
      payload JSON NOT NULL
    );

    CREATE TABLE IF NOT EXISTS ClipHomography (
      clip_id INTEGER NOT NULL,
      homography_id INTEGER NOT NULL,
      frame_start INTEGER,
      frame_end INTEGER,
      PRIMARY KEY (clip_id, homography_id),
      FOREIGN KEY (clip_id) REFERENCES clips( id ),
      FOREIGN KEY (homography_id) REFERENCES Homographies( id ),
      CHECK ((frame_start IS NULL AND frame_end IS NULL) OR
             (frame_start IS NOT NULL AND frame_end IS NOT NULL AND frame_start <= frame_end))
    );

  """
  )
  return conn


@contextmanager
def transaction( conn: duckdb.DuckDBPyConnection ) -> Generator[ None, None, None ]:
  conn.begin()
  try:
    yield
    conn.commit()
  except BaseException:
    conn.rollback()
    raise


def parseMatchDate( value: str | datetime | None ) -> datetime | None:
  if value is None or value == "":
    return None
  if isinstance( value, datetime ):
    if value.utcoffset() is not None:
      raise ValueError( "Match dates must use local time without a timezone offset." )
    return value

  for date_format in ( "%Y%m%d%H%M", "%Y-%m-%d %H:%M", "%Y-%m-%d" ):
    try:
      return datetime.strptime( value, date_format ).astimezone( fixedTZ )
    except ValueError:
      continue

  try:
    parsed = datetime.fromisoformat( value )
  except ValueError as error:
    raise ValueError( f"Invalid match date {value!r}; expected YYYYMMDDHHmm or YYYY-MM-DD HH:MM." ) from error
  if parsed.utcoffset() is not None:
    raise ValueError( "Match dates must use local time without a timezone offset." )
  return parsed


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
    existing = conn.execute( "SELECT date, home, away, squadi_id FROM matches WHERE id = ?", [ match.id ] ).fetchone()
  if existing is not None:
    conn.execute(
        "UPDATE matches SET date = ?, home = ?, away = ?, division = ? WHERE id = ?",
        [ match.date, match.home, match.away, match.division, match.id ],
    )
    if match.squadi_id is not None and match.squadi_id != existing[ 3 ]:
      conn.execute( "UPDATE matches SET squadi_id = ? WHERE id = ?", [ match.squadi_id, match.id ] )
  else:
    result = conn.execute(
        "INSERT INTO matches(date, home, away, division, squadi_id) VALUES (?, ?, ?, ?, ?) RETURNING id",
        [ match.date, match.home, match.away, match.division, match.squadi_id ],
    ).fetchone()
    if not result is None:
      match.id = int( result[ 0 ] )
  return match


def listMatches( conn: duckdb.DuckDBPyConnection ) -> list[ Match ]:
  rows = conn.execute( "SELECT id, date, home, away, division, squadi_id FROM matches ORDER BY date, home, away" ).fetchall()
  return [
      Match(
          id=int( row[ 0 ] ),
          date=row[ 1 ],
          home=str( row[ 2 ] or "" ),
          away=str( row[ 3 ] or "" ),
          division=str( row[ 4 ] or "" ),
          squadi_id=int( row[ 5 ] ) if row[ 5 ] is not None else None,
      ) for row in rows
  ]


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


def listClipParticipants( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> list[ PersonParticipation ]:
  rows = conn.execute(
      """
      SELECT
        m.id, m.date, m.home, m.away, m.division, m.squadi_id,
        p.id, p.first_name, p.last_name,
        pp.shirt_number, pp.role, pp.is_placeholder
      FROM clips AS c
      JOIN matches AS m ON m.id = c.match_id
      JOIN PersonParticipation AS pp ON pp.match_id = m.id
      JOIN Person AS p ON p.id = pp.person_id
      WHERE c.id = ?
      ORDER BY pp.shirt_number NULLS LAST, p.last_name, p.first_name
      """,
      [ clip_id ],
  ).fetchall()
  return [
      PersonParticipation(
          match_id=Match(
              id=int( row[ 0 ] ),
              date=row[ 1 ],
              home=str( row[ 2 ] or "" ),
              away=str( row[ 3 ] or "" ),
              division=str( row[ 4 ] or "" ),
              squadi_id=int( row[ 5 ] ) if row[ 5 ] is not None else None,
          ),
          person_id=Person(
              id=int( row[ 6 ] ),
              first_name=str( row[ 7 ] ),
              last_name=str( row[ 8 ] ),
          ),
          shirt_number=int( row[ 9 ] ) if row[ 9 ] is not None else None,
          role=cast( ParticipationRole, str( row[ 10 ] ) ),
          is_placeholder=bool( row[ 11 ] ),
      ) for row in rows
  ]


def listClipTracks( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> dict[ int, ClipTrackDB ]:
  rows = conn.execute(
      "SELECT clip_id, track_id, person_id FROM ClipTrack WHERE clip_id = ? ORDER BY track_id",
      [ clip_id ],
  ).fetchall()
  return {
      int( row[ 1 ] ): ClipTrackDB(
          clip_id=int( row[ 0 ] ),
          track_id=int( row[ 1 ] ),
          person_id=int( row[ 2 ] ) if row[ 2 ] is not None else None,
      )
      for row in rows
  }


def deleteClipTracks( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> None:
  conn.execute( "DELETE FROM ClipTrack WHERE clip_id = ?", [ clip_id ] )


def upsertClipTrack( conn: duckdb.DuckDBPyConnection, clip_track: ClipTrackDB ) -> ClipTrackDB:
  row = conn.execute(
      """
      INSERT INTO ClipTrack(clip_id, track_id, person_id)
      VALUES (?, ?, ?)
      ON CONFLICT (clip_id, track_id) DO UPDATE
      SET person_id = excluded.person_id
      RETURNING clip_id, track_id, person_id
      """,
      [ clip_track.clip_id, clip_track.track_id, clip_track.person_id ],
  ).fetchone()
  if row is None:
    raise RuntimeError( f"Could not save track {clip_track.track_id} for clip {clip_track.clip_id}." )
  return ClipTrackDB(
      clip_id=int( row[ 0 ] ),
      track_id=int( row[ 1 ] ),
      person_id=int( row[ 2 ] ) if row[ 2 ] is not None else None,
  )


def listTrackSegments( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> list[ TrackSegmentDB ]:
  rows = conn.execute(
      """
      SELECT clip_id, track_id, person_id, frame_start, frame_end
      FROM TrackSegment
      WHERE clip_id = ?
      ORDER BY frame_start, track_id
      """,
      [ clip_id ],
  ).fetchall()
  return [
      TrackSegmentDB(
          clip_id=int( row[ 0 ] ),
          track_id=int( row[ 1 ] ),
          person_id=int( row[ 2 ] ),
          frame_start=int( row[ 3 ] ),
          frame_end=int( row[ 4 ] ),
      )
      for row in rows
  ]


def upsertTrackSegment( conn: duckdb.DuckDBPyConnection, segment: TrackSegmentDB ) -> TrackSegmentDB:
  if segment.frame_start < 0 or segment.frame_end < 0:
    raise ValueError( "Track segment frames must be non-negative." )
  if segment.frame_start > segment.frame_end:
    raise ValueError( "Track segment start frame must not exceed its end frame." )

  participant = conn.execute(
      """
      SELECT 1
      FROM clips AS c
      JOIN PersonParticipation AS pp ON pp.match_id = c.match_id
      WHERE c.id = ? AND pp.person_id = ?
      """,
      [ segment.clip_id, segment.person_id ],
  ).fetchone()
  if participant is None:
    raise ValueError( f"Person {segment.person_id} is not a participant in the match for clip {segment.clip_id}." )

  overlap = conn.execute(
      """
      SELECT 1
      FROM TrackSegment
      WHERE clip_id = ?
        AND track_id = ?
        AND frame_start <> ?
        AND frame_start <= ?
        AND frame_end >= ?
      LIMIT 1
      """,
      [ segment.clip_id, segment.track_id, segment.frame_start, segment.frame_end, segment.frame_start ],
  ).fetchone()
  if overlap is not None:
    raise ValueError(
        f"Track {segment.track_id} already has a segment overlapping frames "
        f"{segment.frame_start}-{segment.frame_end} in clip {segment.clip_id}."
    )

  row = conn.execute(
      """
      INSERT INTO TrackSegment(clip_id, track_id, person_id, frame_start, frame_end)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT (clip_id, track_id, frame_start) DO UPDATE
      SET person_id = excluded.person_id, frame_end = excluded.frame_end
      RETURNING clip_id, track_id, person_id, frame_start, frame_end
      """,
      [ segment.clip_id, segment.track_id, segment.person_id, segment.frame_start, segment.frame_end ],
  ).fetchone()
  if row is None:
    raise RuntimeError( f"Could not save segment for track {segment.track_id} in clip {segment.clip_id}." )
  return TrackSegmentDB(
      clip_id=int( row[ 0 ] ),
      track_id=int( row[ 1 ] ),
      person_id=int( row[ 2 ] ),
      frame_start=int( row[ 3 ] ),
      frame_end=int( row[ 4 ] ),
  )


def deleteTrackSegments( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> None:
  conn.execute( "DELETE FROM TrackSegment WHERE clip_id = ?", [ clip_id ] )


def getClipHomography( conn: duckdb.DuckDBPyConnection, clip_id: int ) -> ClipHomographyDB | None:
  row = conn.execute(
      """
      SELECT ch.clip_id, ch.homography_id, h.payload::VARCHAR, ch.frame_start, ch.frame_end
      FROM ClipHomography AS ch
      JOIN Homographies AS h ON h.id = ch.homography_id
      WHERE ch.clip_id = ?
      ORDER BY ch.homography_id DESC
      LIMIT 1
      """,
      [ clip_id ],
  ).fetchone()
  if row is None:
    return None
  payload = json.loads( str( row[ 2 ] ) )
  if not isinstance( payload, dict ):
    raise TypeError( f"Stored homography {row[ 1 ]} must contain a JSON object." )
  return ClipHomographyDB(
      clip_id=int( row[ 0 ] ),
      homography_id=int( row[ 1 ] ),
      payload=payload,
      frame_start=int( row[ 3 ] ) if row[ 3 ] is not None else None,
      frame_end=int( row[ 4 ] ) if row[ 4 ] is not None else None,
  )


def saveClipHomography(
    conn: duckdb.DuckDBPyConnection,
    clip_id: int,
    payload: dict[ str, Any ],
    homography_id: int | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> ClipHomographyDB:
  if ( frame_start is None ) != ( frame_end is None ):
    raise ValueError( "Homography frame range must specify both start and end, or neither for a locked-off camera." )
  if frame_start is not None and frame_end is not None and frame_start > frame_end:
    raise ValueError( "Homography frame range start must not exceed its end." )
  payload_json = json.dumps( payload )
  with transaction( conn ):
    if homography_id is None:
      row = conn.execute(
          "INSERT INTO Homographies(payload) VALUES (CAST(? AS JSON)) RETURNING id",
          [ payload_json ],
      ).fetchone()
      if row is None:
        raise RuntimeError( "Could not create homography record." )
      homography_id = int( row[ 0 ] )
      conn.execute(
          """
          INSERT INTO ClipHomography(clip_id, homography_id, frame_start, frame_end)
          VALUES (?, ?, ?, ?)
          """,
          [ clip_id, homography_id, frame_start, frame_end ],
      )
    else:
      association = conn.execute(
          "SELECT 1 FROM ClipHomography WHERE clip_id = ? AND homography_id = ?",
          [ clip_id, homography_id ],
      ).fetchone()
      if association is None:
        raise ValueError( f"Homography {homography_id} is not associated with clip {clip_id}." )
      conn.execute( "UPDATE Homographies SET payload = CAST(? AS JSON) WHERE id = ?", [ payload_json, homography_id ] )
      conn.execute(
          "UPDATE ClipHomography SET frame_start = ?, frame_end = ? WHERE clip_id = ? AND homography_id = ?",
          [ frame_start, frame_end, clip_id, homography_id ],
      )
  return ClipHomographyDB(
      clip_id=clip_id,
      homography_id=homography_id,
      payload=payload,
      frame_start=frame_start,
      frame_end=frame_end,
  )


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


def upsertPerson( conn: duckdb.DuckDBPyConnection, person: Person ) -> Person:
  existing = None
  if person.id > 0:
    existing = conn.execute( "SELECT id FROM Person WHERE id = ?", [ person.id ] ).fetchone()
  if existing is None:
    existing = conn.execute(
        "SELECT id FROM Person WHERE first_name = ? AND last_name = ? ORDER BY id LIMIT 1",
        [ person.first_name, person.last_name ],
    ).fetchone()

  if existing is not None:
    person.id = int( existing[ 0 ] )
    conn.execute(
        "UPDATE Person SET first_name = ?, last_name = ? WHERE id = ?",
        [ person.first_name, person.last_name, person.id ],
    )
  else:
    result = conn.execute(
        "INSERT INTO Person(first_name, last_name) VALUES (?, ?) RETURNING id",
        [ person.first_name, person.last_name ],
    ).fetchone()
    if result is None:
      raise RuntimeError( f"Could not create person {person.first_name!r} {person.last_name!r}." )
    person.id = int( result[ 0 ] )
  return person


def upsertPersonParticipation( conn: duckdb.DuckDBPyConnection, participation: PersonParticipationDB ) -> PersonParticipationDB:
  result = conn.execute(
      """
      INSERT INTO PersonParticipation(match_id, person_id, shirt_number, role, is_placeholder)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT (match_id, person_id) DO UPDATE
      SET shirt_number = excluded.shirt_number, role = excluded.role, is_placeholder = excluded.is_placeholder
      RETURNING match_id, person_id, shirt_number, role, is_placeholder
      """,
      [ participation.match_id, participation.person_id, participation.shirt_number, participation.role, participation.is_placeholder ],
  ).fetchone()
  if result is None:
    raise RuntimeError( f"Could not save participation for match {participation.match_id} and person {participation.person_id}." )
  return PersonParticipationDB(
      match_id=int( result[ 0 ] ),
      person_id=int( result[ 1 ] ),
      shirt_number=int( result[ 2 ] ) if result[ 2 ] is not None else None,
      role=participation.role,
      is_placeholder=bool( result[ 4 ] ),
  )


def createPlaceholderParticipation(
    conn: duckdb.DuckDBPyConnection,
    match_id: int,
    role: ParticipationRole = "unknown",
) -> PersonParticipationDB:
  with transaction( conn ):
    person_row = conn.execute(
        "INSERT INTO Person(first_name, last_name) VALUES ('Unknown opponent', '') RETURNING id"
    ).fetchone()
    if person_row is None:
      raise RuntimeError( "Could not create placeholder person." )
    person_id = int( person_row[ 0 ] )
    conn.execute(
        "UPDATE Person SET last_name = ? WHERE id = ?",
        [ f"Player {person_id}", person_id ],
    )
    return upsertPersonParticipation(
        conn,
        PersonParticipationDB(
            match_id=match_id,
            person_id=person_id,
            shirt_number=None,
            role=role,
            is_placeholder=True,
        ),
    )


def importSquadiDivision(
    conn: duckdb.DuckDBPyConnection,
    division: DivisionData,
    match_teams: dict[ int, tuple[ str, str ] ] | None = None,
) -> MatchImportResult:
  rVal = MatchImportResult()

  for fixture_wrapper in division.matches:
    fixture = fixture_wrapper.match
    try:
      squadi_id = int( fixture.id )
    except ( duckdb.Error, ValueError, RuntimeError ) as error:
      rVal.errors.append( f"Squadi match {fixture.id!r}: {error}" )
      continue

    try:
      match_date = parseMatchDate( fixture.date )
    except ValueError as error:
      match_date = None
      rVal.errors.append( f"Squadi match {squadi_id}: {error}" )

    try:
      team_names = ( match_teams or {} ).get( squadi_id )
      existing = conn.execute( "SELECT id, date, home, away FROM matches WHERE squadi_id = ?", [ squadi_id ] ).fetchone()
      if existing is None:
        with transaction( conn ):
          match = upsertMatch(
              conn,
              Match(
                  id=0,
                  date=match_date,
                  home=team_names[ 0 ] if team_names is not None else "",
                  away=team_names[ 1 ] if team_names is not None else "",
                  division=division.div.name,
                  squadi_id=squadi_id,
              ),
          )
        match_id = match.id
        rVal.matches_created += 1
      else:
        match_id = int( existing[ 0 ] )
        if match_date is not None or team_names is not None:
          with transaction( conn ):
            upsertMatch(
                conn,
                Match(
                    id=match_id,
                    date=match_date if match_date is not None else existing[ 1 ],
                    home=team_names[ 0 ] if team_names is not None else str( existing[ 2 ] or "" ),
                    away=team_names[ 1 ] if team_names is not None else str( existing[ 3 ] or "" ),
                    division=division.div.name,
                    squadi_id=squadi_id,
                ),
            )
    except ( duckdb.Error, ValueError, RuntimeError ) as error:
      rVal.errors.append( f"Squadi match {fixture.id!r}: {error}" )
      continue

    for player in fixture.players:
      try:
        full_name = player.name.strip()
        if not full_name:
          raise ValueError( f"Match {squadi_id} contains a player with an empty name." )
        name_parts = full_name.split( " ", 1 )
        first_name = name_parts[ 0 ]
        last_name = name_parts[ 1 ].strip() if len( name_parts ) > 1 else ""
        shirt_number = int( player.shirt )
        with transaction( conn ):
          existing_person = conn.execute(
              "SELECT id FROM Person WHERE first_name = ? AND last_name = ? ORDER BY id LIMIT 1",
              [ first_name, last_name ],
          ).fetchone()
          person = upsertPerson( conn, Person( id=0, first_name=first_name, last_name=last_name ) )
          existing_participation = conn.execute(
              "SELECT 1 FROM PersonParticipation WHERE match_id = ? AND person_id = ?",
              [ match_id, person.id ],
          ).fetchone()
          upsertPersonParticipation(
              conn,
              PersonParticipationDB(
                  match_id=match_id,
                  person_id=person.id,
                  shirt_number=shirt_number,
                  role="unknown",
              ),
          )
        if existing_person is None:
          rVal.persons_created += 1
        if existing_participation is None:
          rVal.participations_created += 1
        else:
          rVal.participations_updated += 1
      except ( duckdb.Error, ValueError, RuntimeError ) as error:
        rVal.errors.append( f"Squadi match {squadi_id}, player {player.name!r}: {error}" )

  return rVal


def reorderClips( conn: duckdb.DuckDBPyConnection, clips: list[ ClipDB ] ) -> None:
  if not clips:
    return
  with transaction( conn ):
    for offset, clip in enumerate( clips ):
      conn.execute( "UPDATE clips SET sequence = ? WHERE id = ?", [ -( offset + 1 ), clip.id ] )
    for offset, clip in enumerate( clips ):
      conn.execute( "UPDATE clips SET sequence = ? WHERE id = ?", [ offset, clip.id ] )


def getVideoByID( conn: duckdb.DuckDBPyConnection, video_id: int ) -> Video | None:
  row = conn.execute( "SELECT id, file FROM videos WHERE id = ?", [ video_id ] ).fetchone()
  if row is None:
    return None
  return Video( id=int( row[ 0 ] ), file=str( row[ 1 ] ) )


def getMatchByID( conn: duckdb.DuckDBPyConnection, match_id: int ) -> Match | None:
  row = conn.execute( "SELECT id, date, home, away, division, squadi_id FROM matches WHERE id = ?", [ match_id ] ).fetchone()
  if row is None:
    return None
  return Match(
      id=int( row[ 0 ] ),
      date=row[ 1 ],
      home=str( row[ 2 ] or "" ),
      away=str( row[ 3 ] or "" ),
      division=str( row[ 4 ] or "" ),
      squadi_id=int( row[ 5 ] ) if row[ 5 ] is not None else None,
  )


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
