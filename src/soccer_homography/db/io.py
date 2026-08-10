# io.py

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from soccer_homography.dataTypes import Person, Track, TrackData

# ---------------------------------------------------------
# Schema for writing tracking details to a Parquet file
# ---------------------------------------------------------

# yapf: disable
TRACKING_SCHEMA = pa.schema( [
                              ( "clip", pa.int32() ),
                              ( "frame", pa.int32() ),
                              ( "track", pa.int32() ),
                              ( "x1", pa.float32() ),
                              ( "y1", pa.float32() ),
                              ( "x2", pa.float32() ),
                              ( "y2", pa.float32() ),
                              ( "confidence", pa.float32() ),
                              ( "person", pa.int32() ),  # nullable
                            ] )
# yapf: enable


def dictToArrow( records: list[ Track ] ) -> pa.Table:
  flat = []
  for r in records:
    for t in r.boxes:
      actId = None
      if isinstance( r.person, Person ):
        actId = r.person.id
      elif isinstance( r.person, int ):
        actId = r.person
      flat.append( {
          "clip": 1,
          "frame": t.index,
          "track": r.id,
          "x1": float( t.x1 ),
          "y1": float( t.y1 ),
          "x2": float( t.x2 ),
          "y2": float( t.y2 ),
          "confidence": float( t.conf ),
          "person": actId,
      } )

  return pa.Table.from_pylist( flat, schema=TRACKING_SCHEMA )


def fileFromClipChunk( clipID: int, chunkID: int ) -> str:
  return f"tracking/chunk_{clipID}_{chunkID}.parquet"


def writeBatch( clipID: int, chunkID: int, records: list[ Track ] ):
  table = dictToArrow( records )
  path = fileFromClipChunk( clipID, chunkID )
  pq.write_table( table, path, compression="zstd" )


def readAllData( clipID: int ) -> pl.DataFrame | None:
  out_dir = Path( "tracking" )
  files = sorted( out_dir.glob( f"chunk_{clipID}_*.parquet" ) )

  if not files:
    return None

  # Concatenate all chunks vertically
  df = pl.concat( [ pl.read_parquet( f ) for f in files ], how="vertical" )

  # Ensure correct ordering
  df = df.sort( [ "track", "frame" ] )


def readBatch( clipID: int ) -> list[ Track ]:

  df = readAllData( clipID )
  if df is None:
    return []

  tracks = {}
  for row in df.iter_rows( named=True ):
    tid = row[ 'track' ]

    if tid not in tracks:
      tracks[ tid ] = Track( tid, None, [] )
      # At this point, we need to do a person lookup ...

    tracks[ tid ].boxes.append( TrackData( row[ 'x1' ], row[ 'y1' ], row[ 'x2' ], row[ 'y2' ], row[ 'confidence' ], row[ 'frame' ] ) )

  return list( tracks.values() )
