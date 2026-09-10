# io.py

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from soccer_homography.dataTypes import BoundingBox, Track

# ---------------------------------------------------------
# Schema for writing tracking details to a Parquet file
# ---------------------------------------------------------

# yapf: disable
BBOX_SCHEMA = pa.schema( [
                              ( "clip", pa.int32() ),
                              ( "frame", pa.int32() ),
                              ( "x1", pa.float32() ),
                              ( "y1", pa.float32() ),
                              ( "x2", pa.float32() ),
                              ( "y2", pa.float32() ),
                              ( "cls", pa.int32() ),
                              ( "confidence", pa.float32() )
                            ] )
# yapf: enable

TRACK_SCHEMA = pa.schema( [
    # Identifiers
    ( "clip", pa.int32() ), ( "frame", pa.int32() ), ( "track", pa.int32() ),
    # Smoothed bounding box (Kalman-filtered)
    ( "x1", pa.float32() ), ( "y1", pa.float32() ), ( "x2", pa.float32() ), ( "y2", pa.float32() ),
    # Class and confidence
    ( "cls", pa.int32() ), ( "confidence", pa.float32() )
] )

TRACK_ASSOC_SCHEMA = pa.schema( [
    # Identifiers
    ( "clip", pa.int32() ), ( "track", pa.int32() ), ( "person", pa.int32() )
] )


def tracksToArrow( records: list[ Track ] ) -> pa.Table:
  flat = []
  for r in records:
    for t in r.boxes:
      flat.append( {
          "clip": 1,
          "frame": t.frame,
          "track": r.id,
          "x1": float( t.x1 ),
          "y1": float( t.y1 ),
          "x2": float( t.x2 ),
          "y2": float( t.y2 ),
          "cls": int( t.cls ),
          "confidence": float( t.conf )
      } )

  return pa.Table.from_pylist( flat, schema=TRACK_SCHEMA )


def boxesToArrow( records: dict[ int, list[ BoundingBox ] ] ) -> pa.Table:
  flat = []
  for k, v in records.items():
    for t in v:
      flat.append( {
          "clip": 1,
          "frame": t.frame,
          "track": k,
          "x1": float( t.x1 ),
          "y1": float( t.y1 ),
          "x2": float( t.x2 ),
          "y2": float( t.y2 ),
          "cls": int( t.cls ),
          "confidence": float( t.conf )
      } )

  return pa.Table.from_pylist( flat, schema=BBOX_SCHEMA )


def fileFromClipChunk( clipID: int, chunkID: int, type: str ) -> str:
  return f"tracking/chunk_{type}_{clipID}_{chunkID}.parquet"


def writeBatchDetections( clipID: int, chunkID: int, records: dict[ int, list[ BoundingBox ] ] ):
  table = boxesToArrow( records )
  path = fileFromClipChunk( clipID, chunkID, "detections" )
  pq.write_table( table, path, compression="zstd" )


def writeBatchTracking( clipID: int, chunkID: int, records: list[ Track ] ):
  table = tracksToArrow( records )
  path = fileFromClipChunk( clipID, chunkID, "tracking" )
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
      tracks[ tid ] = Track( clipID, tid, None, [] )
      # At this point, we need to do a person lookup ...

    tracks[ tid ].boxes.append( BoundingBox( row[ 'x1' ], row[ 'y1' ], row[ 'x2' ], row[ 'y2' ], row[ 'confidence' ], row[ 'frame' ], row[ 'det_class' ] ) )

  return list( tracks.values() )
