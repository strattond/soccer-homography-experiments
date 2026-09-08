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
    ( "frame", pa.int32() ),
    ( "track", pa.int32() ),

    # Smoothed bounding box (Kalman-filtered)
    ( "x1", pa.float32() ),
    ( "y1", pa.float32() ),
    ( "x2", pa.float32() ),
    ( "y2", pa.float32() ),

    # Predicted bounding box (when detection missing)
    ( "pred_x1", pa.float32() ),
    ( "pred_y1", pa.float32() ),
    ( "pred_x2", pa.float32() ),
    ( "pred_y2", pa.float32() ),

    # Lifecycle
    ( "is_confirmed", pa.bool_() ),
    ( "is_tentative", pa.bool_() ),
    ( "is_deleted", pa.bool_() ),

    # Age + time since update
    ( "age", pa.int32() ),
    ( "time_since_update", pa.int32() ),

    # Trajectory history (store as list of float tuples)
    ( "history", pa.list_( pa.list_( pa.float32() ) ) ),
] )


def dictToArrow( records: list[ Track ] ) -> pa.Table:
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

  return pa.Table.from_pylist( flat, schema=BBOX_SCHEMA )


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

    tracks[ tid ].boxes.append( BoundingBox( row[ 'x1' ], row[ 'y1' ], row[ 'x2' ], row[ 'y2' ], row[ 'confidence' ], row[ 'frame' ], row[ 'det_class' ] ) )

  return list( tracks.values() )
