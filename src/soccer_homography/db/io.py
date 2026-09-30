from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from soccer_homography.dataTypes import BoundingBox, Track

BBOX_SCHEMA = pa.schema( [
    ( "clip", pa.int32() ),
    ( "frame", pa.int32() ),
    ( "x1", pa.float32() ),
    ( "y1", pa.float32() ),
    ( "x2", pa.float32() ),
    ( "y2", pa.float32() ),
    ( "cls", pa.int32() ),
    ( "confidence", pa.float32() ),
] )

TRACK_SCHEMA = pa.schema( [
    ( "clip", pa.int32() ),
    ( "frame", pa.int32() ),
    ( "track", pa.int32() ),
    ( "x1", pa.float32() ),
    ( "y1", pa.float32() ),
    ( "x2", pa.float32() ),
    ( "y2", pa.float32() ),
    ( "cls", pa.int32() ),
    ( "confidence", pa.float32() ),
] )


def boxToRow( clip_id: int, box: BoundingBox, track_id: int | None = None ) -> dict[ str, int | float | None ]:
  row: dict[ str, int | float | None ] = {
      "clip": clip_id,
      "frame": box.frame,
      "x1": float( box.x1 ),
      "y1": float( box.y1 ),
      "x2": float( box.x2 ),
      "y2": float( box.y2 ),
      "cls": box.cls,
      "confidence": float( box.conf ),
  }
  if track_id is not None:
    row[ "track" ] = track_id
  return row


def tracksToArrow( records: list[ Track ] ) -> pa.Table:
  flat = [
      boxToRow( track.clip, box, track.id )
      for track in records
      for box in track.boxes
  ]
  return pa.Table.from_pylist( flat, schema=TRACK_SCHEMA )


def boxesToArrow( clip_id: int, records: dict[ int, list[ BoundingBox ] ] ) -> pa.Table:
  flat = [
      boxToRow( clip_id, box )
      for boxes in records.values()
      for box in boxes
  ]
  return pa.Table.from_pylist( flat, schema=BBOX_SCHEMA )


def fileFromClipChunk( clipID: int, chunkID: int, type: str ) -> str:
  return f"tracking/chunk_{type}_{clipID}_{chunkID}.parquet"


def writeBatchDetections( clipID: int, chunkID: int, records: dict[ int, list[ BoundingBox ] ] ) -> None:
  path = Path( fileFromClipChunk( clipID, chunkID, "detections" ) )
  path.parent.mkdir( parents=True, exist_ok=True )
  pq.write_table( boxesToArrow( clipID, records ), path, compression="zstd" )


def writeBatchTracking( clipID: int, chunkID: int, records: list[ Track ] ) -> None:
  path = Path( fileFromClipChunk( clipID, chunkID, "tracking" ) )
  path.parent.mkdir( parents=True, exist_ok=True )
  pq.write_table( tracksToArrow( records ), path, compression="zstd" )


def _chunkFiles( clip_id: int, data_type: str ) -> list[ Path ]:
  return sorted( Path( "tracking" ).glob( f"chunk_{data_type}_{clip_id}_*.parquet" ) )


def _boundingBoxFromRow( row: dict ) -> BoundingBox:
  return BoundingBox(
      x1=int( row[ "x1" ] ),
      y1=int( row[ "y1" ] ),
      x2=int( row[ "x2" ] ),
      y2=int( row[ "y2" ] ),
      conf=float( row[ "confidence" ] ),
      cls=int( row[ "cls" ] ),
      frame=int( row[ "frame" ] ),
  )


def readDetectionChunks( clip_id: int ) -> dict[ int, list[ BoundingBox ] ]:
  detections: dict[ int, list[ BoundingBox ] ] = {}
  for path in _chunkFiles( clip_id, "detections" ):
    for row in pq.read_table( path ).to_pylist():
      frame = int( row[ "frame" ] )
      detections.setdefault( frame, [] ).append( _boundingBoxFromRow( row ) )
  for boxes in detections.values():
    boxes.sort( key=lambda box: box.frame )
  return detections


def readTrackingChunks( clip_id: int ) -> dict[ int, Track ]:
  tracks: dict[ int, Track ] = {}
  for path in _chunkFiles( clip_id, "tracking" ):
    for row in pq.read_table( path ).to_pylist():
      track_id = int( row[ "track" ] )
      track = tracks.setdefault( track_id, Track( clip_id, track_id ) )
      track.boxes.append( _boundingBoxFromRow( row ) )
  for track in tracks.values():
    track.boxes.sort( key=lambda box: box.frame )
  return tracks
