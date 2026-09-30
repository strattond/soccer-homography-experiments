import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pyarrow.parquet as pq
import pytest

from soccer_homography.dataTypes import BoundingBox, Track
from soccer_homography.db import (
    AsyncChunkWriter,
    readDetectionChunks,
    readTrackingChunks,
    writeBatchDetections,
    writeBatchTracking,
)

T = TypeVar( "T" )


class ManualScheduler:

  def __init__( self ) -> None:
    self.callbacks: dict[ str, Callable[ [], None ] ] = {}
    self.next_id = 0
    self.scheduled_delays: list[ int ] = []

  def after( self, delay: int, callback: Callable[ [], None ] ) -> str:
    self.next_id += 1
    callback_id = str( self.next_id )
    self.callbacks[ callback_id ] = callback
    self.scheduled_delays.append( delay )
    return callback_id

  def after_cancel( self, callback_id: str ) -> None:
    self.callbacks.pop( callback_id, None )

  def run_pending( self ) -> None:
    callbacks = list( self.callbacks.values() )
    self.callbacks.clear()
    for callback in callbacks:
      callback()


def flush_writer[T]( writer: AsyncChunkWriter[ T ], scheduler: ManualScheduler ) -> None:
  deadline = time.monotonic() + 5
  while writer.futures and time.monotonic() < deadline:
    time.sleep( 0.01 )
    scheduler.run_pending()
  assert not writer.futures


def test_async_chunk_writer_serializes_submissions_and_reports_errors() -> None:
  scheduler = ManualScheduler()
  written: list[ tuple[ int, int, list[ int ] ] ] = []
  errors: list[ tuple[ int, Exception ] ] = []

  def write_batch( clip_id: int, chunk_id: int, records: list[ int ] ) -> None:
    written.append( ( clip_id, chunk_id, records ) )
    if chunk_id == 2:
      raise OSError( "write failed" )

  writer = AsyncChunkWriter(
      "test",
      write_batch,
      scheduler.after,
      scheduler.after_cancel,
      lambda chunk_id, error: errors.append( ( chunk_id, error ) ),
      len,
  )
  try:
    writer.submit( 7, 1, [ 10 ] )
    writer.submit( 7, 2, [ 20, 21 ] )
    flush_writer( writer, scheduler )

    assert written == [ ( 7, 1, [ 10 ] ), ( 7, 2, [ 20, 21 ] ) ]
    assert len( errors ) == 1
    assert errors[ 0 ][ 0 ] == 2
    assert isinstance( errors[ 0 ][ 1 ], OSError )
  finally:
    writer.shutdown()


def test_detection_and_tracking_writers_write_separate_parquet_chunks( tmp_path, monkeypatch ) -> None:
  monkeypatch.chdir( tmp_path )
  Path( "tracking" ).mkdir()
  scheduler = ManualScheduler()
  errors: list[ tuple[ str, int, Exception ] ] = []
  detections = { 4: [ BoundingBox( 1, 2, 3, 4, 0.9, 0, 4 ) ] }
  tracks = [ Track( 4, 9, boxes=[ BoundingBox( 5, 6, 7, 8, 0.8, 0, 4 ) ] ) ]
  detection_writer = AsyncChunkWriter(
      "detection",
      writeBatchDetections,
      scheduler.after,
      scheduler.after_cancel,
      lambda chunk_id, error: errors.append( ( "detection", chunk_id, error ) ),
      lambda records: sum( len( boxes ) for boxes in records.values() ),
  )
  tracking_writer = AsyncChunkWriter(
      "tracking",
      writeBatchTracking,
      scheduler.after,
      scheduler.after_cancel,
      lambda chunk_id, error: errors.append( ( "tracking", chunk_id, error ) ),
      lambda records: sum( len( track.boxes ) for track in records ),
  )
  try:
    detection_writer.submit( 4, 0, detections )
    tracking_writer.submit( 4, 0, tracks )
    flush_writer( detection_writer, scheduler )
    flush_writer( tracking_writer, scheduler )

    assert pq.read_table( "tracking/chunk_detections_4_0.parquet" ).num_rows == 1
    assert pq.read_table( "tracking/chunk_tracking_4_0.parquet" ).num_rows == 1
    loaded_detections = readDetectionChunks( 4 )
    loaded_tracks = readTrackingChunks( 4 )
    assert loaded_detections[ 4 ][ 0 ].conf == pytest.approx( detections[ 4 ][ 0 ].conf )
    assert loaded_tracks[ 9 ].boxes[ 0 ].conf == pytest.approx( tracks[ 0 ].boxes[ 0 ].conf )
    assert not errors
  finally:
    detection_writer.shutdown()
    tracking_writer.shutdown()
