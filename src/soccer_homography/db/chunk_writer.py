import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TypeVar

from soccer_homography.log import logger

T = TypeVar( "T" )


class AsyncChunkWriter[T]:

  def __init__(
      self,
      name: str,
      write_batch: Callable[ [ int, int, T ], None ],
      schedule: Callable[ [ int, Callable[ [], None ] ], str ],
      cancel: Callable[ [ str ], None ],
      on_error: Callable[ [ int, Exception ], None ],
      item_count: Callable[ [ T ], int ],
      poll_interval_ms: int = 50,
  ) -> None:
    self.name = name
    self.write_batch = write_batch
    self.schedule = schedule
    self.cancel = cancel
    self.on_error = on_error
    self.item_count = item_count
    self.poll_interval_ms = poll_interval_ms
    self.executor = ThreadPoolExecutor( max_workers=1, thread_name_prefix=f"{name}-parquet" )
    self.futures: dict[ Future[ None ], int ] = {}
    self.poll_id: str | None = None
    self.closed = False

  def submit( self, clip_id: int, chunk_id: int, records: T ) -> None:
    if self.closed:
      raise RuntimeError( f"{self.name} chunk writer is closed" )
    future = self.executor.submit( self.writeChunk, clip_id, chunk_id, records )
    self.futures[ future ] = chunk_id
    self.schedulePoll()

  def writeChunk( self, clip_id: int, chunk_id: int, records: T ) -> None:
    started = time.perf_counter()
    self.write_batch( clip_id, chunk_id, records )
    count = self.item_count( records )
    logger.info( f"Wrote {self.name} chunk {chunk_id} ({count} items) in {time.perf_counter() - started:.2f}s" )

  def schedulePoll( self ) -> None:
    if self.poll_id is None and self.futures and not self.closed:
      self.poll_id = self.schedule( self.poll_interval_ms, self.poll )

  def poll( self ) -> None:
    self.poll_id = None
    for future, chunk_id in list( self.futures.items() ):
      if not future.done():
        continue
      del self.futures[ future ]
      try:
        future.result()
      except Exception as error:
        logger.exception( f"Failed to write {self.name} chunk {chunk_id}" )
        self.on_error( chunk_id, error )

    self.schedulePoll()

  def waitForPending( self ) -> None:
    if self.poll_id is not None:
      self.cancel( self.poll_id )
      self.poll_id = None
    pending = tuple( self.futures )
    if pending:
      wait( pending )
      self.poll()

  def shutdown( self ) -> None:
    self.closed = True
    if self.poll_id is not None:
      self.cancel( self.poll_id )
      self.poll_id = None
    self.executor.shutdown( wait=True )
    self.poll()
