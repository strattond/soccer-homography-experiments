"""PyTest configuration and fixtures for soccer_homography tests."""

import os
from datetime import datetime
from pathlib import Path

import pytest

from soccer_homography.db.persist import Camera, ClipDB, Match, Video

# Test database directory
TEST_DB = os.path.join( Path( __file__ ).parent, "test_soccer_homography.db" )


@pytest.fixture
def clear_test_database():
  """Initialize and clean the test database before running tests."""
  # Use soccer_homography.db which imports the right duckdb version
  from soccer_homography.db import persist

  if os.path.exists( TEST_DB ):
    print( f"Removing {TEST_DB})")
    os.remove( TEST_DB )

  conn = persist.initDB( TEST_DB )
  try:
    yield conn
  finally:
    conn.close()


@pytest.fixture
def conn( clear_test_database ):
  """Get the freshly initialized test database connection."""
  return clear_test_database


@pytest.fixture
def video():
  """Create a sample Video fixture."""
  return Video( id=0, file="video_001.mp4" )


@pytest.fixture
def match():
  """Create a sample Match fixture."""
  fixedTZ = datetime.now().astimezone().tzinfo
  return Match( id=0, date=datetime( 2026, 9, 1, tzinfo=fixedTZ ), home="A", away="B", division="D" )


@pytest.fixture
def camera():
  """Create a sample Camera fixture."""
  return Camera( id=0, name="Cam-1" )

@pytest.fixture
def clip():
  """Create a sample ClipDB fixture."""
  return ClipDB( id=0, video_id=1, match_id=2, camera_id=3, sequence=1 )

@pytest.fixture
def updated_clip():
  """Create an updated ClipDB fixture with id > 0 for upsert tests."""
  return ClipDB( id=5, video_id=4, match_id=6, camera_id=7, sequence=2 )