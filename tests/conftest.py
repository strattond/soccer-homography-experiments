"""PyTest configuration and fixtures for soccer_homography tests."""

import os
from pathlib import Path

import pytest

from soccer_homography.db.persist import Camera, ClipDB, Match, Video

# Test database directory
TEST_DB = os.path.join( Path( __file__ ).parent, "test_soccer_homography.db" )


@pytest.fixture( scope="session", autouse=True )
def clear_test_database():
  """Initialize and clean the test database before running tests."""
  # Use soccer_homography.db which imports the right duckdb version
  from soccer_homography.db import persist

  if os.path.exists( TEST_DB ):
    os.remove( TEST_DB )
  conn = persist.getConn( TEST_DB )

  persist.initDB( TEST_DB )
  conn.close()


@pytest.fixture
def conn():
  """Get a fresh database connection for each test."""
  # Import DuckDB's Python API wrapper to ensure we get the right module
  from soccer_homography.db import persist

  conn = persist.getConn( TEST_DB )
  return conn


@pytest.fixture
def clean_db( conn ):
  """Get a cleaned database with initial data for testing upsert operations."""
  from soccer_homography.db import persist

  # Clear all tables
  tables = [ "clips", "person_participation", "person", "cameras", "matches", "videos" ]
  for table in reversed( tables ):
    try:
      conn.execute( f"DELETE FROM {table}" )
    except Exception:
      pass

  yield conn


@pytest.fixture
def video():
  """Create a sample Video fixture."""
  return Video( id=0, file="video_001.mp4" )


@pytest.fixture
def match():
  """Create a sample Match fixture."""
  return Match( id=0, date="2026-09-01", home="A", away="B", division="D" )
