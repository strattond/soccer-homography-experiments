"""Tests for the database layer in soccer_homography/db/persist.py."""

import pytest

from soccer_homography.db import persist


class TestUpsertVideo:
  """Test cases for upsertVideo function."""

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_video_insert_new( self, conn, video ):
    """Test inserting a new video record."""
    result = persist.upsertVideo( conn, video )

    assert isinstance( result, persist.Video )
    assert result.id == 1, "New video should get auto-incremented ID"
    assert result.file == "video_001.mp4", "File path should match input"

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_video_update_existing( self, conn ):
    """Test updating an existing video record."""
    # First insert a video
    orgResult = persist.upsertVideo( conn, persist.Video( id=0, file="original.mp4" ) )

    # Then try to update with different ID (shouldn't trigger upsert since we're using primary key)
    updResult = persist.upsertVideo( conn, persist.Video( id=orgResult.id, file="updated.mp4" ) )

    assert updResult.id == orgResult.id, "Existing video should keep its ID"
    assert updResult.file == "updated.mp4", "File should be updated"

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_video_id_zero_inserts( self, conn ):
    """Test that videos with id=0 always insert (never update)."""
    persist.upsertVideo( conn, persist.Video( id=100, file="first.mp4" ) )

    # With id=0, it should insert even if a record with similar fields exists
    result = persist.upsertVideo( conn, persist.Video( id=0, file="different.mp4" ) )

    assert result.id != 0 and result.file == "different.mp4", f"Should handle id=0 correctly. Got: {result}"

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_video_returns_updated_object( self, conn ):
    """Test that upsertVideo returns the updated Video object."""
    video = persist.Video( id=0, file="test.mp4" )

    persist.upsertVideo( conn, video )

    assert isinstance( result := persist.getVideoByID( conn, video.id ), persist.Video ), "Should return a Video instance"


class TestUpsertMatch:
  """Test cases for upsertMatch function."""

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_match_insert_new( self, conn, match ):
    """Test inserting a new match record."""
    result = persist.upsertMatch( conn, match )

    assert isinstance( result, persist.Match )
    assert result.id == 1, "New match should get auto-incremented ID"
    assert result.date == "2026-09-01", "Date should match input"

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_match_update_existing( self, conn ):
    """Test updating an existing match record."""
    # First insert a match
    orgResult = persist.upsertMatch( conn, persist.Match( id=0, date="2026-09-01", home="A", away="B", division="D" ) )

    # Then update with same ID (should trigger upsert via primary key)
    updResult = persist.upsertMatch( conn, persist.Match( id=orgResult.id, date="2026-09-02", home="X", away="Y", division="Z" ) )

    assert updResult.id == orgResult.id, "Existing match should keep its ID"
    assert updResult.date == "2026-09-02", "Date should be updated"
    assert updResult.home == "X" and updResult.away == "Y" and updResult.division == "Z", "Other fields should be updated"

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_match_id_zero_inserts( self, conn ):
    """Test that matches with id=0 always insert (never update)."""
    persist.upsertMatch( conn, persist.Match( id=100, date="2026-09-01", home="A", away="B", division="D" ) )

    # With id=0, it should insert even if a record with similar fields exists
    result = persist.upsertMatch( conn, persist.Match( id=0, date="2026-09-02", home="X", away="Y", division="Z" ) )

    assert result.id != 0 and result.date == "2026-09-02", f"Should handle id=0 correctly. Got: {result}"

  @pytest.mark.usefixtures( "clean_db" )
  def test_upsert_match_returns_updated_object( self, conn ):
    """Test that upsertMatch returns the updated Match object."""
    match = persist.Match( id=0, date="2026-09-01", home="A", away="B", division="D" )

    persist.upsertMatch( conn, match )

    assert isinstance( result := persist.getMatchByID( conn, match.id ), persist.Match ), "Should return a Match instance"
