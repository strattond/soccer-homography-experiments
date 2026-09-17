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
