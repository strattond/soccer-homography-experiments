"""Tests for the database layer in soccer_homography/db/persist.py."""

from datetime import datetime

import pytest

from soccer_homography.db import persist

fixedTZ = datetime.now().astimezone().tzinfo

class TestUpsertVideo:
  """Test cases for upsertVideo function."""

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_video_insert_new( self, conn, video ):
    """Test inserting a new video record."""
    result = persist.upsertVideo( conn, video )

    assert isinstance( result, persist.Video )
    assert result.id == 1, "New video should get auto-incremented ID"
    assert result.file == "video_001.mp4", "File path should match input"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_video_update_existing( self, conn ):
    """Test updating an existing video record."""
    # First insert a video
    orgResult = persist.upsertVideo( conn, persist.Video( id=0, file="original.mp4" ) )

    # Then try to update with different ID (shouldn't trigger upsert since we're using primary key)
    updResult = persist.upsertVideo( conn, persist.Video( id=orgResult.id, file="updated.mp4" ) )

    assert updResult.id == orgResult.id, "Existing video should keep its ID"
    assert updResult.file == "updated.mp4", "File should be updated"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_video_id_zero_inserts( self, conn ):
    """Test that videos with id=0 always insert (never update)."""
    persist.upsertVideo( conn, persist.Video( id=100, file="first.mp4" ) )

    # With id=0, it should insert even if a record with similar fields exists
    result = persist.upsertVideo( conn, persist.Video( id=0, file="different.mp4" ) )

    assert result.id != 0 and result.file == "different.mp4", f"Should handle id=0 correctly. Got: {result}"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_video_returns_updated_object( self, conn ):
    """Test that upsertVideo returns the updated Video object."""
    video = persist.Video( id=0, file="test.mp4" )

    persist.upsertVideo( conn, video )

    assert isinstance( persist.getVideoByID( conn, video.id ), persist.Video ), "Should return a Video instance"


class TestUpsertMatch:
  """Test cases for upsertMatch function."""

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_match_insert_new( self, conn, match ):
    """Test inserting a new match record."""
    result = persist.upsertMatch( conn, match )

    assert isinstance( result, persist.Match )
    assert result.id == 1, "New match should get auto-incremented ID"
    assert result.date == datetime( 2026, 9, 1, tzinfo=fixedTZ ), "Date should match input"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_match_update_existing( self, conn ):
    """Test updating an existing match record."""
    # First insert a match
    orgResult = persist.upsertMatch( conn, persist.Match( id=0, date=datetime( 2026, 9, 1, tzinfo=fixedTZ ), home="A", away="B", division="D" ) )

    # Then update with same ID (should trigger upsert via primary key)
    updResult = persist.upsertMatch( conn, persist.Match( id=orgResult.id, date=datetime( 2026, 9, 2, tzinfo=fixedTZ ), home="X", away="Y", division="Z" ) )

    assert updResult.id == orgResult.id, "Existing match should keep its ID"
    assert updResult.date == datetime( 2026, 9, 2, tzinfo=fixedTZ ), "Date should be updated"
    assert updResult.home == "X" and updResult.away == "Y" and updResult.division == "Z", "Other fields should be updated"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_match_id_zero_inserts( self, conn ):
    """Test that matches with id=0 always insert (never update)."""
    persist.upsertMatch( conn, persist.Match( id=100, date=datetime( 2026, 9, 1, tzinfo=fixedTZ ), home="A", away="B", division="D" ) )

    # With id=0, it should insert even if a record with similar fields exists
    result = persist.upsertMatch( conn, persist.Match( id=0, date=datetime( 2026, 9, 2, tzinfo=fixedTZ ), home="X", away="Y", division="Z" ) )

    assert result.id != 0 and result.date == datetime( 2026, 9, 2, tzinfo=fixedTZ ), f"Should handle id=0 correctly. Got: {result}"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_match_returns_updated_object( self, conn ):
    """Test that upsertMatch returns the updated Match object."""
    match = persist.Match( id=0, date=datetime( 2026, 9, 1, tzinfo=fixedTZ ), home="A", away="B", division="D" )

    persist.upsertMatch( conn, match )

    assert isinstance( persist.getMatchByID( conn, match.id ), persist.Match ), "Should return a Match instance"


class TestUpsertCamera:
  """Test cases for upsertCamera function."""

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_camera_insert_new( self, conn, camera ):
    """Test inserting a new camera record."""
    result = persist.upsertCamera( conn, camera )

    assert isinstance( result, persist.Camera )
    assert result.id == 1, "New camera should get auto-incremented ID"
    assert result.name == "Cam-1", "Name should match input"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_camera_update_existing( self, conn ):
    """Test updating an existing camera record."""
    # First insert a camera
    orgResult = persist.upsertCamera( conn, persist.Camera( id=0, name="Cam-1" ) )

    # Then update with same ID (should trigger upsert via primary key)
    updResult = persist.upsertCamera( conn, persist.Camera( id=orgResult.id, name="Cam-99" ) )

    assert updResult.id == orgResult.id, "Existing camera should keep its ID"
    assert updResult.name == "Cam-99", "Name should be updated"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_camera_id_zero_inserts( self, conn ):
    """Test that cameras with id=0 always insert (never update)."""
    persist.upsertCamera( conn, persist.Camera( id=100, name="First-Cam" ) )

    # With id=0, it should insert even if a record with similar fields exists
    result = persist.upsertCamera( conn, persist.Camera( id=0, name="Different-Cam" ) )

    assert result.id != 0 and result.name == "Different-Cam", f"Should handle id=0 correctly. Got: {result}"

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_camera_returns_updated_object( self, conn ):
    """Test that upsertCamera returns the updated Camera object."""
    camera = persist.Camera( id=0, name="test-cam" )

    persist.upsertCamera( conn, camera )

    assert isinstance( persist.getCameraByID( conn, camera.id ), persist.Camera ), "Should return a Camera instance"

class TestUpsertClip:
  """Test cases for upsertClip function."""
  
  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_clip_insert_new( self, conn ):
    """Test inserting a new clip record after creating required parent records."""
    # First insert dependencies (video, match, camera)
    video = persist.Video( id=0, file="clip_video.mp4" )
    match = persist.Match( id=0, date=datetime( 2026, 9, 15, tzinfo=fixedTZ ), home="Team A", away="Team B", division="L" )
    camera = persist.Camera( id=0, name="Cam-A" )
    
    persist.upsertVideo( conn, video )
    persist.upsertMatch( conn, match )
    persist.upsertCamera( conn, camera )
    
    # Now insert clip with actual IDs
    clip_obj = persist.ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 )
    result = persist.upsertClip( conn, clip_obj )
    
    assert isinstance( result, persist.ClipDB ), "Should return a persist.ClipDB instance"
    assert result.id == 1, "New clip should get auto-incremented ID"
    assert result.video_id == video.id
    assert result.match_id == match.id
    assert result.camera_id == camera.id
    assert result.sequence == 1
  
  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_clip_update_existing( self, conn ):
    """Test updating an existing clip record."""
    # First create dependencies
    video = persist.Video( id=0, file="clip_video.mp4" )
    match = persist.Match( id=0, date=datetime( 2026, 9, 15, tzinfo=fixedTZ ), home="Team A", away="Team B", division="L" )
    camera = persist.Camera( id=0, name="Cam-A" )
    
    persist.upsertVideo( conn, video )
    persist.upsertMatch( conn, match )
    persist.upsertCamera( conn, camera )
    
    # Insert clip and get its ID
    clip_obj = persist.ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 )
    origResult = persist.upsertClip( conn, clip_obj )
    
    # Update with same ID but different fields
    updResult = persist.upsertClip( conn, persist.ClipDB( id=origResult.id, video_id=clip_obj.video_id, match_id=clip_obj.match_id, camera_id=clip_obj.camera_id, sequence=2 ) )
    
    assert updResult.id == origResult.id, "Existing clip should keep its ID"
    assert updResult.sequence == 2, "Sequence should be updated"
  
  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_clip_id_zero_inserts( self, conn ):
    """Test that clips with id=0 always insert (never update)."""
    # Create dependencies
    video = persist.Video( id=0, file="first.mp4" )
    match = persist.Match( id=0, date=datetime( 2026, 9, 15, tzinfo=fixedTZ ), home="A", away="B", division="D" )
    camera = persist.Camera( id=0, name="First-Cam" )
    
    persist.upsertVideo( conn, video )
    persist.upsertMatch( conn, match )
    persist.upsertCamera( conn, camera )
    
    # Insert first clip with auto-generated ID
    newClip = persist.ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 )
    persist.upsertClip( conn, newClip )
    
    # With id=0, it should insert even if a record with similar fields exists
    result = persist.upsertClip( conn, persist.ClipDB( id=0, video_id=newClip.video_id, match_id=newClip.match_id, camera_id=newClip.camera_id, sequence=99 ) )
    
    assert result.id != 0 and result.video_id == newClip.video_id and result.match_id == newClip.match_id, f"Should handle id=0 correctly. Got: {result}"
  
  @pytest.mark.usefixtures( "clear_test_database" )
  def test_upsert_clip_returns_updated_object( self, conn ):
    """Test that upsertClip returns the updated persist.ClipDB object."""
    # Create dependencies
    video = persist.Video( id=0, file="test.mp4" )
    match = persist.Match( id=0, date=datetime( 2026, 9, 15, tzinfo=fixedTZ ), home="A", away="B", division="D" )
    camera = persist.Camera( id=0, name="test-cam" )
    
    persist.upsertVideo( conn, video )
    persist.upsertMatch( conn, match )
    persist.upsertCamera( conn, camera )
    
    clip_obj = persist.ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 )
    persist.upsertClip( conn, clip_obj )
    
    assert isinstance( persist.getClipByID( conn, clip_obj.id ), persist.ClipDB ), "Should return a persist.ClipDB instance"


class TestClipParticipants:

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_list_clip_participants_returns_only_people_from_clip_match( self, conn ):
    video = persist.upsertVideo( conn, persist.Video( id=0, file="participants.mp4" ) )
    first_match = persist.upsertMatch( conn, persist.Match( id=0, date=None, home="A", away="B", division="D" ) )
    second_match = persist.upsertMatch( conn, persist.Match( id=0, date=None, home="C", away="D", division="D" ) )
    camera = persist.upsertCamera( conn, persist.Camera( id=0, name="participants-cam" ) )
    first_clip = persist.upsertClip(
        conn,
        persist.ClipDB( id=0, video_id=video.id, match_id=first_match.id, camera_id=camera.id, sequence=1 ),
    )
    second_clip = persist.upsertClip(
        conn,
        persist.ClipDB( id=0, video_id=video.id, match_id=second_match.id, camera_id=camera.id, sequence=1 ),
    )
    first_person = persist.upsertPerson( conn, persist.Person( id=0, first_name="Alex", last_name="Smith" ) )
    second_person = persist.upsertPerson( conn, persist.Person( id=0, first_name="Jordan", last_name="Lee" ) )
    other_match_person = persist.upsertPerson( conn, persist.Person( id=0, first_name="Taylor", last_name="Brown" ) )
    persist.upsertPersonParticipation(
        conn,
        persist.PersonParticipationDB( first_match.id, first_person.id, 7, "home_player" ),
    )
    persist.upsertPersonParticipation(
        conn,
        persist.PersonParticipationDB( first_match.id, second_person.id, 10, "away_player" ),
    )
    persist.upsertPersonParticipation(
        conn,
        persist.PersonParticipationDB( second_match.id, other_match_person.id, 1, "home_goalkeeper" ),
    )

    participants = persist.listClipParticipants( conn, first_clip.id )

    assert all( isinstance( participant, persist.PersonParticipation ) for participant in participants )
    assert [ participant.person_id.id for participant in participants ] == [ first_person.id, second_person.id ]
    assert participants[ 0 ].person_id.first_name == "Alex"
    assert participants[ 0 ].person_id.last_name == "Smith"
    assert participants[ 0 ].match_id.id == first_match.id
    assert participants[ 0 ].shirt_number == 7
    assert participants[ 0 ].role == "home_player"
    assert persist.listClipParticipants( conn, second_clip.id )[ 0 ].person_id.id == other_match_person.id


class TestClipTrack:

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_clip_track_upsert_allows_unknown_person_and_updates_role( self, conn ):
    video = persist.upsertVideo( conn, persist.Video( id=0, file="track.mp4" ) )
    match = persist.upsertMatch( conn, persist.Match( id=0, date=None, home="A", away="B", division="D" ) )
    camera = persist.upsertCamera( conn, persist.Camera( id=0, name="track-camera" ) )
    clip = persist.upsertClip( conn, persist.ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 ) )
    person = persist.upsertPerson( conn, persist.Person( id=0, first_name="Sam", last_name="Player" ) )

    unknown = persist.upsertClipTrack( conn, persist.ClipTrackDB( clip.id, 3, None, "unknown" ) )
    assigned = persist.upsertClipTrack( conn, persist.ClipTrackDB( clip.id, 3, person.id, "home_player" ) )
    tracks = persist.listClipTracks( conn, clip.id )

    assert unknown.person_id is None
    assert assigned.person_id == person.id
    assert tracks[ 3 ] == assigned
    assert len( tracks ) == 1


class TestClipHomography:

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_save_clip_homography_creates_then_updates_locked_off_record( self, conn ):
    video = persist.upsertVideo( conn, persist.Video( id=0, file="homography.mp4" ) )
    match = persist.upsertMatch( conn, persist.Match( id=0, date=None, home="A", away="B", division="D" ) )
    camera = persist.upsertCamera( conn, persist.Camera( id=0, name="homography-camera" ) )
    clip = persist.upsertClip( conn, persist.ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 ) )
    first_payload = { "homography": [[1, 0], [0, 1]], "points": { "image": [], "world": [] } }
    saved = persist.saveClipHomography( conn, clip.id, first_payload )
    updated_payload = { **first_payload, "revision": 2 }
    updated = persist.saveClipHomography( conn, clip.id, updated_payload, homography_id=saved.homography_id )
    loaded = persist.getClipHomography( conn, clip.id )

    assert saved.homography_id == updated.homography_id
    assert loaded is not None
    assert loaded.payload == updated_payload
    assert loaded.frame_start is None
    assert loaded.frame_end is None

  @pytest.mark.usefixtures( "clear_test_database" )
  def test_save_clip_homography_requires_complete_frame_range( self, conn ):
    with pytest.raises( ValueError, match="both start and end" ):
      persist.saveClipHomography( conn, 1, {}, frame_start=0 )
