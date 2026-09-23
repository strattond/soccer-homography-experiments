    @pytest.mark.usefixtures("clean_db")
    def test_upsert_clip_returns_updated_object(self, conn):
        """Test that upsertClip returns the updated ClipDB object with correct foreign keys."""

        # Create dependencies
        video = persist.Video( id=0, file="test.mp4" )
        match = persist.Match( id=0, date="2026-09-15", home="A", away="B", division="D" )
        camera = persist.Camera( id=0, name="test-cam" )

        persist.upsertVideo( conn, video )
        persist.upsertMatch( conn, match )
        persist.upsertCamera( conn, camera )

        clip = ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 )
        result = persist.upsertClip( conn, clip )

        assert isinstance( result, persist.ClipDB ), "Should return a ClipDB instance"
        assert result.video_id == 4, "video_id should be updated in the returned object"
