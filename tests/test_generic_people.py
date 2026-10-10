from collections import Counter
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from soccer_homography import App
from soccer_homography.db import (
  Camera,
  ClipDB,
  Match,
  Video,
  listClipParticipants,
  upsertCamera,
  upsertClip,
  upsertMatch,
  upsertVideo,
)


def test_add_generic_people_to_current_clip_assigns_them_to_clip_match( conn, monkeypatch ):
  video = upsertVideo( conn, Video( id=0, file="generic-people.mp4" ) )
  match = upsertMatch( conn, Match( id=0, date=None, home="Home", away="Away", division="D" ) )
  camera = upsertCamera( conn, Camera( id=0, name="Generic test camera" ) )
  clip = upsertClip( conn, ClipDB( id=0, video_id=video.id, match_id=match.id, camera_id=camera.id, sequence=1 ) )
  app = cast( Any, App.__new__( App ) )
  app.root = Mock()
  app.appState = SimpleNamespace( db=conn, curClipID=clip.id )
  app.tabData = SimpleNamespace( tabClipParticipants=Mock() )
  showinfo = Mock()
  monkeypatch.setattr( "soccer_homography.messagebox.showinfo", showinfo )

  app.addGenericPeopleToCurrentClip()

  participants = listClipParticipants( conn, clip.id )
  assert len( participants ) == 25
  assert Counter( participant.role for participant in participants ) == {
      "referee": 3,
      "away_goalkeeper": 2,
      "away_player": 20,
  }
  assert all( participant.is_placeholder for participant in participants )
  app.tabData.tabClipParticipants.refresh.assert_called_once_with()
  showinfo.assert_called_once()
