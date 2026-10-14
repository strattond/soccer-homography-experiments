from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from PIL import Image, ImageDraw

from soccer_homography.data import ParticipationRole, Point2D, Track
from soccer_homography.pitch import SoccerPitchImage

role_colors: dict[ ParticipationRole, str ] = {
  "home_player":        "#ff0000",
  "away_player":        "#0000ff",
  "home_goalkeeper":    "#00ff00",
  "away_goalkeeper":    "#66ccff",
  "unknown":            "#000000",
  "referee":            "#ffffff",
}


@dataclass(frozen=True, slots=True)
class HomographyMarker:
  role: ParticipationRole
  point: Point2D


@dataclass(frozen=True, slots=True)
class HomographyFrame:
  frame_number: int
  markers: tuple[ HomographyMarker, ... ]


class HomographyFrameSource:

  def __init__(
      self,
      tracks: Mapping[ int, Track ],
      roles_by_person: Mapping[ int, ParticipationRole ],
  ) -> None:
    self.tracks = tracks
    self.roles_by_person = dict( roles_by_person )

  def iterFrames( self, start: int, end: int ) -> Iterator[ HomographyFrame ]:
    for frame_number in range( start, end ):
      markers: list[ HomographyMarker ] = []
      for track in self.tracks.values():
        index = track.getListIndex( frame_number )
        if index is None or index >= len( track.homog ) or index >= len( track.homog_smooth ):
          continue
        role = track.roleAt( frame_number, self.roles_by_person )
        markers.extend( (
            HomographyMarker( role, track.homog[ index ] ),
            HomographyMarker( role, track.homog_smooth[ index ] ),
        ) )
      yield HomographyFrame( frame_number, tuple( markers ) )


class HomographyFrameRenderer:

  def __init__( self, pitch: SoccerPitchImage, width: int, height: int ) -> None:
    self.pitch = pitch
    self.width = width
    self.height = height
    self.scale_width, self.scale_length = pitch.get_pitch_scale

  def render(
      self,
      frame: HomographyFrame,
      heatmap: Image.Image | None = None,
      heatmap_offset: tuple[ int, int ] = ( 0, 0 ),
  ) -> Image.Image:
    image = Image.fromarray( self.pitch.empty ).crop( ( 0, 0, self.width, self.height ) )
    if heatmap is not None:
      image.paste( heatmap, heatmap_offset )
    draw = ImageDraw.Draw( image )
    radius = 6
    for marker in frame.markers:
      x = int( marker.point.x * self.scale_width + self.pitch.padding )
      y = int( marker.point.y * self.scale_length + self.pitch.padding )
      draw.ellipse(
          ( x - radius, y - radius, x + radius, y + radius ),
          fill=role_colors.get( marker.role, "#000000" ),
          outline="black",
          width=1,
      )
    return image
