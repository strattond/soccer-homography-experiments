from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import cv2
from PIL import Image, ImageDraw

from soccer_homography.data import ParticipationRole, Point2D, Track, heatmap
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
  heatmap_points: tuple[ Point2D, ... ]


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
      heatmap_points: list[ Point2D ] = []
      for track in self.tracks.values():
        index = track.getListIndex( frame_number )
        if index is None or index >= len( track.homog ) or index >= len( track.homog_smooth ):
          continue
        role = track.roleAt( frame_number, self.roles_by_person )
        smooth_point = track.homog_smooth[ index ]
        markers.extend( (
            HomographyMarker( role, track.homog[ index ] ),
            HomographyMarker( role, smooth_point ),
        ) )
        heatmap_points.append( smooth_point )
      yield HomographyFrame( frame_number, tuple( markers ), tuple( heatmap_points ) )


class HomographyFrameRenderer:

  def __init__(
      self,
      pitch: SoccerPitchImage,
      width: int,
      height: int,
      include_heatmap: bool = True,
  ) -> None:
    self.pitch = pitch
    self.width = width
    self.height = height
    self.include_heatmap = include_heatmap
    self.scale_width, self.scale_length = pitch.get_pitch_scale
    self.heatmap = heatmap(
        1.0,
        int( pitch.cfg.STD_PITCH_WIDTH ),
        int( pitch.cfg.STD_PITCH_LENGTH ),
    )
    self.heatmap_width = max( 1, width - 20 )
    self.heatmap_height = max( 1, height - 20 )
    self.heatmap_offset = ( 10, 10 )

  def render( self, frame: HomographyFrame ) -> Image.Image:
    image = Image.fromarray( self.pitch.empty ).crop( ( 0, 0, self.width, self.height ) )
    image = image.convert( "RGBA" )
    if self.include_heatmap:
      for point in frame.heatmap_points:
        self.heatmap.accumulate( int( point.x ), int( point.y ) )
      heatmap_image = self.heatmap.get_display_image()
      heatmap_image = cv2.resize(
          heatmap_image,
          ( self.heatmap_width, self.heatmap_height ),
          interpolation=cv2.INTER_CUBIC,
      )
      overlay = Image.fromarray( heatmap_image )
      image.alpha_composite( overlay, self.heatmap_offset )
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
    return image.convert( "RGB" )
