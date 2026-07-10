import abc
import cv2
import numpy as np
from PIL import Image


class BaseVideoEncoder( abc.ABC ):
  """
    Abstract base class for video encoders.
    """

  @abc.abstractmethod
  def save( self, source_capture: cv2.VideoCapture, frames: list[ Image.Image ] ) -> None:
    """
        Encode a video using the provided source capture and raw PIL frames.
        Returns MP4 bytes in memory.
        """
    pass


class Mp4Encoder( BaseVideoEncoder ):
  """
    Concrete MP4 encoder using OpenCV.
    """

  @staticmethod
  def pil_to_cv( img: Image.Image ) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR ndarray."""
    arr = np.array( img )
    return cv2.cvtColor( arr, cv2.COLOR_RGB2BGR )

  def save( self, source: cv2.VideoCapture, frames: list[ Image.Image ] ) -> None:
    """
        Encode frames into an MP4 entirely in memory.
        The source_capture is provided for metadata (fps, size, etc).
        """
    if not frames:
      raise ValueError( "No frames provided to encoder." )

    width = int( source.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    height = int( source.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    fps = int( source.get( cv2.CAP_PROP_FPS ) )

    # Configure MP4 writer
    fourcc = cv2.VideoWriter.fourcc( *"mp4v" )
    writer = cv2.VideoWriter( "saved_homography.mp4", fourcc, fps, ( width, height ) )

    # Write frames
    for frame in frames:
      writer.write( self.pil_to_cv( frame ) )

    writer.release()


class GifEncoder( BaseVideoEncoder ):
  """
    In-memory GIF encoder using Pillow.
    """

  def save( self, source: cv2.VideoCapture, frames: list[ Image.Image ] ) -> None:

    if not frames:
      raise ValueError( "No frames provided to encoder." )

    fps = int( source.get( cv2.CAP_PROP_FPS ) )
    duration = 1000 / fps

    frames[ 0 ].save(
        "saved_homography.gif",
        format="GIF",
        save_all=True,
        append_images=frames[ 1: ],
        duration=duration,
        loop=1,
        optimize=False,
    )
