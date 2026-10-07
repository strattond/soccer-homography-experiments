from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class heatmap:
  height: int
  width: int
  data: np.ndarray = field( init=False )

  def __init__( self, height: int, width: int ):
    self.height = height
    self.width = width
    self.reset()

  def reset( self ):
    self.data = np.zeros( ( self.height, self.width ), dtype=np.float32 )

  def accumulate( self, x: int, y: int, value: float = 1.0 ):
    if 0 <= x < self.width and 0 <= y < self.height:
      self.data[ y, x ] += value

  def get_display_image( self ) -> np.ndarray:
    normalized_data = ( self.data - np.min( self.data ) ) / ( np.max( self.data ) - np.min( self.data ) + 1e-5 )
    heatmap_image = ( normalized_data * 255 ).astype( np.uint8 )
    heatmap_image = cv2.applyColorMap( heatmap_image, cv2.COLORMAP_JET )
    heatmap_image = cv2.GaussianBlur( heatmap_image, (0, 0), sigmaX=5, sigmaY=5 )
    return heatmap_image