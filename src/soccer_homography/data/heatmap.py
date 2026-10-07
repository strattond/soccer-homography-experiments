from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class heatmap:
  cellSize: float
  height: int
  width: int
  data: np.ndarray = field( init=False )

  def __init__( self, cellSize: float, height: int, width: int ):
    self.cellSize = cellSize
    self.height = int( height / cellSize )
    self.width = int( width / cellSize )
    self.reset()

  def reset( self ):
    self.data = np.zeros( ( self.height, self.width ), dtype=np.float32 )

  def accumulate( self, x: int, y: int, value: float = 1.0 ):
    #print( f"Accumulating heatmap at ({x}, {y}) with value {value}" )
    cx = int( x / self.cellSize )
    cy = int( y / self.cellSize )
    if 0 <= cx < self.width and 0 <= cy < self.height:
      self.data[ cy, cx ] += value

  def get_display_image( self, label="" ) -> np.ndarray:

    if np.max( self.data ) <= 0:
      transparent = np.zeros( ( int( self.width / self.cellSize ), int( self.height / self.cellSize ), 4 ), dtype=np.uint8 )
      return transparent

    blurred = cv2.GaussianBlur( self.data, ( 0, 0 ), sigmaX=3, sigmaY=3 )
    sqrtView = np.sqrt( blurred )
    normalized_data = ( sqrtView - np.min( sqrtView ) ) / ( np.max( sqrtView ) - np.min( sqrtView ) + 1e-5 )
    alpha = normalized_data.copy()
    cutoff = np.percentile( alpha[ alpha > 0 ], 10 )
    alpha = np.maximum( alpha - cutoff, 0 )
    alpha /= alpha.max() + 1e-5
    alpha = ( normalized_data * 180 ).astype( np.uint8 )
    heatmap_image = ( normalized_data * 255 ).astype( np.uint8 )
    heatmap_image = cv2.applyColorMap( heatmap_image, cv2.COLORMAP_TURBO )
    rgba = cv2.cvtColor( heatmap_image, cv2.COLOR_BGR2RGBA )
    rgba[ :, :, 3 ] = alpha
    return heatmap_image
