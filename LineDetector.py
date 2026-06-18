import cv2
import numpy as np
from cv2.typing import MatLike

from appState import ImageOptions


class LineDetector:

  def __init__( self, imgOpts: ImageOptions ) -> None:
    self.imgOpts = imgOpts

  def getSkyMask( self, image: MatLike ):
    _, binary = cv2.threshold( image, 0, 255, cv2.THRESH_OTSU )
    contours, _ = cv2.findContours( binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    if contours:
      sky_contour = max( contours, key=cv2.contourArea )
      sky_mask_clean = np.zeros_like( image )
      cv2.drawContours( sky_mask_clean, [ sky_contour ], -1, 255, thickness=cv2.FILLED )
      return sky_mask_clean
    else:
      return binary

  def blurIt( self, image: MatLike ):
    return cv2.bilateralFilter( image, d=9, sigmaColor=75, sigmaSpace=75 )

  def enhanceIt( self, image: MatLike ):
    clahe = cv2.createCLAHE( clipLimit=2.0, tileGridSize=( 8, 8 ) )
    return clahe.apply( image )

  def getHoughEdges( self, image: MatLike ):
    return cv2.Canny( image, 50, 150 )

  def getHoughLines( self, edges ):
    return cv2.HoughLinesP( edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10 )

  def getScharrEdges( self, image: MatLike ):
    grad_x = cv2.Scharr( image, cv2.CV_32F, 1, 0 )
    grad_y = cv2.Scharr( image, cv2.CV_32F, 0, 1 )
    mag = cv2.magnitude( grad_x, grad_y )
    mag = cv2.convertScaleAbs( mag )
    smooth = cv2.normalize( mag, mag, 0, 255, cv2.NORM_MINMAX )
    return cv2.adaptiveThreshold( smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -5 )

  def rebuildEdges( self, edges ):
    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 9, 9 ) )
    return cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )

  def drawOnMask( self, lines, image: MatLike ):
    mask = np.zeros_like( image, dtype=np.uint8 )
    for line in lines:
      x1, y1, x2, y2 = map( int, line[ 0 ] )
      cv2.line( mask, ( x1, y1 ), ( x2, y2 ), 255, 2 )
    return mask

  def getLines( self, gray ):
    imgOpts = self.imgOpts
    if imgOpts.removeSky:
      skyMask = self.getSkyMask( gray )
      gray = cv2.bitwise_and( gray, gray, mask=cv2.bitwise_not( skyMask ) )
    if imgOpts.preBlur:
      gray = self.blurIt( gray )
    if imgOpts.edgeEnhance:
      gray = self.enhanceIt( gray )

    match imgOpts.edgeType:
      case 'Canny':
        edges = self.getHoughEdges( gray )
      case 'Scharr':
        edges = self.getScharrEdges( gray )
      case _:
        edges = self.getHoughEdges( gray )

    lines = self.processForLines( edges, gray )
    #
    if imgOpts.closeEdges:
      masked = self.drawOnMask( lines, gray )
      edges = self.rebuildEdges( masked )
      lines = self.processForLines( edges, masked )

    return lines

  def processForLines( self, edges, gray ):
    match self.imgOpts.lineType:
      case 'Hough':
        lines = self.getHoughLines( edges )
      case 'LineSegmentDetector':
        lsd = cv2.createLineSegmentDetector( refine=cv2.LSD_REFINE_STD )
        lines, _, _, _ = lsd.detect( gray )
      case _:
        lines = self.getHoughLines( edges )
    return lines
