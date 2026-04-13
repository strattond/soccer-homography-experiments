import cv2
import logging
import os
import numpy as np
from PIL import Image
from cv2.typing import MatLike

def save_image( output_path, image, filename ):
  out_path = os.path.join( output_path, filename )
  if isinstance(image, Image.Image):
    image.save( out_path )
    logging.info( f"Saved {out_path}" )
  else:
    if not cv2.imwrite( out_path, image ):
      logging.info( f"Failed saving {out_path}")
    else:
      logging.info( f"Saved {out_path}" )

def fit_to_alpha( source_image: MatLike, alpha_source: MatLike ) -> MatLike:

  # Resize and retro the alpha channel 
  result_rgba = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGRA)
  # Ensure result matches original crop size
  resized_result = cv2.resize( result_rgba, (alpha_source.shape[1], alpha_source.shape[0]), interpolation=cv2.INTER_LANCZOS4 )

  # Now alpha channel assignment works
  resized_result[:, :, 3] = alpha_source[:, :, 3]
  return resized_result

def crop_source_by_mask( src_img, refined_mask, box ):
  # Create BGRA copy from in‑memory original
  rgba = cv2.cvtColor(src_img, cv2.COLOR_BGR2BGRA)
  rgba[:, :, 3] = refined_mask

  # Optional crop
  x1, y1, x2, y2 = map(int, box)
  cropped_rgba = rgba[y1:y2, x1:x2]
  return cropped_rgba

def edge_boost_clahe( img_bgr, clip_limit=2.0, tile_grid_size=(8,8) ):
  lab = cv2.cvtColor( img_bgr, cv2.COLOR_BGR2LAB )
  l, a, b = cv2.split( lab )
  clahe = cv2.createCLAHE( clipLimit=clip_limit, tileGridSize=tile_grid_size )
  l_clahe = clahe.apply( l )
  lab_clahe = cv2.merge( ( l_clahe, a, b ) )
  return cv2.cvtColor( lab_clahe, cv2.COLOR_LAB2BGR )

def edge_boost_unsharp( img_bgr, blur_ksize = 5, amount = 1.5 ):
  # Blur the image
  blurred = cv2.GaussianBlur( img_bgr, ( blur_ksize, blur_ksize ), 0 )
  # Weighted sum: original * (1+amount) - blurred * amount
  sharpened = cv2.addWeighted( img_bgr, 1 + amount, blurred, -amount, 0 )
  return sharpened

def polygon_to_mask( polygons, shape ):
  mask = np.zeros(shape, dtype=np.uint8)
  for poly in polygons:
      pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
      cv2.fillPoly(mask, [pts], 255)
  return mask

def flood_fill_background( src_img ):
  # Split RGB and alpha
  rgb = src_img[:, :, :3]
  alpha = src_img[:, :, 3]

  # Create a white background
  background = np.ones_like(rgb, dtype=np.uint8) * 0

  # Composite: where alpha > 0, use rgb; else use background
  rgb_filled = np.where(alpha[:, :, None] > 0, rgb, background)
  return rgb_filled

def prep_image_for_transform( src_img, filename, output_path ):

  rgb_filled = flood_fill_background( src_img )
  # Convert to PIL for the pipeline
  rval = Image.fromarray( cv2.cvtColor( rgb_filled, cv2.COLOR_BGR2RGB ) )
  save_image( output_path, rval, filename )
  return rval

