import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_homography.dataTypes import ParticipationRole, roles
from soccer_homography.inference.abstractions import (
    AbstractInferenceModel,
    IdentificationImageResult,
    IdentificationResult,
    selectModelDevice,
)
from soccer_homography.log import logger

VLM_MODEL_NAME = "vikhyatk/moondream2"
MIN_VLM_GPU_MEMORY_BYTES = 6 * 1024**3


@dataclass( slots=True )
class VLMResponse( IdentificationResult ):
  answer: str = ""


@dataclass( slots=True )
class VLMImageResult( IdentificationImageResult ):
  answer: str = ""


class MoondreamVLM( AbstractInferenceModel ):

  def __init__( self ) -> None:
    self.model: Any = None
    self.loadLock = threading.Lock()

  def query( self, image: np.ndarray, prompt: str ) -> VLMResponse:
    model = self.loadModel()
    from PIL import Image

    result = model.query( Image.fromarray( image ), prompt )
    if isinstance( result, dict ):
      answer = result.get( "answer", result )
    else:
      answer = result
    toRet: ParticipationRole | None = next( ( role for role in roles if role == answer ), None )
    return VLMResponse( toRet, None, answer )

  def loadModel( self ) -> Any:
    if self.model is None:
      with self.loadLock:
        if self.model is None:
          import torch
          from transformers import AutoModelForCausalLM

          device_map = selectModelDevice( torch, MIN_VLM_GPU_MEMORY_BYTES )
          logger.info( f"Loading Moondream on {device_map} to keep model tensors on a single device." )
          self.model = AutoModelForCausalLM.from_pretrained(
              VLM_MODEL_NAME,
              trust_remote_code=True,
              device_map=device_map,
          )
    return self.model

  def processResponse( self, response: IdentificationResult, frame_number: int ) -> IdentificationImageResult:
    if isinstance( response, VLMResponse ):
      image_result: IdentificationImageResult = VLMImageResult(
          frame_number,
          response.role or guessRole( [ response.answer ] ),
          response.confidence,
          response.answer,
      )
    else:
      image_result = VLMImageResult( frame_number, response.role, response.confidence, "" )
    return image_result


def guessRole( answers: list[ str ] ) -> ParticipationRole | None:
  role_patterns: tuple[ tuple[ ParticipationRole, str ], ...] = (
      ( "home_goalkeeper", r"\bhome[\s_-]+goalkeeper\b|\bgoalkeeper[\s_-]+for[\s_-]+home\b" ),
      ( "away_goalkeeper", r"\baway[\s_-]+goalkeeper\b|\bgoalkeeper[\s_-]+for[\s_-]+away\b" ),
      ( "home_player", r"\bhome[\s_-]+player\b" ),
      ( "away_player", r"\baway[\s_-]+player\b" ),
      ( "referee", r"\breferee\b" ),
      ( "unknown", r"\bunknown\b" ),
  )
  counts: dict[ ParticipationRole, int ] = { role: 0 for role, _pattern in role_patterns }
  for answer in answers:
    normalized = answer.lower()
    matches: list[ ParticipationRole ] = [ role for role, pattern in role_patterns if re.search( pattern, normalized ) ]
    if len( matches ) == 1:
      counts[ matches[ 0 ] ] += 1
  max_count = max( counts.values(), default=0 )
  if max_count == 0:
    return None
  winners: list[ ParticipationRole ] = [ role for role, count in counts.items() if count == max_count ]
  return winners[ 0 ] if len( winners ) == 1 else None
