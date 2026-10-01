from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_homography.dataTypes import ParticipationRole


@dataclass
class IdentificationImageResult:
  frame_number: int = -1
  role: ParticipationRole | None = None
  confidence: float | None = None

@dataclass
class IdentificationResult:
  role: ParticipationRole | None = None
  confidence: float | None = None


class AbstractInferenceModel:

  @abstractmethod
  def query( self, image: np.ndarray, prompt: str ) -> IdentificationResult:
    pass

  @abstractmethod
  def loadModel( self ) -> Any:
    pass

  @abstractmethod
  def processResponse( self, response: IdentificationResult, frame_number: int ) -> IdentificationImageResult:
    pass

def selectModelDevice( torch_module: Any, minBytes ) -> str:
  if torch_module.cuda.is_available():
    free_memory, _total_memory = torch_module.cuda.mem_get_info( 0 )
    if free_memory >= minBytes:
      return "cuda:0"
  return "cpu"
