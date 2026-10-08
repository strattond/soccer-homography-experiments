import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_homography.data import ParticipationRole
from soccer_homography.inference.abstractions import (
    AbstractInferenceModel,
    IdentificationImageResult,
    IdentificationResult,
    selectModelDevice,
)
from soccer_homography.log import logger

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

MIN_CLIP_GPU_MEMORY_BYTES = 1 * 1024**3

CLIP_ROLES: tuple[ ParticipationRole, ...] = (
    "home_player",
    "home_goalkeeper",
    "away_player",
    "away_goalkeeper",
    "referee",
    "unknown",
)
CLIP_ROLE_LABELS: dict[ ParticipationRole, str ] = {
    "home_player": "a soccer outfield player wearing the red and white striped kit",
    "home_goalkeeper": "a soccer home-team goalkeeper wearing a goalkeeper kit",
    "away_player": "a soccer outfield player wearing a kit that is not red and white striped",
    "away_goalkeeper": "a soccer away-team goalkeeper wearing a goalkeeper kit",
    "referee": "a soccer referee wearing a bright yellow shirt, or a black shirt holding a yellow flag",
    "unknown": "a person whose role in a soccer match cannot be identified",
}


@dataclass( slots=True )
class ClipResponse( IdentificationResult ):
  pass


@dataclass( slots=True )
class ClipImageResult( IdentificationImageResult ):
  pass


class ClipRoleClassifier( AbstractInferenceModel ):

  def __init__( self ) -> None:
    self.model: Any = None
    self.processor: Any = None
    self.device = "cpu"
    self.loadLock = threading.Lock()

  def query( self, image: np.ndarray, prompt: str = "" ) -> ClipResponse:
    model, processor, device = self.loadModel()
    import torch
    from PIL import Image

    inputs = processor(
        text=[ CLIP_ROLE_LABELS[ role ] for role in CLIP_ROLES ],
        images=Image.fromarray( image ),
        return_tensors="pt",
        padding=True,
    )
    inputs = { name: value.to( device ) for name, value in inputs.items() }
    with torch.inference_mode():
      scores = model( **inputs ).logits_per_image[ 0 ].softmax( dim=-1 )
    confidence, index = scores.max( dim=-1 )
    role = CLIP_ROLES[ int( index.item() ) ]
    return ClipResponse( role, float( confidence.item() ) )

  def loadModel( self ) -> tuple[ Any, Any, str ]:
    if self.model is None or self.processor is None:
      with self.loadLock:
        if self.model is None or self.processor is None:
          import torch
          from transformers import CLIPModel, CLIPProcessor

          self.device = selectModelDevice( torch, MIN_CLIP_GPU_MEMORY_BYTES )
          logger.info( f"Loading CLIP role classifier on {self.device}." )
          self.model = CLIPModel.from_pretrained(
              CLIP_MODEL_NAME,
              device_map={ "": self.device},
          )
          self.processor = CLIPProcessor.from_pretrained( CLIP_MODEL_NAME )
    return self.model, self.processor, self.device

  def processResponse( self, response: IdentificationResult, frame_number: int ) -> IdentificationImageResult:
    image_result: IdentificationImageResult = ClipImageResult(
        frame_number,
        response.role,
        response.confidence,
    )
    return image_result
