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
ROLE_PROMPTS: dict[ ParticipationRole, list[ str ] ] = {
    "home_player": [
        "a soccer player wearing a red and white striped jersey",
        "a football player wearing red and white vertical stripes",
        "a soccer player in a red and white uniform",
    ],
    "home_goalkeeper": [
        "a soccer goalkeeper for the home team",
        "a football goalkeeper wearing goalkeeper gloves",
        "a home team goalkeeper",
    ],
    "away_player": [
        "a soccer player on the away team",
        "a football player wearing the away team kit",
        "an opposing soccer player",
    ],
    "away_goalkeeper": [
        "a soccer goalkeeper for the away team",
        "a football goalkeeper on the opposing team",
        "an away team goalkeeper",
    ],
    "referee": [
        "a soccer referee",
        "a football referee",
        "a soccer referee wearing a bright yellow shirt",
        "a soccer referee holding a yellow flag",
        "a football assistant referee holding a yellow flag",
    ],
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

    prompt_roles: list[ ParticipationRole ] = []
    prompts: list[ str ] = []
    for role, role_prompts in ROLE_PROMPTS.items():
      for role_prompt in role_prompts:
        prompt_roles.append( role )
        prompts.append( role_prompt )

    inputs = processor(
        text=prompts,
        images=Image.fromarray( image ),
        return_tensors="pt",
        padding=True,
    )
    inputs = { name: value.to( device ) for name, value in inputs.items() }
    with torch.inference_mode():
      prompt_scores = model( **inputs ).logits_per_image[ 0 ].softmax( dim=-1 )

    role_scores: dict[ ParticipationRole, Any ] = {}
    for role in ROLE_PROMPTS:
      indices = [ index for index, prompt_role in enumerate( prompt_roles ) if prompt_role == role ]
      prompt_score = prompt_scores[ indices ]
      role_scores[ role ] = prompt_score.mean()
      for index in indices:
        print( f"CLIP prompt score [{role}] {prompts[index]!r}: {prompt_scores[index].item():.4f}" )

    total_role_score = sum( role_scores.values() )
    role_confidences: dict[ ParticipationRole, Any ] = {
        role: score / total_role_score
        for role, score in role_scores.items()
    }
    for role, confidence in role_confidences.items():
      print( f"CLIP role aggregate [{role}]: {confidence.item():.4f}" )

    best_role = max( role_confidences.items(), key=lambda item: item[ 1 ] )[ 0 ]
    confidence = float( role_confidences[ best_role ].item() )
    role: ParticipationRole = best_role if confidence > 0.5 else "unknown"
    return ClipResponse( role, confidence )

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
