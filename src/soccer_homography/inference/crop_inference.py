import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soccer_homography.data import ParticipationRole
from soccer_homography.inference.abstractions import (
  AbstractInferenceModel,
  IdentificationImageResult,
)
from soccer_homography.inference.clip_model import CLIP_ROLES

IDENTIFICATION_PROMPT_PATH = Path( __file__ ).resolve().parents[ 4 ] / "identificationPrompt.txt"
DEFAULT_IDENTIFICATION_PROMPT = (
    "Identify the role of the person in the crop.\n"
    "A home player is wearing a red and white striped shirt; an away player is wearing the opposing team's kit.\n"
    "A goalkeeper may wear a distinct goalkeeper kit.\n"
    "A referee is wearing a bright yellow shirt, or a black shirt and holding a flag."
)
ROLE_PROMPT_SUFFIX = ( "\nChoose the single best role and respond with exactly one of these labels: "
                       "home_player, home_goalkeeper, away_player, away_goalkeeper, referee, unknown." )

def loadIdentificationPrompt( path: Path = IDENTIFICATION_PROMPT_PATH ) -> str:
  try:
    return path.read_text( encoding="utf-8" ).strip() or DEFAULT_IDENTIFICATION_PROMPT
  except FileNotFoundError:
    return DEFAULT_IDENTIFICATION_PROMPT


def saveIdentificationPrompt( prompt: str, path: Path = IDENTIFICATION_PROMPT_PATH ) -> None:
  prompt = prompt.strip()
  if not prompt:
    raise ValueError( "Identification prompt cannot be empty." )
  path.write_text( prompt + "\n", encoding="utf-8" )


@dataclass( slots=True )
class CropInferenceJobMessage:
  generation: int
  kind: str
  track_id: int
  clip_id: int
  completed: int = 0
  total: int = 0
  answers: list[ IdentificationImageResult ] | None = None
  role: ParticipationRole | None = None
  vote_counts: dict[ ParticipationRole, int ] | None = None
  image_result: IdentificationImageResult | None = None
  error: Exception | None = None


class CropInferenceWorker( threading.Thread ):

  def __init__(
      self,
      generation: int,
      clip_id: int,
      track_id: int,
      crops: list[ tuple[ int, np.ndarray ] ],
      prompt: str,
      model: AbstractInferenceModel,
      results: queue.Queue[ CropInferenceJobMessage ],
  ) -> None:
    super().__init__( daemon=True, name=f"track-vlm-{clip_id}-{track_id}" )
    self.generation = generation
    self.clip_id = clip_id
    self.track_id = track_id
    self.crops = crops
    self.prompt = f"{prompt.strip()}{ROLE_PROMPT_SUFFIX}"
    self.model = model
    self.results = results
    self.cancelEvent = threading.Event()

  def cancel( self ) -> None:
    self.cancelEvent.set()

  def run( self ) -> None:
    answers: list[ IdentificationImageResult ] = []
    try:
      for completed, ( frame_number, crop ) in enumerate( self.crops, start=1 ):
        if self.cancelEvent.is_set():
          return
        response = self.model.query( crop, self.prompt )
        image_result = self.model.processResponse( response, frame_number )
        answers.append( image_result )
        self.results.put( CropInferenceJobMessage(
            self.generation,
            "answer",
            self.track_id,
            self.clip_id,
            completed,
            len( self.crops ),
            image_result=image_result,
        ) )
        self.results.put( CropInferenceJobMessage(
            self.generation,
            "progress",
            self.track_id,
            self.clip_id,
            completed,
            len( self.crops ),
        ) )
      if not self.cancelEvent.is_set():
        vote_counts = roleVoteCounts( answers )
        self.results.put(
            CropInferenceJobMessage(
                generation=self.generation,
                kind="done",
                track_id=self.track_id,
                clip_id=self.clip_id,
                completed=len( answers ),
                total=len( self.crops ),
                answers=answers,
                role=mostLikelyRole( vote_counts ),
                vote_counts=vote_counts,
            )
        )
    except Exception as error:
      if not self.cancelEvent.is_set():
        self.results.put( CropInferenceJobMessage( self.generation, "error", self.track_id, self.clip_id, error=error ) )


def roleVoteCounts( results: list[ IdentificationImageResult ] ) -> dict[ ParticipationRole, int ]:
  counts: dict[ ParticipationRole, int ] = { role: 0 for role in CLIP_ROLES }
  for result in results:
    if result.role is not None:
      counts[ result.role ] += 1
  return counts


def mostLikelyRole( counts: dict[ ParticipationRole, int ] ) -> ParticipationRole | None:
  max_count = max( counts.values(), default=0 )
  if max_count == 0:
    return None
  winners: list[ ParticipationRole ] = [ role for role, count in counts.items() if count == max_count ]
  return winners[ 0 ] if len( winners ) == 1 else None


