import queue
import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from soccer_homography.dataTypes import ParticipationRole
from soccer_homography.log import logger

VLM_MODEL_NAME = "vikhyatk/moondream2"
MIN_VLM_GPU_MEMORY_BYTES = 6 * 1024**3
ROLE_PROMPT_SUFFIX = (
    "\nChoose the single best role and respond with exactly one of these labels: "
    "home_player, home_goalkeeper, away_player, away_goalkeeper, referee, unknown."
)

RoleGuess = ParticipationRole | None


def selectVLMDeviceMap( torch_module: Any ) -> dict[ str, str ]:
  if torch_module.cuda.is_available():
    free_memory, _total_memory = torch_module.cuda.mem_get_info( 0 )
    if free_memory >= MIN_VLM_GPU_MEMORY_BYTES:
      return { "": "cuda:0" }
  return { "": "cpu" }


@dataclass( slots=True )
class VLMJobMessage:
  generation: int
  kind: str
  track_id: int
  clip_id: int
  completed: int = 0
  total: int = 0
  answers: list[ tuple[ int, str ] ] | None = None
  role: RoleGuess = None
  error: Exception | None = None


class MoondreamVLM:

  def __init__( self ) -> None:
    self.model: Any = None
    self.loadLock = threading.Lock()

  def query( self, image: np.ndarray, prompt: str ) -> str:
    model = self.loadModel()
    from PIL import Image

    result = model.query( Image.fromarray( image ), prompt )
    if isinstance( result, dict ):
      answer = result.get( "answer", result )
    else:
      answer = result
    return str( answer )

  def loadModel( self ) -> Any:
    if self.model is None:
      with self.loadLock:
        if self.model is None:
          from transformers import AutoModelForCausalLM
          import torch

          device_map = selectVLMDeviceMap( torch )
          logger.info( f"Loading Moondream on {device_map['']} to keep model tensors on a single device." )
          self.model = AutoModelForCausalLM.from_pretrained(
              VLM_MODEL_NAME,
              trust_remote_code=True,
              device_map=device_map,
          )
    return self.model


class VLMInferenceWorker( threading.Thread ):

  def __init__(
      self,
      generation: int,
      clip_id: int,
      track_id: int,
      crops: list[ tuple[ int, np.ndarray ] ],
      prompt: str,
      model: MoondreamVLM,
      results: queue.Queue[ VLMJobMessage ],
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
    answers: list[ tuple[ int, str ] ] = []
    try:
      for completed, ( frame_number, crop ) in enumerate( self.crops, start=1 ):
        if self.cancelEvent.is_set():
          return
        answer = self.model.query( crop, self.prompt )
        answers.append( ( frame_number, answer ) )
        self.results.put(
            VLMJobMessage(
                self.generation,
                "progress",
                self.track_id,
                self.clip_id,
                completed,
                len( self.crops ),
            )
        )
      if not self.cancelEvent.is_set():
        self.results.put(
            VLMJobMessage(
                self.generation,
                "done",
                self.track_id,
                self.clip_id,
                len( answers ),
                len( self.crops ),
                answers,
                guessRole( [ answer for _frame_number, answer in answers ] ),
            )
        )
    except Exception as error:
      if not self.cancelEvent.is_set():
        self.results.put(
            VLMJobMessage( self.generation, "error", self.track_id, self.clip_id, error=error )
        )


def guessRole( answers: list[ str ] ) -> RoleGuess:
  role_patterns: tuple[ tuple[ ParticipationRole, str ], ... ] = (
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
    matches: list[ ParticipationRole ] = [
        role
        for role, pattern in role_patterns
        if re.search( pattern, normalized )
    ]
    if len( matches ) == 1:
      counts[ matches[ 0 ] ] += 1
  max_count = max( counts.values(), default=0 )
  if max_count == 0:
    return None
  winners: list[ ParticipationRole ] = [ role for role, count in counts.items() if count == max_count ]
  return winners[ 0 ] if len( winners ) == 1 else None
