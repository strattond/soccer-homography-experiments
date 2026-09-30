import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soccer_homography.dataTypes import ParticipationRole
from soccer_homography.log import logger

VLM_MODEL_NAME = "vikhyatk/moondream2"
MIN_VLM_GPU_MEMORY_BYTES = 6 * 1024**3
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MIN_CLIP_GPU_MEMORY_BYTES = 1 * 1024**3
IDENTIFICATION_PROMPT_PATH = Path( __file__ ).resolve().parents[ 4 ] / "identificationPrompt.txt"
DEFAULT_IDENTIFICATION_PROMPT = (
    "Identify the role of the person in the crop.\n"
    "A home player is wearing a red and white striped shirt; an away player is wearing the opposing team's kit.\n"
    "A goalkeeper may wear a distinct goalkeeper kit.\n"
    "A referee is wearing a bright yellow shirt, or a black shirt and holding a flag."
)
ROLE_PROMPT_SUFFIX = (
    "\nChoose the single best role and respond with exactly one of these labels: "
    "home_player, home_goalkeeper, away_player, away_goalkeeper, referee, unknown."
)

RoleGuess = ParticipationRole | None


@dataclass( frozen=True, slots=True )
class VLMResponse:
  answer: str


@dataclass( frozen=True, slots=True )
class VLMImageResult:
  frame_number: int
  answer: str
  role: RoleGuess


@dataclass( frozen=True, slots=True )
class ClipResponse:
  role: ParticipationRole
  confidence: float


@dataclass( frozen=True, slots=True )
class ClipImageResult:
  frame_number: int
  role: ParticipationRole
  confidence: float


IdentificationImageResult = VLMImageResult | ClipImageResult


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
  answers: list[ IdentificationImageResult ] | None = None
  role: RoleGuess = None
  vote_counts: dict[ ParticipationRole, int ] | None = None
  image_result: IdentificationImageResult | None = None
  error: Exception | None = None


class MoondreamVLM:

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
    return VLMResponse( str( answer ) )

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


CLIP_ROLES: tuple[ ParticipationRole, ...] = (
    "home_player",
    "home_goalkeeper",
    "away_player",
    "away_goalkeeper",
    "referee",
    "unknown",
)
CLIP_ROLE_LABELS: dict[ ParticipationRole, str ] = {
    "home_player": "a soccer home-team outfield player wearing the home team's kit",
    "home_goalkeeper": "a soccer home-team goalkeeper wearing a goalkeeper kit",
    "away_player": "a soccer away-team outfield player wearing the away team's kit",
    "away_goalkeeper": "a soccer away-team goalkeeper wearing a goalkeeper kit",
    "referee": "a soccer referee wearing a referee uniform",
    "unknown": "a person whose role in a soccer match cannot be identified",
}


def selectClipDevice( torch_module: Any ) -> str:
  if torch_module.cuda.is_available():
    free_memory, _total_memory = torch_module.cuda.mem_get_info( 0 )
    if free_memory >= MIN_CLIP_GPU_MEMORY_BYTES:
      return "cuda:0"
  return "cpu"


class ClipRoleClassifier:

  def __init__( self ) -> None:
    self.model: Any = None
    self.processor: Any = None
    self.device = "cpu"
    self.loadLock = threading.Lock()

  def query( self, image: np.ndarray, _prompt: str = "" ) -> ClipResponse:
    model, processor, device = self.loadModel()
    from PIL import Image
    import torch

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
          from transformers import CLIPModel, CLIPProcessor
          import torch

          self.device = selectClipDevice( torch )
          logger.info( f"Loading CLIP role classifier on {self.device}." )
          self.model = CLIPModel.from_pretrained(
              CLIP_MODEL_NAME,
              device_map={ "": self.device },
          )
          self.processor = CLIPProcessor.from_pretrained( CLIP_MODEL_NAME )
    return self.model, self.processor, self.device


class VLMInferenceWorker( threading.Thread ):

  def __init__(
      self,
      generation: int,
      clip_id: int,
      track_id: int,
      crops: list[ tuple[ int, np.ndarray ] ],
      prompt: str,
      model: MoondreamVLM | ClipRoleClassifier,
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
    answers: list[ IdentificationImageResult ] = []
    try:
      for completed, ( frame_number, crop ) in enumerate( self.crops, start=1 ):
        if self.cancelEvent.is_set():
          return
        response = self.model.query( crop, self.prompt )
        if isinstance( response, ClipResponse ):
          image_result: IdentificationImageResult = ClipImageResult(
              frame_number,
              response.role,
              response.confidence,
          )
        else:
          image_result = VLMImageResult(
              frame_number,
              response.answer,
              guessRole( [ response.answer ] ),
          )
        answers.append( image_result )
        self.results.put(
            VLMJobMessage(
                self.generation,
                "answer",
                self.track_id,
                self.clip_id,
                completed,
                len( self.crops ),
                image_result=image_result,
            )
        )
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
        vote_counts = roleVoteCounts( answers )
        self.results.put(
            VLMJobMessage(
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
        self.results.put(
            VLMJobMessage( self.generation, "error", self.track_id, self.clip_id, error=error )
        )


def roleVoteCounts( results: list[ IdentificationImageResult ] ) -> dict[ ParticipationRole, int ]:
  counts: dict[ ParticipationRole, int ] = { role: 0 for role in CLIP_ROLES }
  for result in results:
    if result.role is not None:
      counts[ result.role ] += 1
  return counts


def mostLikelyRole( counts: dict[ ParticipationRole, int ] ) -> RoleGuess:
  max_count = max( counts.values(), default=0 )
  if max_count == 0:
    return None
  winners: list[ ParticipationRole ] = [ role for role, count in counts.items() if count == max_count ]
  return winners[ 0 ] if len( winners ) == 1 else None


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
