from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SelectionPoint:
  index: int = None
  coords: Tuple[int, int] = field( default_factory=lambda: (None, None) )