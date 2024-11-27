"""cuGAL graph alignment algorithm.

Use `cugal` to run the algorithm.
"""

from .config import Config, SinkhornMethod, HungarianMethod  # noqa: F401
from .pred import cugal  # noqa: F401
