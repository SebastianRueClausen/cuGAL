"""cuGAL graph alignment algorithm.

Use `cugal` to run the algorithm.
"""

from .config import Config, SinkhornMethod  # noqa: F401
from .sinkhorn import FixedInit, PrevInit, SelectiveInit  # noqa: F401
from .pred import cugal  # noqa: F401
