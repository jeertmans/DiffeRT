__all__ = ("ArrayType",)

from typing import TypeVar

import numpy as np
from jaxtyping import Array

# NOTE: we declare this is another module to be able to patch it when building the docs
ArrayType = TypeVar("ArrayType", bound=Array | np.ndarray)
