# pyright: reportMissingTypeArgument=false
import numpy as np
from jaxtyping import Float, Int

class TriangleBvh:
    def __init__(self, triangle_vertices: Float[np.ndarray, "num_triangles 9"]) -> None: ...

    @property
    def num_triangles(self) -> int: ...

    def register(self) -> int: ...
    def unregister(self) -> None: ...

    def nearest_hit(
        self,
        ray_origins: Float[np.ndarray, "num_rays 3"],
        ray_directions: Float[np.ndarray, "num_rays 3"],
        active_mask: np.ndarray | None = None,
    ) -> tuple[Int[np.ndarray, " num_rays"], Float[np.ndarray, " num_rays"]]: ...

    def get_candidates(
        self,
        ray_origins: Float[np.ndarray, "num_rays 3"],
        ray_directions: Float[np.ndarray, "num_rays 3"],
        expansion: float = 0.0,
        max_candidates: int = 256,
    ) -> tuple[
        Int[np.ndarray, "num_rays max_candidates"],
        Int[np.ndarray, " num_rays"],
    ]: ...

def bvh_nearest_hit_capsule() -> object: ...
def bvh_get_candidates_capsule() -> object: ...
