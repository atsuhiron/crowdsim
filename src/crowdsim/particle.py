import numpy as np

from crowdsim.typings import Dir, DirList, FieldGenerator


class Particle(FieldGenerator):
    def __init__(
        self,
        group_idx: int,
        mass: float,
        pos: Dir,
        vel: Dir,
    ) -> None:
        self.group_idx = group_idx
        self.mass = mass
        self.pos = pos
        self.vel = vel

    def force(self, positions: DirList) -> DirList:
        rel = positions - self.pos[np.newaxis, :]
        distance = np.linalg.norm(rel, axis=1)

        return rel / (distance**3)
