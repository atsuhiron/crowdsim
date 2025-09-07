import numpy as np

from crowdsim.typings import Dir, DirList, FieldGenerator


class Particle(FieldGenerator):
    def __init__(
        self,
        group_idx: int,
        mass: float,
        fric_coef: float,
        pos: Dir,
        vel: Dir,
    ) -> None:
        self.group_idx = group_idx
        self.mass = mass
        self.fric_coef = fric_coef
        self.pos = pos
        self.vel = vel

    def force(self, positions: DirList) -> DirList:
        rel = positions - self.pos[np.newaxis, :]
        distance = np.linalg.norm(rel, axis=1)
        repulsion = rel / (distance**3)

        friction = -1 * self.vel * self.fric_coef
        return repulsion + friction
