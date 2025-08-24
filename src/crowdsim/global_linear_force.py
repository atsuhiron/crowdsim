import numpy as np

from crowdsim.typings import Dir, DirList, FieldGenerator


class GlobalLinearForce(FieldGenerator):
    def __init__(
        self,
        group_idx: int,
        vec: Dir,
    ) -> None:
        self.group_idx = group_idx
        self.vec = vec

    def force(self, _positions: DirList) -> DirList:
        return self.vec[np.newaxis, :]
