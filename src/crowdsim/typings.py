import abc

import numpy as np
import numpy.typing as npt

type Dir = npt.NDArray[np.float64]  # (2,)
type DirList = npt.NDArray[np.float64]  # (N, 2)


class FieldGenerator(abc.ABC):
    @abc.abstractmethod
    def force(self, positions: DirList) -> DirList:
        pass
