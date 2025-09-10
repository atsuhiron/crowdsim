import numpy as np

from crowdsim.typings import Dir, DirList, FieldGenerator


class Aisle(FieldGenerator):
    def __init__(self, center: Dir, direction: Dir, width: float, amp: float) -> None:
        self.center = center
        self.direction = direction
        self.a = 2 * amp * (width ** (-2))

    def force(self, positions: DirList) -> DirList:
        forces = np.empty(shape=positions.shape)
        for i, position in enumerate(positions):
            x_c, y_c, s = _calc_cross_point(
                u=float(self.direction[0]),
                v=float(self.direction[1]),
                x0=float(self.center[0]),
                y0=float(self.center[1]),
                x1=position[0],
                y1=position[1],
            )
            forces[i] = np.array([x_c - position[0], y_c - position[1]]) * self.a
            # norm_vec_to_cp = np.array([x_c - position[0], y_c - position[1]]) / s
            # coef = self.a * s
        return forces


def _calc_cross_point(u: float, v: float, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float]:
    dx = x1 - x0
    dy = y1 - y0
    sq_norm = v**2 + u**2
    t = (v * dy + u * dx) / sq_norm
    s = (v * dx - u * dy) / sq_norm
    return x0 + t * u, y0 + t * v, s
